from __future__ import annotations
import copy
from pathlib import Path
from typing import TYPE_CHECKING
from lark import Lark, LarkError
from fpy.bytecode.directives import Directive
from fpy.codegen_fpybc import (
    AssignFrameOffsets,
    FinalChecks,
    FpybcBackendState,
    GenerateFunctionEntryPoints,
    GenerateFunctions,
    GenerateSequence,
    IrPass,
    ResolveLabels,
)
from fpy.desugaring import (
    DesugarAnonExprs,
    DesugarAugmentedAssignments,
    DesugarDefaultArgs,
    DesugarForLoops,
    DesugarCheckStatements,
    DesugarTimeOperators,
)
from fpy.semantics import (
    AssignIds,
    AssignNameGroups,
    CreateScopes,
    CheckResolvedSymbolKinds,
    CheckAllUnqualifiedIdentifiersResolved,
    CheckAnonStructMembers,
    CheckAssignSyntax,
    CheckSequenceMetadataDefinedAtTop,
    CalculateConstExprValues,
    CalculateDefaultArgConstValues,
    CheckBreakAndContinueInLoop,
    CheckConstArrayAccesses,
    CheckFunctionReturns,
    CheckReturnInFunc,
    CheckUseBeforeDefine,
    CollectFunctionGlobalUses,
    CollectUsedFunctions,
    ResolveTransitiveGlobalUses,
    CheckGlobalsInitializedBeforeCall,
    CheckSequenceArgs,
    DefineFunctions,
    DefineVariables,
    PickTypesAndResolveFields,
    ResolveQualifiedIdentifiers,
    ResolveSequenceDependencies,
    CheckForConstantSizeTypes,
    UpdateStateWithTypes,
    WarnRangesAreNotEmpty,
)
from fpy.imports import (
    BindImports,
    ConstructAst,
    WarnImportUnderscore,
)
from fpy.syntax import AstBlock, FpyTransformer, PythonIndenter
from fpy.types import (
    DEFAULT_MAX_DIRECTIVE_SIZE,
    DEFAULT_MAX_DIRECTIVES_COUNT,
    SPECIFIC_NUMERIC_TYPES,
    BLOCK_STATE,
    CHECK_STATE,
    CMD_RESPONSE,
    FLAGS_TYPE,
    LOG_SEVERITY,
    SEQ_ARGS,
    TIME_COMPARISON,
    TIME_INTERVAL,
    TIME_BASE,
    FpyType,
)
from fpy.state import (
    CompileState,
)
from fpy.visitors import Visitor

from fpy.error import BackendError, handle_lark_error
import fpy.error

if TYPE_CHECKING:
    from llvmlite import ir

# Load grammar once at module level
_fpy_grammar_path = Path(__file__).parent / "grammar.lark"
_fpy_grammar_str = _fpy_grammar_path.read_text(encoding="utf-8")

# Create parser once at module level with LALR and cache enabled.
# PythonIndenter.process() resets its internal state on each call,
# so it's safe to reuse the same parser instance.
_fpy_indenter = PythonIndenter()
_fpy_parser = Lark(
    _fpy_grammar_str,
    start="input",
    parser="lalr",
    postlex=_fpy_indenter,
    propagate_positions=True,
    maybe_placeholders=True,
)


def _parse_fpy(text: str, **kwargs):
    """Parse fpy source into a Lark tree, tolerating a missing final newline.

    The indenter (PythonIndenter) derives INDENT/DEDENT tokens from the
    whitespace each _NEWLINE token carries. When the source does not end in a
    newline, the final line's leading whitespace is never followed by a newline,
    so at end-of-input it is misread as a brand-new indentation level: a trailing
    tab or an over-indented comment on the last line then emits a spurious INDENT
    (or a bogus dedent) and an otherwise-valid sequence fails to compile
    (https://github.com/fprime-community/fpy/issues/61).

    Appending a newline when one is absent makes the last line's indentation
    resolve to column 0 -- closing any open blocks cleanly -- exactly as
    CPython's tokenizer supplies an implicit NEWLINE before end-of-input. The
    newline is added only at the very end, so it shifts no positions and leaves
    error line/column reporting (and text.splitlines()) unchanged.
    """
    if not text.endswith("\n"):
        text = text + "\n"
    return _fpy_parser.parse(text, **kwargs)


# Load builtin time.fpy functions at module level
_builtin_time_path = Path(__file__).parent / "builtin" / "time.fpy"
_builtin_time_text = _builtin_time_path.read_text(encoding="utf-8")
_builtin_library_ast = None  # Lazily initialized


def _get_builtin_library_ast():
    """Parse and cache the builtin library AST."""
    global _builtin_library_ast
    if _builtin_library_ast is None:
        # Save current error state
        old_input_text = fpy.error.input_text
        old_input_lines = fpy.error.input_lines
        old_file_name = fpy.error.file_name

        fpy.error.file_name = str(_builtin_time_path)
        fpy.error.input_text = _builtin_time_text
        fpy.error.input_lines = _builtin_time_text.splitlines()

        tree = _parse_fpy(_builtin_time_text)
        _builtin_library_ast = FpyTransformer().transform(tree)

        # Restore error state
        fpy.error.input_text = old_input_text
        fpy.error.input_lines = old_input_lines
        fpy.error.file_name = old_file_name

    return _builtin_library_ast


def _build_root_block(program: AstBlock, state: CompileState):
    """Wrap the program in a fresh *library root* block, stored as
    state.root_block, and put the imported sequence blocks as sibling blocks:

        library root (state.root_block)     scope = base
        |- builtin library defs             (time_add, time_cmp, ...)
        |- main block (state.main_block)     scope = child of base
        |- imported sequence block           scope = child of base
        |- ..."""
    library_ast = _get_builtin_library_ast()

    # The program block, as parsed, is the main block; the main sequence's
    # context points at it so BindImports can reach its scope.
    state.main_block = program
    state.main_sequence.block = program

    state.root_block = AstBlock(
        program.meta,
        copy.deepcopy(library_ast.stmts) + [program] + state.imported_blocks,
    )


def text_to_ast(text: str) -> AstBlock:
    """Lex, parse and transform fpy source into an AST block.

    Raises CompileError on failure."""
    from lark.exceptions import VisitError

    fpy.error.input_text = text
    fpy.error.input_lines = text.splitlines()
    try:
        tree = _parse_fpy(text, on_error=handle_lark_error)
    except LarkError as e:
        handle_lark_error(e)
    try:
        transformed = FpyTransformer().transform(tree)
    except RecursionError:
        raise fpy.error.CompileError(
            "Maximum recursion depth exceeded (code is too deeply nested)"
        )
    except VisitError as e:
        # VisitError wraps exceptions that occur during tree transformation
        if isinstance(e.orig_exc, RecursionError):
            raise fpy.error.CompileError(
                "Maximum recursion depth exceeded (code is too deeply nested)"
            )
        elif isinstance(e.orig_exc, fpy.error.SyntaxErrorDuringTransform):
            raise fpy.error.CompileError(e.orig_exc.msg, e.orig_exc.node)
        else:
            raise fpy.error.CompileError(f"Internal error during parsing: {e.orig_exc}")
    return transformed


def analyze_ast(body: AstBlock, state: CompileState) -> CompileState:
    """Run the shared, backend-independent front end on an AST.

    Returns the populated CompileState. Raises the first CompileError encountered.
    """
    # Constructing the AST (SPEC.md Imports): resolve every import statement, on
    # the raw program AST, including each imported sequence file's block in
    # state.imported_blocks and recording a ResolvedImport for BindImports.
    ConstructAst().run(body, state)
    if len(state.errors) != 0:
        raise state.errors[0]

    # Wrap the program in the library root block: builtin library, the main
    # program (state.main_block), and every imported sequence as sibling
    # children. All later passes run on state.root_block.
    _build_root_block(body, state)

    passes: list[Visitor] = [
        DesugarCheckStatements(),
        DesugarAugmentedAssignments(),
        # sequence() metadata, if present, must be the first statement of
        # each sequence block
        CheckSequenceMetadataDefinedAtTop(),
        # assign each node a unique id for indexing/hashing
        AssignIds(),
        # based on position of node in tree, figure out which scope it is in
        CreateScopes(),
        # check that assignment targets are valid
        CheckAssignSyntax(),
        # check that no anonymous struct names a member twice
        CheckAnonStructMembers(),
        # register all user-defined functions in the global callable scope
        # (and the builtin library functions in the shared base callable scope)
        DefineFunctions(),
        # register all variable declarations in their enclosing scopes.
        # Function bodies are deferred so that globals declared later in
        # the source are visible inside functions.
        DefineVariables(),
        # Binding (SPEC.md Imports): now that every sequence's definitions are
        # registered, each import statement associates one or more qualified
        # names with definitions in the importing scope
        BindImports(),
        # check that break/continue are in loops, and store which loop they're in
        CheckBreakAndContinueInLoop(),
        CheckReturnInFunc(),
        # record each expression's name group (callable/type/value) from its
        # syntactic slot, so resolution and kind-checking can read it back
        AssignNameGroups(),
        ResolveQualifiedIdentifiers(),
        CheckAllUnqualifiedIdentifiersResolved(),
        # warn when the importer uses an underscore-prefixed imported definition
        WarnImportUnderscore(),
        CheckResolvedSymbolKinds(),
        CheckForConstantSizeTypes(),
        UpdateStateWithTypes(),
        # make sure we don't use any variables before they are declared
        CheckUseBeforeDefine(),
        # record the globals each function reads and the functions it calls...
        CollectFunctionGlobalUses(),
        # ...then grow those to the transitive closure over the call graph...
        ResolveTransitiveGlobalUses(),
        # ...so we can check globals are initialized before any function that
        # reads them (directly or transitively) is called
        CheckGlobalsInitializedBeforeCall(),
        # discover sequence-run dependencies (the targets' .fpy sources)
        # before type checking
        ResolveSequenceDependencies(),
        # this pass resolves all attributes and items, as well as determines the type of expressions
        PickTypesAndResolveFields(),
        # now that every anonymous struct/array has been given a type, turn
        # each into a call of that type's constructor (and reject any that
        # were not given a type), so no later pass sees an anonymous expr
        DesugarAnonExprs(),
        # Calculate const values for default arguments first (and check they're const).
        # This must happen before CalculateConstExprValues because call sites may
        # reference functions defined later in the source, and we need the default
        # values' const values to be available.
        CalculateDefaultArgConstValues(),
        # okay, now that we're sure we're passing in all the right args to each func,
        # we can calculate values of type ctors etc etc
        CalculateConstExprValues(),
        CheckFunctionReturns(),
        CheckConstArrayAccesses(),
        WarnRangesAreNotEmpty(),
        CheckSequenceArgs(),
        # now that semantic analysis is done, we can desugar things.
        # Fill in default arguments before desugaring for loops
        DesugarDefaultArgs(),
        # Desugar time operators before for loops (time ops may be in loop conditions)
        DesugarTimeOperators(),
        DesugarForLoops(),
        # Collect which functions are reachable through calls from the main
        # sequence. Runs after desugaring because desugared time operators
        # call script functions.
        CollectUsedFunctions(),
    ]
    for compile_pass in passes:
        compile_pass.run(state.root_block, state)
        if len(state.errors) != 0:
            raise state.errors[0]

    return state


def analysis_to_fpybc_directives(
    state: CompileState,
) -> tuple[list[Directive], list[FpyType]]:
    """Runs fpybc codegen passes on analysis results, returning fpybc directives.

    Raises BackendError on failure."""
    state.backend = FpybcBackendState()
    codegen_passes = [
        # Assign variable offsets before generating function bodies
        # so global variable offsets are known when referenced in functions
        AssignFrameOffsets(),
        GenerateFunctionEntryPoints(),
        # generate all function bodies
        GenerateFunctions(),
    ]
    for compile_pass in codegen_passes:
        compile_pass.run(state.root_block, state)
        if len(state.errors) != 0:
            raise state.errors[0]

    ir = GenerateSequence().emit(state.main_block, state)

    ir_passes: list[IrPass] = [ResolveLabels(), FinalChecks()]
    for compile_pass in ir_passes:
        ir = compile_pass.run(ir, state)
        if isinstance(ir, BackendError):
            # early exit on errors
            raise ir

    # print out warnings
    for warning in state.warnings:
        print(warning)

    # all the ir is guaranteed to have been converted to directives by now by FinalChecks
    return ir, state.this_seq_arg_specs


def analysis_to_llvm_module(
    state: CompileState,
) -> tuple[ir.Module, list[FpyType]]:
    """Runs LLVM codegen passes on analysis results, returning an llvmlite ir.Module (the LLVM backend).

    Raises BackendError on failure."""
    # Imported here, not at module scope: the LLVM backend is an optional
    # install, and importing it raises BackendError when it isn't present.
    from fpy.codegen_llvm import GenerateLlvmModule

    module = GenerateLlvmModule().emit(state.root_block, state)

    # print out warnings
    for warning in state.warnings:
        print(warning)

    return module, state.this_seq_arg_specs


def analysis_to_wasm(
    state: CompileState,
) -> tuple[bytes, list[FpyType]]:
    """Runs the LLVM backend and lowers the result to a runnable wasm module.

    Raises BackendError on failure."""
    from fpy.codegen_llvm import llvm_module_to_wasm

    module, seq_arg_types = analysis_to_llvm_module(state)
    return llvm_module_to_wasm(module), seq_arg_types


def analysis_to_wat(
    state: CompileState,
) -> tuple[str, list[FpyType]]:
    """Runs the LLVM backend and lowers the result to WebAssembly text.

    Raises BackendError on failure."""
    from fpy.codegen_llvm import llvm_module_to_wasm_text

    module, seq_arg_types = analysis_to_llvm_module(state)
    return llvm_module_to_wasm_text(module), seq_arg_types
