from __future__ import annotations
import copy
from pathlib import Path
from lark import Lark, LarkError
from llvmlite import ir
from fpy.bytecode.directives import Directive
from fpy.codegen_fpybc import (
    CalculateFrameSizes,
    CollectUsedFunctions,
    FinalChecks,
    GenerateFunctionEntryPoints,
    GenerateFunctions,
    GenerateModule,
    IrPass,
    ResolveLabels,
)
from fpy.codegen_llvm import (
    GenerateLlvmModule,
    llvm_module_to_wasm,
    llvm_module_to_wasm_text,
)
from fpy.desugaring import (
    DesugarDefaultArgs,
    DesugarForLoops,
    DesugarCheckStatements,
    DesugarTimeOperators,
)
from fpy.semantics import (
    AssignIds,
    CreateScopes,
    CheckAllTypesAndCallablesResolved,
    CheckAllUnqualifiedIdentifiersResolved,
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
    ResolveTransitiveGlobalUses,
    CheckGlobalsInitializedBeforeCall,
    CheckSequenceArgs,
    DefineFunctions,
    DefineVariables,
    CollectSequenceDependencies,
    PickTypesAndResolveFields,
    ResolveQualifiedIdentifiers,
    ResolveSequenceDependencies,
    CheckForConstantSizeTypes,
    UpdateStateWithTypes,
    WarnRangesAreNotEmpty,
)
from fpy.imports import (
    BindImports,
    InlineImports,
    SequenceContext,
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
from fpy.symbols import SymbolTable
from fpy.visitors import Visitor

from fpy.error import BackendError, handle_lark_error
import fpy.error

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

        tree = _fpy_parser.parse(_builtin_time_text)
        _builtin_library_ast = FpyTransformer().transform(tree)

        # Restore error state
        fpy.error.input_text = old_input_text
        fpy.error.input_lines = old_input_lines
        fpy.error.file_name = old_file_name

    return _builtin_library_ast


def _build_compilation_unit(body: AstBlock, state: CompileState):
    """Turn `body` into the outermost *library block* wrapping the whole program.

    On entry `body` holds the main program's statements; InlineImports has
    already removed each import and collected its target as a sibling block in
    state.imported_blocks. This makes `body` the library block -- the top of the
    tree, owning the base/universe callable scope -- and moves the program's own
    statements into a new *program block* nested inside it. Its children are:

        library block (body)          callable scope = base (universe)
        |- builtin library defs       (time_add, time_cmp, ...; registered in base)
        |- program block              callable scope = global (child of base)
        |- imported sequence block    callable scope = child of base
        |- ...

    Every sequence block (the program and each import) is thus a direct child of
    the library block, so its scope is a child of base by lexical nesting: syntax
    and scope agree, imported sequences are siblings of the program (isolated from
    it and each other), and the builtin library functions resolve up the parent
    chain for all of them. Only the program block executes; the library and
    imported blocks hold definitions, emitted once each by the dead-function pass.

    `body` keeps its object identity (it stays state.root), so a caller that
    passes the same AST to codegen still hands it the root."""
    library_ast = _get_builtin_library_ast()

    program_block = AstBlock(body.meta, body.stmts)
    state.program_block = program_block
    state.container_sequence[id(program_block)] = state.main_sequence

    library_ctx = SequenceContext(
        callable_scope=state.base_callable_scope,
        value_scope=SymbolTable(parent=state.base_value_scope),
        file_path=None,
        dir_path=None,
    )
    state.container_sequence[id(body)] = library_ctx

    # FIXME ideally we wouldn't have to do this switcheroo. that just
    # obfuscates what's really going on here. please consider a different
    # code flow where we can be explicit about what's going on, in
    # the compiler_main func and analyze_ast funcs esp. also, body is not a good name for this
    # variable any more, because we don't know which body we're talking about.
    # i think we should call this main_block or something
    body.stmts = (
        copy.deepcopy(library_ast.stmts) + [program_block] + state.imported_blocks
    )


def text_to_ast(text: str):
    from lark.exceptions import VisitError

    fpy.error.input_text = text
    fpy.error.input_lines = text.splitlines()
    try:
        tree = _fpy_parser.parse(text, on_error=handle_lark_error)
    except LarkError as e:
        handle_lark_error(e)
        return None
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
    # FIXME is state.root actually always the real root? don't be misleading!
    state.root = body

    # Resolve and inline every import first, on the raw AST: this splices each
    # imported sequence's top-level statements into `body` and records the
    # bindings for BindImports. It runs before everything else so later passes
    # see one combined tree.
    # FIXME why do we start by splicing statements into body??
    # shouldn't we first resolve import statements, then parse them, then
    # include their bodies under the dict/library block?
    InlineImports().run(body, state)
    if len(state.errors) != 0:
        raise state.errors[0]

    # we want to run this past first, because the next
    # stage will add statements to the start of the file
    # which would mess with this pass
    pre_builtin_lib_include_passes = [
        CheckSequenceMetadataDefinedAtTop(),
    ]

    for compile_pass in pre_builtin_lib_include_passes:
        compile_pass.run(body, state)
        if len(state.errors) != 0:
            raise state.errors[0]

    # Wrap the program in the library block: the builtin library functions and
    # every imported sequence become siblings of the program, all children of the
    # library root, so their scopes fall out of lexical nesting (see
    # _build_compilation_unit).
    _build_compilation_unit(body, state)

    pre_semantic_desugaring_passes = [DesugarCheckStatements()]

    semantics_passes: list[Visitor] = [
        # assign each node a unique id for indexing/hashing
        AssignIds(),
        # based on position of node in tree, figure out which scope it is in
        CreateScopes(),
        # check that assignment targets are valid
        CheckAssignSyntax(),
        # register all user-defined functions in the global callable scope
        # (and the builtin library functions in the shared base callable scope)
        DefineFunctions(),
        # register all variable declarations in their enclosing scopes.
        # Function bodies are deferred so that globals declared later in
        # the source are visible inside functions.
        DefineVariables(),
        # now that every sequence's definitions are registered, bind the
        # modules / names each import introduces into the importer's scopes
        BindImports(),
        # check that break/continue are in loops, and store which loop they're in
        CheckBreakAndContinueInLoop(),
        CheckReturnInFunc(),
        ResolveQualifiedIdentifiers(),
        CheckAllUnqualifiedIdentifiersResolved(),
        # warn when the importer uses an underscore-prefixed imported definition
        WarnImportUnderscore(),
        CheckAllTypesAndCallablesResolved(),
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
        # discover sequence-run dependencies (.bin files) before type checking
        ResolveSequenceDependencies(),
        # this pass resolves all attributes and items, as well as determines the type of expressions
        PickTypesAndResolveFields(),
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
    ]
    desugaring_passes: list[Visitor] = [
        # Fill in default arguments before desugaring for loops
        DesugarDefaultArgs(),
        # Desugar time operators before for loops (time ops may be in loop conditions)
        DesugarTimeOperators(),
        # now that semantic analysis is done, we can desugar things. start with for loops
        DesugarForLoops(),
    ]

    for compile_pass in pre_semantic_desugaring_passes:
        compile_pass.run(body, state)
        if len(state.errors) != 0:
            raise state.errors[0]

    for compile_pass in semantics_passes:
        compile_pass.run(body, state)
        if len(state.errors) != 0:
            raise state.errors[0]

    for compile_pass in desugaring_passes:
        compile_pass.run(body, state)
        if len(state.errors) != 0:
            raise state.errors[0]

    return state


def analysis_to_fpybc_directives(
    body: AstBlock, state: CompileState
) -> tuple[list[Directive], list[FpyType]]:
    """Runs fpybc codegen passes on analysis results, returning fpybc directives.

    Raises BackendError on failure."""
    codegen_passes = [
        # Assign variable offsets before generating function bodies
        # so global variable offsets are known when referenced in functions
        CalculateFrameSizes(),
        # Collect which functions are called anywhere in the code
        CollectUsedFunctions(),
        GenerateFunctionEntryPoints(),
        # generate all function bodies
        GenerateFunctions(),
    ]
    for compile_pass in codegen_passes:
        compile_pass.run(body, state)
        if len(state.errors) != 0:
            raise state.errors[0]

    ir = GenerateModule().emit(state.program_block, state)

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
    body: AstBlock, state: CompileState
) -> tuple[ir.Module, list[FpyType]]:
    """Runs LLVM codegen passes on analysis results, returning an llvmlite ir.Module (the LLVM backend).

    Raises BackendError on failure."""

    for compile_pass in []:
        compile_pass.run(body, state)
        if len(state.errors) != 0:
            raise state.errors[0]

    module = GenerateLlvmModule().emit(body, state)

    # print out warnings
    for warning in state.warnings:
        print(warning)

    return module, state.this_seq_arg_specs


def analysis_to_wasm(
    body: AstBlock,
    state: CompileState,
) -> tuple[bytes, list[FpyType]]:
    """Runs the LLVM backend and lowers the result to a runnable wasm module.

    Raises BackendError on failure."""
    module, seq_arg_types = analysis_to_llvm_module(body, state)
    return llvm_module_to_wasm(module), seq_arg_types


def analysis_to_wat(
    body: AstBlock,
    state: CompileState,
) -> tuple[str, list[FpyType]]:
    """Runs the LLVM backend and lowers the result to WebAssembly text.

    Raises BackendError on failure."""
    module, seq_arg_types = analysis_to_llvm_module(body, state)
    return llvm_module_to_wasm_text(module), seq_arg_types


def ast_to_dependencies(body: AstBlock, state: CompileState) -> list[str]:
    """Return the list of .bin paths that a sequence source file depends on.

    Runs only the passes needed to resolve command symbols — does not attempt
    to read the binary files, so this works before any binaries are compiled.

    Raises CompileError on failure.
    """
    state.root = body

    # Inline imports first so sequence-run dependencies in imported sequences
    # are discovered too.
    InlineImports().run(body, state)
    if state.errors:
        raise state.errors[0]

    pre_builtin_passes = [CheckSequenceMetadataDefinedAtTop()]
    for compile_pass in pre_builtin_passes:
        compile_pass.run(body, state)
        if state.errors:
            raise state.errors[0]

    _build_compilation_unit(body, state)

    discovery_passes: list[Visitor] = [
        DesugarCheckStatements(),
        AssignIds(),
        CreateScopes(),
        DefineFunctions(),
        DefineVariables(),
        BindImports(),
        ResolveQualifiedIdentifiers(),
    ]
    for compile_pass in discovery_passes:
        compile_pass.run(body, state)
        if state.errors:
            raise state.errors[0]

    discover = CollectSequenceDependencies()
    discover.run(body, state)
    if state.errors:
        raise state.errors[0]

    if state.ground_binary_dir is not None:
        return [
            str(Path(state.ground_binary_dir) / name) for name in discover.bin_names
        ]
    return discover.bin_names
