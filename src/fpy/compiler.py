from __future__ import annotations
import sys
from functools import lru_cache
from pathlib import Path
from lark import Lark, LarkError
from fpy.bytecode.directives import Directive
from fpy.codegen import (
    CalculateFrameSizes,
    CollectUsedFunctions,
    FinalChecks,
    GenerateFunctionEntryPoints,
    GenerateFunctions,
    GenerateModule,
    IrPass,
    ResolveLabels,
)
from fpy.desugaring import DesugarDefaultArgs, DesugarForLoops, DesugarCheckStatements, DesugarTimeOperators
from fpy.dictionary import load_dictionary, json_default_to_fpy_value
from fpy.semantics import (
    AssignIds,
    CreateScopes,
    CalculateConstExprValues,
    CalculateDefaultArgConstValues,
    CheckBreakAndContinueInLoop,
    CheckConstArrayAccesses,
    CheckFunctionReturns,
    CheckReturnInFunc,
    CheckUseBeforeDefine,
    CreateVariablesAndFuncs,
    PickTypesAndResolveFields,
    ResolveQualifiedNames,
    UpdateTypesAndFuncs,
    WarnRangesAreNotEmpty,
)
from fpy.syntax import AstBlock, FpyTransformer, PythonIndenter
from fpy.macros import MACROS
from fpy.types import (
    DEFAULT_MAX_DIRECTIVE_SIZE,
    DEFAULT_MAX_DIRECTIVES_COUNT,
    SPECIFIC_NUMERIC_TYPES,
    CHECK_STATE,
    CMD_RESPONSE,
    FLAG_ID,
    TIME_COMPARISON,
    TIME_INTERVAL,
    FpyType,
    FpyValue,
    TypeKind,
    TIME,
    BOOL,
    I64,
)
from fpy.state import (
    CallableSymbol,
    CastSymbol,
    CommandSymbol,
    CompileState,
    TypeCtorSymbol,
    create_symbol_table,
    merge_symbol_tables,
)
from fpy.visitors import Visitor

from fpy.error import BackendError, CompileError, handle_lark_error
import fpy.error

# Load grammar once at module level
_fpy_grammar_path = Path(__file__).parent / "grammar.lark"
_fpy_grammar_str = _fpy_grammar_path.read_text()

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
_builtin_time_text = _builtin_time_path.read_text()
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
        print(
            fpy.error.CompileError(
                "Maximum recursion depth exceeded (code is too deeply nested)"
            ),
            file=sys.stderr,
        )
        exit(1)
    except VisitError as e:
        # VisitError wraps exceptions that occur during tree transformation
        if isinstance(e.orig_exc, RecursionError):
            print(
                fpy.error.CompileError(
                    "Maximum recursion depth exceeded (code is too deeply nested)"
                ),
                file=sys.stderr,
            )
        elif isinstance(e.orig_exc, fpy.error.SyntaxErrorDuringTransform):
            print(
                fpy.error.CompileError(e.orig_exc.msg, e.orig_exc.node),
                file=sys.stderr,
            )
        else:
            print(
                fpy.error.CompileError(f"Internal error during parsing: {e.orig_exc}"),
                file=sys.stderr,
            )
        exit(1)
    return transformed



def _validate_and_replace_type(
    type_dict: dict[str, FpyType],
    name: str,
    canonical: FpyType,
) -> None:
    """Validate that a required type exists in the dictionary and matches the
    canonical definition, then replace it with the canonical version."""
    if name not in type_dict:
        raise ValueError(f"Dictionary must contain {name} type")
    dict_type = type_dict[name]
    if dict_type.kind != canonical.kind:
        raise ValueError(
            f"Dictionary {name} has kind {dict_type.kind}, expected {canonical.kind}"
        )
    if canonical.kind == TypeKind.STRUCT:
        if dict_type.members != canonical.members:
            raise ValueError(
                f"Dictionary {name} has members {dict_type.members}, "
                f"expected {canonical.members}"
            )
    elif canonical.kind == TypeKind.ENUM:
        if dict_type.enum_dict != canonical.enum_dict:
            raise ValueError(
                f"Dictionary {name} has enum dict {dict_type.enum_dict}, "
                f"expected {canonical.enum_dict}"
            )
        if dict_type.rep_type != canonical.rep_type:
            raise ValueError(
                f"Dictionary {name} has rep type {dict_type.rep_type}, "
                f"expected {canonical.rep_type}"
            )
    elif canonical.kind == TypeKind.ARRAY:
        if dict_type.elem_type != canonical.elem_type:
            raise ValueError(
                f"Dictionary {name} has elem type {dict_type.elem_type}, "
                f"expected {canonical.elem_type}"
            )
        if dict_type.length != canonical.length:
            raise ValueError(
                f"Dictionary {name} has length {dict_type.length}, "
                f"expected {canonical.length}"
            )
    type_dict[name] = canonical


def _make_type_ctor(name: str, typ: FpyType) -> TypeCtorSymbol | None:
    """Create a TypeCtorSymbol for a type, or return None if it has no callable ctor.

    For structs, inline member arrays (members with a "size" key in the JSON)
    get wrapped in a synthetic array type, but the dictionary's raw default is
    shaped for the *inner* element type, not the wrapper.  This is an FPP bug —
    the default should be nested to match the wrapper shape.  We detect the
    shape mismatch and skip those members' defaults until FPP is fixed.
    """
    if typ.kind == TypeKind.STRUCT:
        struct_defaults: dict[str, FpyValue] = {}
        if typ.default is not None:
            for m in typ.members:
                raw_val = typ.default.get(m.name)
                if raw_val is None:
                    continue
                if m.type.kind == TypeKind.ARRAY and (
                    not isinstance(raw_val, list) or len(raw_val) != m.type.length
                ):
                    continue
                struct_defaults[m.name] = json_default_to_fpy_value(raw_val, m.type)
        args = [(m.name, m.type, struct_defaults.get(m.name)) for m in typ.members]
    elif typ.kind == TypeKind.ARRAY:
        array_defaults: list[FpyValue] = []
        if typ.default is not None:
            default_val = json_default_to_fpy_value(typ.default, typ)
            array_defaults = default_val.val  # list of FpyValue
        args = [
            ("e" + str(i), typ.elem_type, array_defaults[i] if i < len(array_defaults) else None)
            for i in range(typ.length)
        ]
    else:
        return None
    return TypeCtorSymbol(name, typ, args, typ)


@lru_cache(maxsize=4)
def _build_global_scopes(dictionary: str) -> tuple:
    """
    Build and cache the 3 global scopes for a dictionary.
    Returns tuple of (type_scope, callable_scope, values_scope, sequence_config).
    """
    d = load_dictionary(dictionary)
    cmd_name_dict = d["cmd_name_dict"]
    ch_name_dict = d["ch_name_dict"]
    prm_name_dict = d["prm_name_dict"]
    dict_type_name_dict = d["type_defs"]

    # Validate required dictionary types
    _validate_and_replace_type(dict_type_name_dict, "Fw.TimeIntervalValue", TIME_INTERVAL)
    _validate_and_replace_type(dict_type_name_dict, "Svc.Fpy.FlagId", FLAG_ID)
    _validate_and_replace_type(dict_type_name_dict, "Fw.CmdResponse", CMD_RESPONSE)
    _validate_and_replace_type(dict_type_name_dict, "Fw.TimeComparison", TIME_COMPARISON)

    # Build the full type dict: start from (now-validated) dictionary types,
    # then layer on builtins and internal types.  Later entries win, so
    # canonical replacements from _validate_and_replace_type are preserved.
    type_name_dict: dict[str, FpyType] = {
        **dict_type_name_dict,
        "Fw.Time": TIME,
        **{typ.name: typ for typ in SPECIFIC_NUMERIC_TYPES},
        "bool": BOOL,
        "$CheckState": CHECK_STATE,
    }

    # Collect enum constants from the final type dict (after builtins and
    # canonical replacements are in place).
    enum_const_name_dict: dict[str, FpyValue] = {}
    for name, typ in type_name_dict.items():
        if typ.kind == TypeKind.ENUM:
            for enum_const_name in typ.enum_dict:
                enum_const_name_dict[name + "." + enum_const_name] = FpyValue(
                    typ, enum_const_name
                )

    # Build callable dict: commands, numeric casts, type constructors, macros
    callable_name_dict: dict[str, CallableSymbol] = {}

    for name, cmd in cmd_name_dict.items():
        args = [(arg_name, arg_type, None) for arg_name, _, arg_type in cmd.arguments]
        callable_name_dict[name] = CommandSymbol(
            cmd.name, CMD_RESPONSE, args, cmd
        )

    for typ in SPECIFIC_NUMERIC_TYPES:
        callable_name_dict[typ.name] = CastSymbol(
            typ.name, typ, [("value", I64, None)], typ
        )

    for name, typ in type_name_dict.items():
        ctor = _make_type_ctor(name, typ)
        if ctor is not None:
            callable_name_dict[name] = ctor

    for macro_name, macro in MACROS.items():
        callable_name_dict[macro_name] = macro

    # Build the 3 global scopes per SPEC:
    # 1. global type scope - leaf nodes are types
    type_scope = create_symbol_table(type_name_dict)
    # 2. global callable scope - leaf nodes are callables
    callable_scope = create_symbol_table(callable_name_dict)
    # 3. global value scope - leaf nodes are values (tlm channels, parameters, enum constants)
    values_scope = merge_symbol_tables(
        create_symbol_table(ch_name_dict),
        merge_symbol_tables(
            create_symbol_table(prm_name_dict),
            create_symbol_table(enum_const_name_dict),
        ),
    )

    return (type_scope, callable_scope, values_scope)


def get_base_compile_state(dictionary: str, compile_args: dict) -> CompileState:
    """return the initial state of the compiler, based on the given dict path"""
    type_scope, callable_scope, values_scope = _build_global_scopes(dictionary)
    constants = load_dictionary(dictionary)["constants"]

    # Make copies of the scopes since we'll mutate them during compilation
    # (e.g., adding user-defined functions to callable_scope, variables to values_scope)
    # if we don't make copies, then the lru cache will return the modified versions, causing
    # two runs of the compiler to conflict
    state = CompileState(
        global_type_scope=type_scope,  # types are not mutated
        global_callable_scope=callable_scope.copy(),
        global_value_scope=values_scope.copy(),
        compile_args=compile_args or dict(),
        max_directives_count=constants.get("Svc.Fpy.MAX_SEQUENCE_STATEMENT_COUNT", DEFAULT_MAX_DIRECTIVES_COUNT),
        max_directive_size=constants.get("Svc.Fpy.MAX_DIRECTIVE_SIZE", DEFAULT_MAX_DIRECTIVE_SIZE),
    )
    return state


def ast_to_directives(
    body: AstBlock,
    dictionary: str,
    compile_args: dict | None = None,
) -> list[Directive] | CompileError | BackendError:
    compile_args = compile_args or dict()
    
    # Prepend builtin library functions to user code - always available.
    # will be elided if unused
    import copy
    builtin_library_ast = _get_builtin_library_ast()
    body.stmts = copy.deepcopy(builtin_library_ast.stmts) + body.stmts
    
    state = get_base_compile_state(dictionary, compile_args)
    state.root = body

    pre_semantic_desugaring_passes = [
        DesugarCheckStatements()
    ]
    
    semantics_passes: list[Visitor] = [
        # assign each node a unique id for indexing/hashing
        AssignIds(),
        # based on position of node in tree, figure out which scope it is in
        CreateScopes(),
        # based on assignment syntax nodes, we know which variables exist where.
        # Function bodies are deferred so that globals declared later in
        # the source are visible inside functions.
        CreateVariablesAndFuncs(),
        # check that break/continue are in loops, and store which loop they're in
        CheckBreakAndContinueInLoop(),
        CheckReturnInFunc(),
        ResolveQualifiedNames(),
        UpdateTypesAndFuncs(),
        # make sure we don't use any variables before they are declared
        CheckUseBeforeDefine(),
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
    ]
    desugaring_passes: list[Visitor] = [
        # Fill in default arguments before desugaring for loops
        DesugarDefaultArgs(),
        # Desugar time operators before for loops (time ops may be in loop conditions)
        DesugarTimeOperators(),
        # now that semantic analysis is done, we can desugar things. start with for loops
        DesugarForLoops(),
    ]
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
    module_generator = GenerateModule()

    ir_passes: list[IrPass] = [ResolveLabels(), FinalChecks()]

    for compile_pass in pre_semantic_desugaring_passes:
        compile_pass.run(body, state)
        if len(state.errors) != 0:
            return state.errors[0]

    for compile_pass in semantics_passes:
        compile_pass.run(body, state)
        if len(state.errors) != 0:
            return state.errors[0]

    for compile_pass in desugaring_passes:
        compile_pass.run(body, state)
        if len(state.errors) != 0:
            return state.errors[0]

    for compile_pass in codegen_passes:
        compile_pass.run(body, state)
        if len(state.errors) != 0:
            return state.errors[0]

    ir = module_generator.emit(body, state)

    for compile_pass in ir_passes:
        ir = compile_pass.run(ir, state)
        if isinstance(ir, BackendError):
            # early return errors
            return ir

    # print out warnings
    for warning in state.warnings:
        print(warning)

    # all the ir is guaranteed to have been converted to directives by now by FinalChecks
    return ir
