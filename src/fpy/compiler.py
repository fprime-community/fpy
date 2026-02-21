from __future__ import annotations
import json
import sys
from functools import lru_cache
from pathlib import Path
from fprime_gds.common.models.serialize.time_type import TimeType as TimeValue
from fprime_gds.common.models.serialize.bool_type import BoolType as BoolValue
from fprime_gds.common.models.serialize.enum_type import EnumType as EnumValue
from fprime_gds.common.models.serialize.serializable_type import (
    SerializableType as StructValue,
)
from fprime_gds.common.models.serialize.array_type import ArrayType as ArrayValue
from fprime_gds.common.models.serialize.numerical_types import (
    U8Type as U8Value,
    U16Type as U16Value,
    U32Type as U32Value,
    NumericalType as NumericalValue,
)
from fprime_gds.common.models.serialize.type_base import BaseType as FppValue
from lark import Lark
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
    TimeIntervalValue,
    CompileState,
    CallableSymbol,
    CastSymbol,
    CommandSymbol,
    TypeCtorSymbol,
    Visitor,
    create_symbol_table,
    merge_symbol_tables,
)
from fprime_gds.common.loaders.ch_json_loader import ChJsonLoader
from fprime_gds.common.loaders.cmd_json_loader import CmdJsonLoader
from fprime_gds.common.loaders.event_json_loader import EventJsonLoader
from fprime_gds.common.loaders.prm_json_loader import PrmJsonLoader
from fprime_gds.common.loaders.type_json_loader import TypeJsonLoader
from fprime_gds.common.templates.cmd_template import CmdTemplate
from pathlib import Path
from lark import Lark, LarkError

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


@lru_cache(maxsize=4)
def _load_sequence_config(dictionary: str) -> dict:
    """
    Load sequence configuration from the dictionary constants section.
    Returns a dict with max_directives_count and max_directive_size.
    """
    with open(dictionary, "r") as f:
        dict_json = json.load(f)
    
    constants = dict_json.get("constants", [])
    
    # Build a lookup dict for constants by qualified name
    constants_by_name = {c["qualifiedName"]: c["value"] for c in constants if "qualifiedName" in c and "value" in c}
    
    return {
        "max_directives_count": constants_by_name.get("Svc.Fpy.MAX_SEQUENCE_STATEMENT_COUNT", DEFAULT_MAX_DIRECTIVES_COUNT),
        "max_directive_size": constants_by_name.get("Svc.Fpy.MAX_DIRECTIVE_SIZE", DEFAULT_MAX_DIRECTIVE_SIZE),
    }


@lru_cache(maxsize=4)
def _load_dictionary(dictionary: str) -> tuple:
    """
    Load and parse the dictionary file once, caching the results.
    Returns a tuple of (cmd_name_dict, ch_name_dict, prm_name_dict, type_name_dict).
    """
    cmd_json_dict_loader = CmdJsonLoader(dictionary)
    (_, cmd_name_dict, _) = cmd_json_dict_loader.construct_dicts(dictionary)

    ch_json_dict_loader = ChJsonLoader(dictionary)
    (_, ch_name_dict, _) = ch_json_dict_loader.construct_dicts(dictionary)
    prm_json_dict_loader = PrmJsonLoader(dictionary)
    (_, prm_name_dict, _) = prm_json_dict_loader.construct_dicts(dictionary)

    # Load all types from typeDefinitions - this includes types not referenced
    # by any command, channel, parameter, or event (e.g. Fw.TimeComparison)
    type_json_dict_loader = TypeJsonLoader(dictionary)
    (_, type_name_dict, _) = type_json_dict_loader.construct_dicts(dictionary)

    return (cmd_name_dict, ch_name_dict, prm_name_dict, type_name_dict)


@lru_cache(maxsize=4)
def _build_global_scopes(dictionary: str) -> tuple:
    """
    Build and cache the 3 global scopes for a dictionary.
    Returns tuple of (type_scope, callable_scope, values_scope).
    """
    cmd_name_dict, ch_name_dict, prm_name_dict, type_name_dict = _load_dictionary(
        dictionary
    )

    # Make a copy of type_name_dict since we'll mutate it
    type_name_dict = dict(type_name_dict)

    # enum const dict is a dict of fully qualified enum const name (like Ref.Choice.ONE) to its fprime value
    enum_const_name_dict: dict[str, FppValue] = {}

    # find each enum type, and put each of its values in the enum const dict
    for name, typ in type_name_dict.items():
        if issubclass(typ, EnumValue):
            for enum_const_name, val in typ.ENUM_DICT.items():
                enum_const_name_dict[name + "." + enum_const_name] = typ(
                    enum_const_name
                )

    # insert the builtin types into the dict
    type_name_dict["Fw.Time"] = TimeValue
    for typ in SPECIFIC_NUMERIC_TYPES:
        type_name_dict[typ.get_canonical_name()] = typ
    type_name_dict["bool"] = BoolValue
    # note no string type at the moment

    # Require Fw.TimeIntervalValue and Fw.TimeComparison from the dictionary
    if "Fw.TimeIntervalValue" not in type_name_dict:
        raise ValueError("Dictionary must contain Fw.TimeIntervalValue type")
    dict_time_interval_type = type_name_dict["Fw.TimeIntervalValue"]
    # Verify the dictionary's type matches our expected structure
    expected_members = list(TimeIntervalValue.MEMBER_LIST)
    dict_members = list(dict_time_interval_type.MEMBER_LIST)
    if expected_members != dict_members:
        raise ValueError(
            f"Dictionary Fw.TimeIntervalValue has members {dict_members}, "
            f"expected {expected_members}"
        )
    # Use our canonical type (which has the same structure)
    type_name_dict["Fw.TimeIntervalValue"] = TimeIntervalValue

    if "Fw.TimeComparison" not in type_name_dict:
        raise ValueError("Dictionary must contain Fw.TimeComparison enum")

    # once we have timeintervalvalue, make the checkstate struct
    # This is an internal type (prefixed with $) not directly accessible to users,
    # used for desugaring check statements
    CheckStateValue = StructValue.construct_type(
        "$CheckState",
        [
            ("persist", TimeIntervalValue, "", ""),
            ("timeout", TimeValue, "", ""),
            ("freq", TimeIntervalValue, "", ""),
            ("result", BoolValue, "", ""),
            ("last_was_true", BoolValue, "", ""),
            ("last_time_true", TimeValue, "", ""),
            ("time_started", TimeValue, "", ""),
        ],
    )
    # Add to type dict so it can be used internally
    type_name_dict["$CheckState"] = CheckStateValue

    cmd_response_type = type_name_dict["Fw.CmdResponse"]
    callable_name_dict: dict[str, CallableSymbol] = {}
    # add all cmds to the callable dict
    for name, cmd in cmd_name_dict.items():
        cmd: CmdTemplate
        args = []
        for arg_name, _, arg_type in cmd.arguments:
            args.append((arg_name, arg_type, None))  # No default values for cmds
        # cmds are thought of as callables with a Fw.CmdResponse return value
        callable_name_dict[name] = CommandSymbol(
            cmd.get_full_name(), cmd_response_type, args, cmd
        )

    # add numeric type casts to callable dict
    for typ in SPECIFIC_NUMERIC_TYPES:
        callable_name_dict[typ.get_canonical_name()] = CastSymbol(
            typ.get_canonical_name(), typ, [("value", NumericalValue, None)], typ
        )

    # for each type in the dict, if it has a constructor, create an TypeCtorSymbol
    # object to track the constructor and put it in the callable name dict
    for name, typ in type_name_dict.items():
        args = []
        if issubclass(typ, StructValue):
            for arg_name, arg_type, _, _ in typ.MEMBER_LIST:
                args.append(
                    (arg_name, arg_type, None)
                )  # No default values for struct ctors
        elif issubclass(typ, ArrayValue):
            for i in range(0, typ.LENGTH):
                args.append(("e" + str(i), typ.MEMBER_TYPE, None))
        elif issubclass(typ, TimeValue):
            args.append(("time_base", U16Value, None))
            args.append(("time_context", U8Value, None))
            args.append(("seconds", U32Value, None))
            args.append(("useconds", U32Value, None))
        else:
            # bool, enum, string or numeric type
            # none of these have callable ctors
            continue

        callable_name_dict[name] = TypeCtorSymbol(name, typ, args, typ)

    # for each macro function, add it to the callable dict
    for macro_name, macro in MACROS.items():
        callable_name_dict[macro_name] = macro

    # Build the 3 global scopes per SPEC:
    # 1. global type scope - leaf nodes are types
    type_scope = create_symbol_table(type_name_dict)
    # 2. global callable scope - leaf nodes are callables
    callable_scope = create_symbol_table(callable_name_dict)
    # 3. global value scope - leaf nodes are values (tlm channels, parameters, enum constants)
    #    Merge all value sources into one scope
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
    sequence_config = _load_sequence_config(dictionary)

    # Make copies of the scopes since we'll mutate them during compilation
    # (e.g., adding user-defined functions to callable_scope, variables to values_scope)
    # if we don't make copies, then the lru cache will return the modified versions, causing
    # two runs of the compiler to conflict
    state = CompileState(
        global_type_scope=type_scope,  # types are not mutated
        global_callable_scope=callable_scope.copy(),
        global_value_scope=values_scope.copy(),
        compile_args=compile_args or dict(),
        max_directives_count=sequence_config["max_directives_count"],
        max_directive_size=sequence_config["max_directive_size"],
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
