from __future__ import annotations
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
from fprime_gds.common.fpy.bytecode.directives import Directive
from fpy.codegen import (
    AssignVariableOffsets,
    FinalChecks,
    GenerateFunctionEntryPoints,
    GenerateFunctions,
    GenerateModule,
    IrPass,
    ResolveLabels,
)
from fpy.desugaring import DesugarDefaultArgs, DesugarForLoops
from fpy.semantics import (
    AssignIds,
    AssignLocalScopes,
    CalculateConstExprValues,
    CalculateDefaultArgConstValues,
    CheckBreakAndContinueInLoop,
    CheckConstArrayAccesses,
    CheckFunctionReturns,
    CheckReturnInFunc,
    CheckUseBeforeDeclare,
    CreateVariablesAndFuncs,
    PickTypesAndResolveAttrsAndItems,
    ResolveTypeNames,
    ResolveVars,
    WarnRangesAreNotEmpty,
)
from fpy.syntax import AstBlock, FpyTransformer, PythonIndenter
from fpy.macros import MACROS
from fpy.types import (
    SPECIFIC_NUMERIC_TYPES,
    CompileState,
    FppType,
    CallableSymbol,
    CastSymbol,
    CommandSymbol,
    TypeCtorSymbol,
    Visitor,
    create_symbol_table,
)
from fprime_gds.common.loaders.ch_json_loader import ChJsonLoader
from fprime_gds.common.loaders.cmd_json_loader import CmdJsonLoader
from fprime_gds.common.loaders.event_json_loader import EventJsonLoader
from fprime_gds.common.loaders.prm_json_loader import PrmJsonLoader
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
        print(fpy.error.CompileError("Maximum recursion depth exceeded (code is too deeply nested)"), file=sys.stderr)
        exit(1)
    except VisitError as e:
        # VisitError wraps exceptions that occur during tree transformation
        if isinstance(e.orig_exc, RecursionError):
            print(fpy.error.CompileError("Maximum recursion depth exceeded (code is too deeply nested)"), file=sys.stderr)
        else:
            print(fpy.error.CompileError(f"Internal error during parsing: {e.orig_exc}"), file=sys.stderr)
        exit(1)
    return transformed


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
    event_json_dict_loader = EventJsonLoader(dictionary)
    (_, _, _) = event_json_dict_loader.construct_dicts(dictionary)

    # the type name dict is a mapping of a fully qualified name to an fprime type
    # here we put into it all types found while parsing all cmds, params and tlm channels
    type_name_dict: dict[str, FppType] = dict(cmd_json_dict_loader.parsed_types)
    type_name_dict.update(ch_json_dict_loader.parsed_types)
    type_name_dict.update(prm_json_dict_loader.parsed_types)
    type_name_dict.update(event_json_dict_loader.parsed_types)

    return (cmd_name_dict, ch_name_dict, prm_name_dict, type_name_dict)


@lru_cache(maxsize=4)
def _build_scopes(dictionary: str) -> tuple:
    """
    Build and cache the scopes for a dictionary.
    Returns tuple of (tlm_scope, prm_scope, type_scope, callable_scope, const_scope).
    """
    cmd_name_dict, ch_name_dict, prm_name_dict, type_name_dict = _load_dictionary(dictionary)
    
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

    return (
        create_symbol_table(ch_name_dict),
        create_symbol_table(prm_name_dict),
        create_symbol_table(type_name_dict),
        create_symbol_table(callable_name_dict),
        create_symbol_table(enum_const_name_dict),
    )


def get_base_compile_state(dictionary: str, compile_args: dict) -> CompileState:
    """return the initial state of the compiler, based on the given dict path"""
    tlm_scope, prm_scope, type_scope, callable_scope, const_scope = _build_scopes(dictionary)

    state = CompileState(
        tlms=tlm_scope,
        prms=prm_scope,
        types=type_scope,
        callables=callable_scope,
        consts=const_scope,
        compile_args=compile_args or dict(),
    )
    return state


def ast_to_directives(
    body: AstBlock,
    dictionary: str,
    compile_args: dict | None = None,
) -> list[Directive] | CompileError | BackendError:
    compile_args = compile_args or dict()
    state = get_base_compile_state(dictionary, compile_args)
    state.root = body
    semantics_passes: list[Visitor] = [
        # assign each node a unique id for indexing/hashing
        AssignIds(),
        # based on position of node in tree, figure out which scope it is in
        AssignLocalScopes(),
        # based on assignment syntax nodes, we know which variables exist where
        CreateVariablesAndFuncs(),
        # check that break/continue are in loops, and store which loop they're in
        CheckBreakAndContinueInLoop(),
        CheckReturnInFunc(),
        # resolve type annotations first, since they use a restricted syntax (AstTypeExpr)
        # and we need to know variable types before resolving other references
        ResolveTypeNames(),
        # resolve all variable and function references
        ResolveVars(),
        # make sure we don't use any variables before they are declared
        CheckUseBeforeDeclare(),
        # this pass resolves all attributes and items, as well as determines the type of expressions
        PickTypesAndResolveAttrsAndItems(),
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
        # now that semantic analysis is done, we can desugar things. start with for loops
        DesugarForLoops(),
    ]
    codegen_passes = [
        # Assign variable offsets before generating function bodies
        # so global variable offsets are known when referenced in functions
        AssignVariableOffsets(),
        GenerateFunctionEntryPoints(),
        # generate all function bodies
        GenerateFunctions(),
    ]
    module_generator = GenerateModule()

    ir_passes: list[IrPass] = [ResolveLabels(), FinalChecks()]

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

    # all the ir is guaranteed to have been converted to directives by now
    return ir
