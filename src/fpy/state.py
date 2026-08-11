from __future__ import annotations
from functools import lru_cache
from typing import Union
from dataclasses import dataclass, field

import fpy.types
from fpy.dictionary import json_default_to_fpy_value, load_dictionary
from fpy.error import CompileError, CompileWarning, DictionaryError, WarningType
from fpy.macros import MACROS
from fpy.symbols import (
    CallableSymbol,
    CastSymbol,
    CommandSymbol,
    NameGroup,
    Scope,
    Symbol,
    TypeCtorSymbol,
    VariableSymbol,
    create_symbol_table,
    merge_symbol_tables,
)
from fpy.syntax import (
    Ast,
    AstBreak,
    AstContinue,
    AstDef,
    AstExpr,
    AstFor,
    AstFuncCall,
    AstIdent,
    AstOp,
    AstReference,
    AstReturn,
    AstBlock,
    AstWhile,
)
from fpy.types import (
    BLOCK_STATE,
    BOOL,
    CHECK_STATE,
    CMD_RESPONSE,
    DEFAULT_FW_SERIALIZE_FALSE_VALUE,
    DEFAULT_FW_SERIALIZE_TRUE_VALUE,
    DEFAULT_MAX_DIRECTIVES_COUNT,
    DEFAULT_MAX_DIRECTIVE_SIZE,
    DEFAULT_MAX_SEQ_ARG_COUNT,
    DEFAULT_MAX_STACK_SIZE,
    FLAGS_TYPE,
    I64,
    LOG_SEVERITY,
    SEQ_ARGS,
    SPECIFIC_NUMERIC_TYPES,
    TIME,
    TIME_BASE,
    TIME_COMPARISON,
    TIME_INTERVAL,
    FpyType,
    FpyValue,
    TypeKind,
)
from fpy.bytecode.directives import update_configurable_types_from_dict


class BackendState:
    """Base for what a backend works out about a program while lowering it.
    Each backend defines its own subclass and installs it on
    CompileState.backend."""


@dataclass
class ForLoopAnalysis:
    loop_var: VariableSymbol
    upper_bound_var: VariableSymbol


@dataclass
class CompileState:
    """a collection of input, internal and output state variables and maps"""

    main_scope: Scope = None
    """The main sequence's scope: a child of `base_scope` holding the main
    sequence's own top-level definitions. None until CreateScopes runs"""

    type_defs: dict = field(default_factory=dict)
    """Flat map of fully-qualified type name to FpyType, for resolving types at compile time."""
    ground_binary_dir: str | None = None
    """Local directory for resolving compiled sequence binaries (.bin files)."""
    # Sequence limits loaded from dictionary (or defaults if not specified)
    max_directives_count: int = DEFAULT_MAX_DIRECTIVES_COUNT
    max_directive_size: int = DEFAULT_MAX_DIRECTIVE_SIZE
    max_seq_arg_count: int = DEFAULT_MAX_SEQ_ARG_COUNT
    max_stack_size: int = DEFAULT_MAX_STACK_SIZE
    # Command transport limits loaded from dictionary. None means the constant
    # is not in the dictionary, in which case the corresponding check is skipped.
    com_buffer_max_size: int | None = None
    cmd_arg_buffer_max_size: int | None = None
    cmd_names_by_opcode: dict[int, str] = field(default_factory=dict)
    """Map of command opcode to fully-qualified command name, for error messages."""

    next_node_id: int = 0
    root_block: AstBlock = None
    """the outermost block: the library root, which owns the base scope and holds
    the builtin library functions, the main block, and every imported sequence
    block as its children. Built by _build_root_block."""
    main_block: AstBlock = None
    """the main sequence's block (the executable body), a child of `root`. The
    frame layout and directive stream are generated from this block, not `root`
    (which also contains the builtin library and the imported sequences)."""
    imported_blocks: list = field(default_factory=list, repr=False)
    """every imported sequence file's block B, collected by ConstructAst.
    _build_root_block installs them as children of the library `root`
    (siblings of `main_block`), making each "a sibling of the main sequence's
    block": an isolated scope block that does not execute inline."""
    parent_map: dict[Ast, Ast] = field(default_factory=dict, repr=False)
    """map of each node to its parent node in the AST"""
    enclosing_scope: dict[Ast, Scope] = field(default_factory=dict, repr=False)
    """map of node to its enclosing Scope (block scope, function scope, a
    sequence's root scope, or the base scope)."""
    for_loops: dict[AstFor, ForLoopAnalysis] = field(default_factory=dict)
    """map of for loops to a ForLoopAnalysis struct, which contains additional info about the loops"""
    enclosing_loops: dict[Union[AstBreak, AstContinue], Union[AstFor, AstWhile]] = (
        field(default_factory=dict)
    )
    """map of break/continue to the loop which contains the break/continue"""
    desugared_for_loops: dict[AstWhile, AstFor] = field(default_factory=dict)
    """mapping of while loops which are desugared for loops, to the original node from which they came"""

    enclosing_funcs: dict[AstReturn, AstDef] = field(default_factory=dict)

    contextual_name_group: dict[AstExpr, NameGroup] = field(
        default_factory=dict, repr=False
    )
    """The name group an expression is used in, determined purely from where it
    appears (a func's callee is CALLABLE, a type annotation is TYPE, an
    operand/argument/rhs is VALUE)."""

    resolved_symbols: dict[AstReference, Symbol] = field(
        default_factory=dict, repr=False
    )
    """reference to its singular resolution"""

    synthesized_types: dict[AstExpr, FpyType] = field(default_factory=dict)
    """expr to its fprime type, before type conversions are applied"""

    op_intermediate_types: dict[AstOp, FpyType] = field(default_factory=dict)
    """the intermediate type that all args should be converted to for the given op"""

    expr_explicit_casts: list[AstExpr] = field(default_factory=list)
    """a list of nodes which are explicit casts"""
    contextual_types: dict[AstExpr, FpyType] = field(default_factory=dict)
    """expr to fprime type it will end up being on the stack after type conversions"""

    const_expr_values: dict[AstExpr, FpyValue | None] = field(default_factory=dict)
    """expr to the fprime value it will end up being on the stack after type conversions.
    None if unsure at compile time.  NOTHING_VALUE for void expressions."""

    resolved_args: dict[Ast, list[AstExpr]] = field(default_factory=dict)
    """Maps function calls, anon structs, and anon arrays to resolved arguments
    in positional order. Default values are filled in for arguments not provided
    at the call site (or struct members / array elements with defaults)."""

    function_global_uses: dict[AstDef, list[VariableSymbol]] = field(
        default_factory=dict
    )
    """function definition -> globals it reads. Populated with direct uses by
    CollectFunctionGlobalUses, then grown to the transitive closure (globals
    read through called functions too) by ResolveTransitiveGlobalUses."""

    function_callees: dict[AstDef, list[AstDef]] = field(default_factory=dict)
    """function definition -> the user function definitions it directly calls"""

    used_funcs: set[AstDef] = field(default_factory=set)
    """the function definitions with a call site anywhere in the program, i.e.
    the ones a backend must generate code for (over-approximate: includes
    functions called only from unused functions)"""

    does_return: dict[Ast, bool] = field(default_factory=dict)

    backend: BackendState | None = None
    """the state of the backend lowering this program, or None if none is.
    Written and read only by that backend."""

    flags_var: VariableSymbol = None
    """The built-in 'flags' variable ($Flags struct) that controls sequencer behavior."""

    this_seq_arg_specs: list[tuple[str, FpyType]] = field(default_factory=list)
    """Ordered list of (arg_name, arg_type) for sequence parameters.
    Populated during semantic analysis."""

    called_seq_arg_specs: dict[str, list[tuple[str, FpyType]]] = field(
        default_factory=dict
    )
    """Map of .bin filename to resolved (arg_name, arg_type) pairs, populated by ResolveSequenceDependencies."""

    errors: list[CompileError] = field(default_factory=list)
    """a list of all compile exceptions generated by passes"""

    warnings: list[CompileWarning] = field(default_factory=list)
    """a list of all compiler warnings generated by passes (after filtering out
    ignored ones; escalated ones are appended to `errors` instead)"""

    ignored_warnings: set[WarningType] = field(default_factory=set)
    """warning types to silently drop (`--ignore`)"""

    error_warnings: set[WarningType] = field(default_factory=set)
    """warning types to promote to hard compile errors (`--error`)"""

    import_directories: list[str] = field(default_factory=list)
    """"The import directories are an ordered list of absolute paths of
    directories provided by the environment in which the compiler is
    invoked." (SPEC.md Imports) An absolute import statement's first identifier is
    resolved in each of them in order until it succeeds. Populated from
    `-i/--imports` in the CLI."""

    main_file_dir: str | None = None
    """directory containing the main sequence's file, from which a relative
    import statement's anchor directory is counted. None when compiling from
    a stream (a relative import in the main sequence is then an error)."""

    main_file_path: str | None = None
    """the absolute path of the main sequence's file, so that an import path
    resolving to it is recognized as the main sequence and skipped rather
    than included again. None when compiling from a stream."""

    base_scope: Scope = None
    """the dictionary/builtin scope: dictionary types, commands/casts/type
    constructors/macros (callable group), and dictionary channels, params, enum
    constants, constants and `flags` (value group), plus the builtin library
    functions (builtin/time.fpy) that DefineFunctions registers via the library
    root block."""

    main_sequence: object = None
    """the SequenceContext for the main (top-level) sequence being compiled."""
    resolved_imports: list = field(default_factory=list, repr=False)
    """each import statement whose import path has resolved to a sequence
    definition (a ResolvedImport), in inner-first order. BindImports
    associates their names once every sequence's definitions have been
    registered (by DefineFunctions / DefineVariables)."""
    loaded_sequences: dict = field(default_factory=dict, repr=False)
    """maps a sequence file (absolute path) -> its SequenceContext: every
    imported sequence file included in the program's AST, plus the main
    sequence when its file is known. An import path resolving to a file in
    this map is skipped rather than included again, so a file is lexed,
    parsed and included once, however many import statements name it, and
    its definitions are shared, never duplicated."""

    next_anon_var_id: int = 0

    def new_anonymous_variable_name(self) -> str:
        id = self.next_anon_var_id
        self.next_anon_var_id += 1
        return f"$value{id}"

    def err(self, msg, n):
        """adds a compile exception to internal state"""
        self.errors.append(CompileError(msg, n))

    def warn(self, type: WarningType, msg, n):
        """Record a warning of the given type"""
        if type in self.ignored_warnings:
            return
        if type in self.error_warnings:
            self.errors.append(CompileError(f"{msg} [{type.value}]", n))
            return
        self.warnings.append(CompileWarning(type, msg, n))


def _validate_and_replace_type(
    type_dict: dict[str, FpyType],
    name: str,
    canonical: FpyType,
) -> None:
    """Validate that a required type exists in the dictionary and matches the
    canonical definition, then replace it with the canonical version.

    Raises DictionaryError (with a user-facing explanation) if the dictionary is
    missing the type or defines it incompatibly with the canonical version."""
    if name not in type_dict:
        raise DictionaryError(name, "The dictionary does not define this type at all.")
    dict_type = type_dict[name]
    if dict_type.kind != canonical.kind:
        raise DictionaryError(
            name,
            f"The dictionary defines it as a {dict_type.kind.name} type, "
            f"but Fpy expects a {canonical.kind.name} type.",
        )
    if canonical.kind == TypeKind.STRUCT:
        if dict_type.members != canonical.members:
            raise DictionaryError(
                name,
                f"Its struct members do not match what fpy expects.\n"
                f"    dictionary: {dict_type.members}\n"
                f"    expected:   {canonical.members}",
            )
    elif canonical.kind == TypeKind.ENUM:
        if dict_type.enum_dict != canonical.enum_dict:
            raise DictionaryError(
                name,
                f"Its enum constants do not match what fpy expects.\n"
                f"    dictionary: {dict_type.enum_dict}\n"
                f"    expected:   {canonical.enum_dict}",
            )
        if dict_type.rep_type != canonical.rep_type:
            raise DictionaryError(
                name,
                f"Its underlying representation type is {dict_type.rep_type}, "
                f"but Fpy expects {canonical.rep_type}.",
            )
    elif canonical.kind == TypeKind.ARRAY:
        if dict_type.elem_type != canonical.elem_type:
            raise DictionaryError(
                name,
                f"Its element type is {dict_type.elem_type}, "
                f"but Fpy expects {canonical.elem_type}.",
            )
        if dict_type.length != canonical.length:
            raise DictionaryError(
                name,
                f"It has length {dict_type.length}, "
                f"but Fpy expects length {canonical.length}.",
            )
    type_dict[name] = canonical
    # Preserve raw JSON defaults from the dictionary definition on the canonical type
    canonical.json_default = dict_type.json_default


def _update_time_base_from_dict(dict_type_name_dict: dict[str, FpyType]) -> None:
    """Update the canonical TIME_BASE singleton from the dictionary's TimeBase.

    The dictionary's TimeBase enum supercedes the hardcoded placeholder.
    We only require that TB_NONE exists with value 0.  The full set of enum
    constants and the representation type (FwTimeBaseStoreType) come from the
    dictionary.
    """
    if "TimeBase" not in dict_type_name_dict:
        raise DictionaryError(
            "TimeBase", "The dictionary does not define this enum type at all."
        )
    dict_tb = dict_type_name_dict["TimeBase"]
    if dict_tb.kind != TypeKind.ENUM:
        raise DictionaryError(
            "TimeBase",
            f"The dictionary defines it as a {dict_tb.kind.name} type, "
            f"but Fpy expects an ENUM type.",
        )
    if "TB_NONE" not in dict_tb.enum_dict:
        raise DictionaryError(
            "TimeBase", "Its enum constants must include TB_NONE, but it is missing."
        )
    if dict_tb.enum_dict["TB_NONE"] != 0:
        raise DictionaryError(
            "TimeBase",
            f"Its TB_NONE constant must have value 0, "
            f"but the dictionary gives it value {dict_tb.enum_dict['TB_NONE']}.",
        )

    # Adopt the dictionary's enum constants and representation type
    TIME_BASE.enum_dict = dict_tb.enum_dict
    TIME_BASE.rep_type = dict_tb.rep_type
    TIME_BASE.json_default = dict_tb.json_default

    # Replace the dict entry with the canonical singleton
    dict_type_name_dict["TimeBase"] = TIME_BASE


def _update_seq_args_from_dict(dict_type_name_dict: dict[str, FpyType]) -> None:
    """Update the canonical SEQ_ARGS singleton from the dictionary's Svc.SeqArgs.

    The dictionary's `Svc.SeqArgs` defines the actual buffer capacity for this
    deployment.  The compiler adopts that length onto the canonical singleton's
    buffer type so codegen and semantics use the correct size.
    """
    assert (
        "Svc.SeqArgs" in dict_type_name_dict
    ), "Dictionary must contain Svc.SeqArgs type"
    dict_seq_args = dict_type_name_dict["Svc.SeqArgs"]
    assert (
        dict_seq_args.kind == TypeKind.STRUCT
    ), f"Dictionary Svc.SeqArgs has kind {dict_seq_args.kind}, expected struct"
    assert dict_seq_args.members is not None and len(dict_seq_args.members) == 2, (
        f"Dictionary Svc.SeqArgs must have exactly 2 members, "
        f"got {dict_seq_args.members}"
    )
    size_member, buffer_member = dict_seq_args.members
    canonical_size_member, canonical_buffer_member = SEQ_ARGS.members
    assert size_member.name == canonical_size_member.name, (
        f"Dictionary Svc.SeqArgs first member is '{size_member.name}', "
        f"expected '{canonical_size_member.name}'"
    )
    assert size_member.type == canonical_size_member.type, (
        f"Dictionary Svc.SeqArgs.{size_member.name} has type {size_member.type}, "
        f"expected {canonical_size_member.type}"
    )
    assert buffer_member.name == canonical_buffer_member.name, (
        f"Dictionary Svc.SeqArgs second member is '{buffer_member.name}', "
        f"expected '{canonical_buffer_member.name}'"
    )
    dict_buffer_type = buffer_member.type
    canonical_buffer_type = canonical_buffer_member.type
    assert dict_buffer_type.kind == TypeKind.ARRAY, (
        f"Dictionary Svc.SeqArgs.{buffer_member.name} has kind "
        f"{dict_buffer_type.kind}, expected array"
    )
    assert dict_buffer_type.elem_type == canonical_buffer_type.elem_type, (
        f"Dictionary Svc.SeqArgs.{buffer_member.name} has element type "
        f"{dict_buffer_type.elem_type}, "
        f"expected {canonical_buffer_type.elem_type}"
    )
    assert dict_buffer_type.length is not None and dict_buffer_type.length > 0, (
        f"Dictionary Svc.SeqArgs.{buffer_member.name} must have a positive "
        f"length, got {dict_buffer_type.length}"
    )

    # Adopt the dictionary's buffer length onto the canonical buffer singleton.
    canonical_buffer_type.length = dict_buffer_type.length
    canonical_buffer_type.name = f"Array_U8_{dict_buffer_type.length}"

    # Preserve raw JSON defaults from the dictionary so _populate_type_defaults
    # can derive the correct-length buffer default.
    SEQ_ARGS.json_default = dict_seq_args.json_default

    # Replace the dict entry with the canonical singleton.
    dict_type_name_dict["Svc.SeqArgs"] = SEQ_ARGS


def _update_time_context_type_from_dict(
    dict_type_name_dict: dict[str, FpyType],
) -> None:
    """Update TIME's timeContext member type from FwTimeContextStoreType."""
    if "FwTimeContextStoreType" not in dict_type_name_dict:
        return  # Keep the default U8
    ctx_type = dict_type_name_dict["FwTimeContextStoreType"]
    if not ctx_type.is_primitive:
        raise DictionaryError(
            "FwTimeContextStoreType",
            f"It must resolve to a primitive type, but the dictionary defines "
            f"it as {ctx_type}.",
        )
    TIME.members[1].type = ctx_type


def _get_elem_type_default(elem_type: FpyType) -> FpyValue:
    """Return the zero/default FpyValue for a primitive, enum, or other element type."""
    if elem_type.kind == TypeKind.BOOL:
        return FpyValue(elem_type, False)
    if elem_type.is_integer:
        return FpyValue(elem_type, 0)
    if elem_type.is_float:
        return FpyValue(elem_type, 0.0)
    if elem_type.kind in (TypeKind.STRING, TypeKind.INTERNAL_STRING):
        return FpyValue(elem_type, "")
    assert (
        elem_type.json_default is not None
    ), f"Element type {elem_type.name} must have json_default"
    return json_default_to_fpy_value(elem_type.json_default, elem_type)


def _derive_elem_defaults(typ: FpyType) -> list[FpyValue]:
    """Derive elem_defaults for a struct member array type (no json_default)."""
    elem_default = _get_elem_type_default(typ.elem_type)
    return [elem_default] * typ.length


def _populate_type_defaults(typ: FpyType) -> None:
    """Populate per-member/per-element defaults on an FpyType from its json_default.

    Sets FpyType.member_defaults for structs and FpyType.elem_defaults for arrays.
    Every type is guaranteed to have a default value.
    """
    if typ.kind == TypeKind.STRUCT:
        if typ.member_defaults is not None:
            return  # Already populated (e.g., built-in FLAGS_TYPE)
        assert typ.json_default is not None, f"Struct {typ.name} must have json_default"
        struct_defaults: dict[str, FpyValue] = {}
        for m in typ.members:
            raw_val = typ.json_default.get(m.name)
            assert (
                raw_val is not None
            ), f"Missing default for member '{m.name}' of struct {typ.name}"
            if m.type.kind == TypeKind.ARRAY and not (
                isinstance(raw_val, list) and len(raw_val) == m.type.length
            ):
                # Struct member arrays (members with a "size" key in the JSON)
                # get wrapped in a struct member array type, but the dictionary's
                # raw default is a single element value rather than an
                # array-shaped value.  Per FPP's spec
                # (see https://github.com/nasa/fpp/issues/925), this single
                # value initializes every element of the member array.
                # We replicate it to build the full array-shaped FpyValue.
                elem_val = json_default_to_fpy_value(raw_val, m.type.elem_type)
                struct_defaults[m.name] = FpyValue(m.type, [elem_val] * m.type.length)
            else:
                struct_defaults[m.name] = json_default_to_fpy_value(raw_val, m.type)
        typ.member_defaults = struct_defaults
    elif typ.kind == TypeKind.ARRAY:
        if typ.json_default is not None:
            default_val = json_default_to_fpy_value(typ.json_default, typ)
            array_defaults = default_val.val  # list of FpyValue
            assert len(array_defaults) == typ.length, (
                f"Dictionary array type {typ.name} has default with "
                f"{len(array_defaults)} elements but declared length {typ.length}"
            )
        else:
            # Struct member array type (no json_default of its own) —
            # derive element defaults from the element type.
            array_defaults = _derive_elem_defaults(typ)
        typ.elem_defaults = tuple(array_defaults)


def _make_type_ctor(name: str, typ: FpyType) -> TypeCtorSymbol | None:
    """Create a TypeCtorSymbol for a type, or return None if it has no callable ctor."""
    if typ.kind == TypeKind.STRUCT:
        args = [(m.name, m.type, typ.member_defaults[m.name]) for m in typ.members]
    elif typ.kind == TypeKind.ARRAY:
        args = [
            ("e" + str(i), typ.elem_type, typ.elem_defaults[i])
            for i in range(typ.length)
        ]
    else:
        return None
    return TypeCtorSymbol(name, typ, args, typ)


@lru_cache(maxsize=4)
def _build_global_scopes(dictionary: str) -> tuple:
    """
    Build and cache the 3 global scopes and type_name_dict for a dictionary.
    Returns tuple of (type_scope, callable_scope, values_scope, type_name_dict).
    """
    d = load_dictionary(dictionary)
    cmd_name_dict = d["cmd_name_dict"]
    ch_name_dict = d["ch_name_dict"]
    prm_name_dict = d["prm_name_dict"]
    dict_type_name_dict = d["type_defs"]

    # Update user-configurable bytecode types (FwChanIdType, FwOpcodeType,
    # FwPrmIdType, FwSizeStoreType) from the dictionary before they are used.
    update_configurable_types_from_dict(dict_type_name_dict)

    # Validate required dictionary types
    _update_time_base_from_dict(dict_type_name_dict)
    _update_time_context_type_from_dict(dict_type_name_dict)
    _validate_and_replace_type(dict_type_name_dict, "Fw.TimeValue", TIME)
    _validate_and_replace_type(
        dict_type_name_dict, "Fw.TimeIntervalValue", TIME_INTERVAL
    )
    _validate_and_replace_type(dict_type_name_dict, "Fw.CmdResponse", CMD_RESPONSE)
    _validate_and_replace_type(
        dict_type_name_dict, "Fw.TimeComparison", TIME_COMPARISON
    )
    _validate_and_replace_type(dict_type_name_dict, "Svc.BlockState", BLOCK_STATE)
    _update_seq_args_from_dict(dict_type_name_dict)

    # Build the full type dict: start from (now-validated) dictionary types,
    # then layer on builtins and internal types.  Later entries win, so
    # canonical replacements from _validate_and_replace_type are preserved.
    type_name_dict: dict[str, FpyType] = {
        **dict_type_name_dict,
        # Aliases: Fw.Time -> Fw.TimeValue, Fw.TimeInterval -> Fw.TimeIntervalValue
        "Fw.Time": TIME,
        "Fw.TimeInterval": TIME_INTERVAL,
        **{typ.name: typ for typ in SPECIFIC_NUMERIC_TYPES},
        BOOL.name: BOOL,
        CHECK_STATE.name: CHECK_STATE,
        FLAGS_TYPE.name: FLAGS_TYPE,
        LOG_SEVERITY.name: LOG_SEVERITY,
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

    # Populate per-member/per-element defaults on types before building ctors
    for typ in type_name_dict.values():
        _populate_type_defaults(typ)

    # Build callable dict: commands, numeric casts, type constructors, macros
    callable_name_dict: dict[str, CallableSymbol] = {}

    for name, cmd in cmd_name_dict.items():
        args = [(arg_name, arg_type, None) for arg_name, _, arg_type in cmd.arguments]
        # Detect sequence-run commands by matching the 3-arg signature:
        # (fileName: string, block: Svc.BlockState, args: Svc.SeqArgs)
        if (
            len(args) == 3
            and args[0][1].is_string
            and args[1][1].name == "Svc.BlockState"
            and args[2][1].name == "Svc.SeqArgs"
        ):
            # Strip the SeqArgs param; user provides varargs instead
            fixed_args = args[:2]
            callable_name_dict[name] = CommandSymbol(
                cmd.name, CMD_RESPONSE, fixed_args, cmd, is_seq_run_with_args=True
            )
        else:
            callable_name_dict[name] = CommandSymbol(cmd.name, CMD_RESPONSE, args, cmd)

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
    # 3. global value scope - leaf nodes are values (tlm channels, parameters, enum constants, FPP constants)
    fpp_constants = d["constants"]
    values_scope = merge_symbol_tables(
        create_symbol_table(ch_name_dict),
        merge_symbol_tables(
            create_symbol_table(prm_name_dict),
            merge_symbol_tables(
                create_symbol_table(enum_const_name_dict),
                create_symbol_table(fpp_constants),
            ),
        ),
    )

    return (type_scope, callable_scope, values_scope, type_name_dict)


def get_base_compile_state(
    dictionary: str,
    ground_binary_dir: str | None = None,
    ignored_warnings: set[WarningType] | None = None,
    error_warnings: set[WarningType] | None = None,
    import_directories: list[str] | None = None,
    main_file_dir: str | None = None,
    main_file_path: str | None = None,
) -> CompileState:
    """return the initial state of the compiler, based on the given dict path"""
    type_scope, callable_scope, values_scope, type_defs = _build_global_scopes(
        dictionary
    )
    constants = load_dictionary(dictionary)["constants"]

    def _const_int(key: str, default: int | None) -> int | None:
        """Extract an integer constant value, falling back to *default*."""
        val = constants.get(key)
        if val is None:
            return default
        assert isinstance(
            val.val, int
        ), f"Expected int for constant {key}, got {type(val.val)}"
        return val.val

    # Override the live boolean wire-format constants (read directly by the stateless FpyValue.serialize()) from the dictionary.
    fpy.types.FW_SERIALIZE_TRUE_VALUE = _const_int(
        "FW_SERIALIZE_TRUE_VALUE", DEFAULT_FW_SERIALIZE_TRUE_VALUE
    )
    fpy.types.FW_SERIALIZE_FALSE_VALUE = _const_int(
        "FW_SERIALIZE_FALSE_VALUE", DEFAULT_FW_SERIALIZE_FALSE_VALUE
    )

    # The base scope holds the dictionary/builtin names, split by name group:
    # types in the type group; commands, casts, type constructors and macros in
    # the callable group (plus the builtin library functions once compiled); and
    # dictionary channels, params, enum constants, constants and `flags` in the
    # value group. It is the parent of every sequence's scope -- the main
    # sequence's main_scope and every imported sequence's -- so those names
    # resolve up the parent chain for all of them, while each sequence's own
    # definitions live in its own child scope, isolated from the rest.
    #
    # The group dicts are copied out of the (lru-cached) module tables so that
    # mutating the base scope during a compile does not corrupt the cache.
    base_scope = Scope()
    base_scope.group(NameGroup.TYPE).update(type_scope)
    base_scope.group(NameGroup.CALLABLE).update(callable_scope)
    base_scope.group(NameGroup.VALUE).update(values_scope)

    state = CompileState(
        type_defs=type_defs,
        ground_binary_dir=ground_binary_dir,
        max_directives_count=_const_int(
            "Svc.Fpy.MAX_SEQUENCE_STATEMENT_COUNT", DEFAULT_MAX_DIRECTIVES_COUNT
        ),
        max_directive_size=_const_int(
            "Svc.Fpy.MAX_DIRECTIVE_SIZE", DEFAULT_MAX_DIRECTIVE_SIZE
        ),
        max_seq_arg_count=_const_int(
            "Svc.Fpy.MAX_SEQUENCE_ARG_COUNT", DEFAULT_MAX_SEQ_ARG_COUNT
        ),
        max_stack_size=_const_int("Svc.Fpy.MAX_STACK_SIZE", DEFAULT_MAX_STACK_SIZE),
        com_buffer_max_size=_const_int("FW_COM_BUFFER_MAX_SIZE", None),
        cmd_arg_buffer_max_size=_const_int("FW_CMD_ARG_BUFFER_MAX_SIZE", None),
        cmd_names_by_opcode={
            opcode: cmd.name
            for opcode, cmd in load_dictionary(dictionary)["cmd_id_dict"].items()
        },
        ignored_warnings=set(ignored_warnings) if ignored_warnings else set(),
        error_warnings=set(error_warnings) if error_warnings else set(),
        import_directories=list(import_directories) if import_directories else [],
        main_file_dir=main_file_dir,
        main_file_path=main_file_path,
    )
    state.base_scope = base_scope

    # Create the built-in 'flags' variable ($Flags struct). declaration=None
    # marks it as a built-in that is always defined. It lives in the base scope's
    # value group, so it is shared across all sequences.
    flags_var = VariableSymbol("flags", None, None, FLAGS_TYPE, is_global=True)
    base_scope.define(NameGroup.VALUE, "flags", flags_var)
    state.flags_var = flags_var

    return state
