"""Tests of the LLVM/wasm backend that the dual-backend assert_* helpers
cannot express. Runtime behavior shared with the bytecode backend lives in
the dual-backend suites (which compile and run every sequence through both
backends); what stays here is:

* lowering and codegen properties read off the LLVM module or the emitted
  wasm itself (function lowering, target CPU features, host imports); and
* the exact bytes the module hands its host imports -- events, dispatched
  command buffers, and serial writes -- which the runner harness reports
  back and the dual-backend helpers discard.

Runtime behavior is exercised through variables: an all-literal expression
folds at compile time, so tests that want the wasm to actually compute
something route one operand through a variable.
"""

import struct

import pytest

from llvmlite import ir
import llvmlite.binding as llvm

from fpy.codegen_llvm import (
    FPY_ENTRY_POINT,
    LLVM_CPU,
    LLVM_TRIPLE,
    EmitLlvmExpr,
    GenerateLlvmModule,
    llvm_module_to_wasm,
    _ensure_llvm_targets,
)
from fpy.wasm_host import (
    ERROR_CODE_TYPE,
    HOST_EXIT_FUNC_NAME,
    declare_host_imports,
)
from fpy.compiler import analyze_ast, text_to_ast
from fpy.dictionary import load_dictionary
from fpy.bytecode.directives import DirectiveErrorCode
from fpy.state import get_base_compile_state
from fpy.test_helpers import (
    compile_seq_wasm,
    default_dictionary,
    run_seq_wasm_with_cmds,
    run_seq_wasm_with_events,
    run_seq_wasm_with_serial,
    run_wasm,
)
import fpy.types
from fpy.types import (
    BOOL,
    F32,
    F64,
    FpyValue,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
)

# Every test in this module drives the LLVM/wasm backend end-to-end, through
# its own helpers rather than the dual-backend assert_* ones. The wasm marker
# skips them under --backend fpybc.
pytestmark = pytest.mark.wasm


NO_ERROR = DirectiveErrorCode.NO_ERROR.value


def _seq_to_llvm_module(seq: str):
    """Lower *seq* to an llvmlite ir.Module (pre-codegen, target-independent)."""
    state = get_base_compile_state(default_dictionary)
    body = text_to_ast(seq)
    state = analyze_ast(body, state)
    return GenerateLlvmModule().emit(state.root_block, state)


def _emit_wasm_asm(seq: str, cpu: str) -> str:
    """Lower *seq* and emit its wasm textual assembly for the given target CPU.

    Re-parses the IR each call: emitting codegen mutates the parsed module (it
    bakes target-features attributes into the functions), so a parsed module
    can't be reused across CPUs without cross-contaminating results.
    """
    _ensure_llvm_targets()
    parsed = llvm.parse_assembly(str(_seq_to_llvm_module(seq)))
    parsed.verify()
    target = llvm.Target.from_triple(LLVM_TRIPLE)
    return target.create_target_machine(cpu=cpu).emit_assembly(parsed)


class TestWasmLowering:
    """Properties of the lowered LLVM module and the emitted wasm assembly."""

    def test_unused_function_is_not_lowered(self):
        # Only used functions are lowered: an uncalled def contributes no LLVM
        # function to the module, so the only defined function is the entry
        # point (the rest are host-import declarations).
        module = _seq_to_llvm_module(
            "def unused() -> U32:\n    return 1\nassert 1 == 1\n"
        )
        defined = [f.name for f in module.functions if not f.is_declaration]
        assert defined == [FPY_ENTRY_POINT]

    def test_unused_mutually_recursive_functions_are_not_lowered(self):
        # Two functions that only call each other, with nothing reachable from
        # the main sequence calling either, are not used -- even though call
        # sites for both exist (inside each other's bodies).
        seq = (
            "def a() -> U64:\n"
            "    return b()\n"
            "def b() -> U64:\n"
            "    return a()\n"
            "assert 1 == 1\n"
        )
        module = _seq_to_llvm_module(seq)
        defined = [f.name for f in module.functions if not f.is_declaration]
        assert defined == [FPY_ENTRY_POINT]

    def test_stays_mvp_no_trunc_sat(self):
        """Out-of-range float->int casts lower to llvm.fptosi.sat /
        llvm.fptoui.sat, and the saturating intrinsic must not pull in the
        post-MVP saturating op: the MVP target lowers it to a guarded trunc
        (no trunc_sat), whereas the default 'generic' CPU would use trunc_sat.
        Guards against the backend dropping cpu=LLVM_CPU or LLVM changing its
        feature defaults."""
        seq = "x: F64 = 1e20\ny: I32 = I32(x)\nassert y == 0\n"
        assert "i32.trunc_sat_f64_s" not in _emit_wasm_asm(seq, cpu=LLVM_CPU)
        assert "i32.trunc_sat_f64_s" in _emit_wasm_asm(seq, cpu="generic")


class TestWasmHostImports:
    """Document the host-call contract: which imports the linked module asks
    for. An import-section entry encodes as <len>module <len>name <kind>, so a
    function import is exactly that byte run in the binary."""

    def test_pow_emits_env_pow_import(self):
        # `**` lowers to the llvm.pow intrinsic, which the wasm target leaves
        # as an imported env.pow host call.
        wasm = compile_seq_wasm("x: F64 = 2.0\nassert x ** 3.0 == 8.0\n")
        assert b"\x03env\x03pow\x00" in wasm

    def test_cmd_emits_fprime_cmd_import(self):
        wasm = compile_seq_wasm("CdhCore.cmdDisp.CMD_NO_OP()\n")
        assert b"\x09fprime_v1\x03cmd\x00" in wasm

    def test_tlm_emits_fprime_tlm_import(self):
        wasm = compile_seq_wasm("x: U32 = CdhCore.cmdDisp.CommandsDispatched\n")
        assert b"\x09fprime_v1\x03tlm\x00" in wasm

    def test_prm_emits_fprime_prm_import(self):
        wasm = compile_seq_wasm("c: Ref.Choice = Ref.typeDemo.CHOICE_PRM\n")
        assert b"\x09fprime_v1\x03prm\x00" in wasm


class TestWasmLog:
    """log() lowers to the host `event(severity, ptr, len)` call, with the
    message bytes in a constant in linear memory. The runner harness reports
    each call back as a (severity, message) pair."""

    def test_default_severity_is_activity_hi(self):
        code, events = run_seq_wasm_with_events('log("hello world")\n')
        assert code == NO_ERROR
        assert events == [(5, "hello world")]  # ACTIVITY_HI = 5

    def test_explicit_severity(self):
        code, events = run_seq_wasm_with_events('log("oh no", Fw.LogSeverity.FATAL)\n')
        assert code == NO_ERROR
        assert events == [(1, "oh no")]  # FATAL = 1

    def test_multiple_events_in_call_order(self):
        code, events = run_seq_wasm_with_events(
            'log("first")\n'
            'log("second", Fw.LogSeverity.WARNING_HI)\n'
            'log("first")\n'
        )
        assert code == NO_ERROR
        assert events == [(5, "first"), (2, "second"), (5, "first")]

    def test_empty_message(self):
        code, events = run_seq_wasm_with_events('log("")\n')
        assert code == NO_ERROR
        assert events == [(5, "")]

    def test_log_before_exit_still_reported(self):
        # The event host call must happen before the sequence terminates.
        code, events = run_seq_wasm_with_events('log("bye")\nexit(9)\n')
        assert code == 9
        assert events == [(5, "bye")]


class TestWasmWriteToPort:
    """write_to_port(port, value) lowers to the host
    `serial_send(port, ptr, len)` call, with the value serialized into linear
    memory in fprime wire format. The runner harness reports each send back as
    a (port index, bytes) pair."""

    def test_runtime_int(self):
        # The value is a variable read, so it serializes at runtime.
        code, serial = run_seq_wasm_with_serial(
            "value: U32 = 42\n"
            "write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, value)\n"
        )
        assert code == NO_ERROR
        assert serial == [(0, struct.pack(">I", 42))]

    def test_constant_expression(self):
        # A constant value serializes at compile time, into the buffer's
        # initializer.
        code, serial = run_seq_wasm_with_serial(
            "write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, U32(100 + 200))\n"
        )
        assert code == NO_ERROR
        assert serial == [(0, struct.pack(">I", 300))]

    def test_constant_string(self):
        # A string travels with its FwSizeStoreType (U16) length prefix.
        code, serial = run_seq_wasm_with_serial(
            'write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_1, "hello world")\n'
        )
        assert code == NO_ERROR
        assert serial == [(1, struct.pack(">H", 11) + b"hello world")]

    def test_empty_constant_string(self):
        code, serial = run_seq_wasm_with_serial(
            'write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, "")\n'
        )
        assert code == NO_ERROR
        assert serial == [(0, struct.pack(">H", 0))]

    def test_runtime_bool(self):
        code, serial = run_seq_wasm_with_serial(
            "v: bool = True\n"
            "write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, v)\n"
        )
        assert code == NO_ERROR
        assert serial == [(0, bytes([fpy.types.FW_SERIALIZE_TRUE_VALUE]))]

    def test_runtime_signed_int(self):
        code, serial = run_seq_wasm_with_serial(
            "v: I16 = -5\n" "write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, v)\n"
        )
        assert code == NO_ERROR
        assert serial == [(0, struct.pack(">h", -5))]

    def test_runtime_float(self):
        code, serial = run_seq_wasm_with_serial(
            "v: F64 = 0.5\n"
            "write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, v)\n"
        )
        assert code == NO_ERROR
        assert serial == [(0, struct.pack(">d", 0.5))]

    def test_runtime_struct(self):
        code, serial = run_seq_wasm_with_serial(
            "v: Ref.SignalPair = Ref.SignalPair(time=1.0, value=2.0)\n"
            "write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_2, v)\n"
        )
        assert code == NO_ERROR
        assert serial == [(2, struct.pack(">ff", 1.0, 2.0))]

    def test_runtime_array(self):
        code, serial = run_seq_wasm_with_serial(
            "v: Ref.SignalSet = Ref.SignalSet(1.0, 2.0, 3.0, 4.0)\n"
            "write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_3, v)\n"
        )
        assert code == NO_ERROR
        assert serial == [(3, struct.pack(">ffff", 1.0, 2.0, 3.0, 4.0))]

    def test_runtime_enum(self):
        # An enum serializes at its dictionary rep type (Ref.Choice is I32).
        d = load_dictionary(default_dictionary)
        expected = FpyValue(d["type_defs"]["Ref.Choice"], "RED").serialize()
        code, serial = run_seq_wasm_with_serial(
            "v: Ref.Choice = Ref.Choice.RED\n"
            "write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, v)\n"
        )
        assert code == NO_ERROR
        assert serial == [(0, expected)]

    def test_multiple_writes_in_call_order(self):
        code, serial = run_seq_wasm_with_serial(
            "value: U32 = 42\n"
            "write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, value)\n"
            'write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_1, "hi")\n'
            "write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, value)\n"
        )
        assert code == NO_ERROR
        assert serial == [
            (0, struct.pack(">I", 42)),
            (1, struct.pack(">H", 2) + b"hi"),
            (0, struct.pack(">I", 42)),
        ]

    def test_write_before_exit_still_reported(self):
        # The serial_send host call must happen before the sequence terminates.
        code, serial = run_seq_wasm_with_serial(
            "value: U32 = 7\n"
            "write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, value)\n"
            "exit(9)\n"
        )
        assert code == 9
        assert serial == [(0, struct.pack(">I", 7))]

    def test_write_in_loop(self):
        # The same call site's buffer is rewritten each iteration.
        code, serial = run_seq_wasm_with_serial(
            "i: U64 = 0\n"
            "while i < 3:\n"
            "    write_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, i)\n"
            "    i = i + 1\n"
        )
        assert code == NO_ERROR
        assert serial == [(0, struct.pack(">Q", i)) for i in range(3)]


class TestWasmCommands:
    """Command calls lower to the host `cmd(buf ptr, buf len)` call, where the
    buffer holds the big-endian serialized FwOpcodeType followed by the
    fprime-serialized arguments. The runner harness reports each buffer back
    verbatim, so these assert the exact wire bytes against an independent
    struct.pack encoding. Constant arguments are baked into the buffer at
    compile time; runtime arguments are byte-swapped and stored into their
    packed offsets before each dispatch."""

    def _opcode(self, name: str) -> bytes:
        d = load_dictionary(default_dictionary)
        return struct.pack(">I", d["cmd_name_dict"][name].opcode)

    def test_const_no_arg_command(self):
        code, cmds = run_seq_wasm_with_cmds("CdhCore.cmdDisp.CMD_NO_OP()\n")
        assert code == NO_ERROR
        assert cmds == [self._opcode("CdhCore.cmdDisp.CMD_NO_OP")]

    def test_const_string_arg_is_compact(self):
        # A constant string serializes at its actual length (u16 big-endian
        # prefix + bytes), not its declared capacity.
        code, cmds = run_seq_wasm_with_cmds('CdhCore.cmdDisp.CMD_NO_OP_STRING("hi")\n')
        assert code == NO_ERROR
        expected = self._opcode("CdhCore.cmdDisp.CMD_NO_OP_STRING")
        expected += struct.pack(">H", 2) + b"hi"
        assert cmds == [expected]

    def test_empty_string_arg(self):
        # An empty string is just the zero length prefix.
        code, cmds = run_seq_wasm_with_cmds('CdhCore.cmdDisp.CMD_NO_OP_STRING("")\n')
        assert code == NO_ERROR
        expected = self._opcode("CdhCore.cmdDisp.CMD_NO_OP_STRING")
        expected += struct.pack(">H", 0)
        assert cmds == [expected]

    def test_utf8_string_arg_prefix_counts_bytes(self):
        # The length prefix counts encoded utf-8 bytes, not characters:
        # "héllo✓" is 6 characters but 9 bytes.
        code, cmds = run_seq_wasm_with_cmds(
            'CdhCore.cmdDisp.CMD_NO_OP_STRING("héllo✓")\n'
        )
        assert code == NO_ERROR
        data = "héllo✓".encode("utf-8")
        assert len(data) == 9
        expected = self._opcode("CdhCore.cmdDisp.CMD_NO_OP_STRING")
        expected += struct.pack(">H", len(data)) + data
        assert cmds == [expected]

    def test_runtime_scalar_args(self):
        # A negative int pins the big-endian sign bytes; the float pins the
        # bitcast-then-swap path.
        code, cmds = run_seq_wasm_with_cmds(
            "var1: I32 = -2\n"
            "var2: F32 = 1.5\n"
            "var3: U8 = 8\n"
            "CdhCore.cmdDisp.CMD_TEST_CMD_1(var1, var2, var3)\n"
        )
        assert code == NO_ERROR
        expected = self._opcode("CdhCore.cmdDisp.CMD_TEST_CMD_1")
        expected += struct.pack(">ifB", -2, 1.5, 8)
        assert cmds == [expected]

    @pytest.mark.parametrize("flag, byte", [(True, b"\xff"), (False, b"\x00")])
    def test_runtime_bool_arg(self, flag, byte):
        # Bools serialize as the FW_SERIALIZE truth bytes, not 1/0.
        code, cmds = run_seq_wasm_with_cmds(
            "idx: U32 = 3\n"
            f"flag: bool = {flag}\n"
            "Ref.cmdSeq0.SET_BREAKPOINT(idx, flag)\n"
        )
        assert code == NO_ERROR
        expected = self._opcode("Ref.cmdSeq0.SET_BREAKPOINT")
        expected += struct.pack(">I", 3) + byte
        assert cmds == [expected]

    def test_mixed_const_and_runtime_args(self):
        # A compact constant string ahead of a runtime argument pins the
        # runtime argument's offset computation.
        code, cmds = run_seq_wasm_with_cmds(
            "en: Fw.Enabled = Fw.Enabled.ENABLED\n"
            'CdhCore.health.HLTH_PING_ENABLE("task1", en)\n'
        )
        assert code == NO_ERROR
        expected = self._opcode("CdhCore.health.HLTH_PING_ENABLE")
        expected += struct.pack(">H", 5) + b"task1"  # entry: String_40
        expected += struct.pack(">B", 1)  # enable: Fw.Enabled (u8 rep), ENABLED
        assert cmds == [expected]

    def test_runtime_struct_arg_all_scalar_widths(self):
        # A runtime struct argument walks every member; ScalarStruct covers
        # every scalar width, including the 8-byte swaps.
        code, cmds = run_seq_wasm_with_cmds(
            "s: Ref.ScalarStruct = "
            "Ref.ScalarStruct(-1, -2, -3, -4, 1, 2, 3, 4, 1.5, -2.5)\n"
            "Ref.typeDemo.SEND_SCALARS(s)\n"
        )
        assert code == NO_ERROR
        expected = self._opcode("Ref.typeDemo.SEND_SCALARS")
        expected += struct.pack(">bhiqBHIQfd", -1, -2, -3, -4, 1, 2, 3, 4, 1.5, -2.5)
        assert cmds == [expected]

    def test_runtime_array_arg(self):
        # An array of i32-rep enums exercises the element walk.
        code, cmds = run_seq_wasm_with_cmds(
            "c: Ref.ManyChoices = Ref.ManyChoices(Ref.Choice.TWO, Ref.Choice.RED)\n"
            "Ref.typeDemo.CHOICES(c)\n"
        )
        assert code == NO_ERROR
        expected = self._opcode("Ref.typeDemo.CHOICES")
        expected += struct.pack(">ii", 1, 2)  # TWO = 1, RED = 2
        assert cmds == [expected]

    def test_multiple_commands_in_call_order(self):
        code, cmds = run_seq_wasm_with_cmds(
            'CdhCore.cmdDisp.CMD_NO_OP_STRING("a")\n' "CdhCore.cmdDisp.CMD_NO_OP()\n"
        )
        assert code == NO_ERROR
        assert cmds == [
            self._opcode("CdhCore.cmdDisp.CMD_NO_OP_STRING")
            + struct.pack(">H", 1)
            + b"a",
            self._opcode("CdhCore.cmdDisp.CMD_NO_OP"),
        ]

    def test_captured_response_carries_host_value(self):
        # The captured Fw.CmdResponse is the value the host reported, not a
        # canned OK. Only the wasm runner harness can inject a non-OK
        # response for a command that still "succeeds" from the script's
        # point of view (the capture takes responsibility for it).
        code, _ = run_seq_wasm_with_cmds(
            "ret: Fw.CmdResponse = CdhCore.cmdDisp.CMD_NO_OP()\n"
            "assert ret == Fw.CmdResponse.BUSY\n",
            cmd_response=5,  # BUSY
        )
        assert code == NO_ERROR

    def test_recursive_call_in_command_arg_does_not_clobber_buffer(self):
        # A command's buffer is one module global per call site. The third
        # argument's expression recursively dispatches this same call site, so
        # the arguments must all be evaluated before any is stored into the
        # buffer -- or the inner activation would overwrite the slots the
        # outer one had already filled.
        seq = (
            "def f(depth: U64) -> U64:\n"
            "    if depth == 0:\n"
            "        return 0\n"
            "    Ref.sendBuffComp.SB_GEN_FATAL(U32(depth), U32(depth), U32(f(depth - 1)))\n"
            "    return depth\n"
            "f(2)\n"
        )
        code, cmds = run_seq_wasm_with_cmds(seq)
        assert code == NO_ERROR
        opcode = self._opcode("Ref.sendBuffComp.SB_GEN_FATAL")
        # The inner activation dispatches first, then the outer one -- with
        # its own arguments, not the inner one's.
        assert cmds == [
            opcode + struct.pack(">III", 1, 1, 0),
            opcode + struct.pack(">III", 2, 2, 1),
        ]


def _big_endian_cases():
    """Round-trip test values spanning every serializable runtime type: each
    scalar width, both bools, both enum rep widths, and nested aggregates."""
    types = load_dictionary(default_dictionary)["type_defs"]
    choice = types["Ref.Choice"]
    slurry = types["Ref.ChoiceSlurry"]
    slurry_members = {m.name: m.type for m in slurry.members}
    many_choices = types["Ref.ManyChoices"]

    def choice_v(name):
        return FpyValue(choice, name)

    def many(a, b):
        return FpyValue(many_choices, [choice_v(a), choice_v(b)])

    # Asymmetric byte patterns, so a wrong (little-endian) order and a
    # partial/misplaced store are both visible in the bytes.
    scalars = FpyValue(
        types["Ref.ScalarStruct"],
        {
            "i8": FpyValue(I8, -1),
            "i16": FpyValue(I16, -2),
            "i32": FpyValue(I32, -3),
            "i64": FpyValue(I64, -4),
            "u8": FpyValue(U8, 1),
            "u16": FpyValue(U16, 0x0102),
            "u32": FpyValue(U32, 0x01020304),
            "u64": FpyValue(U64, 0x0102030405060708),
            "f32": FpyValue(F32, 1.5),
            "f64": FpyValue(F64, -2.5),
        },
    )
    slurry_val = FpyValue(
        slurry,
        {
            "tooManyChoices": FpyValue(
                slurry_members["tooManyChoices"],
                [many("ONE", "TWO"), many("RED", "BLUE")],
            ),
            "separateChoice": choice_v("RED"),
            "choicePair": FpyValue(
                slurry_members["choicePair"],
                {"firstChoice": choice_v("TWO"), "secondChoice": choice_v("BLUE")},
            ),
            "choiceAsMemberArray": FpyValue(
                slurry_members["choiceAsMemberArray"],
                [FpyValue(U8, 0xAA), FpyValue(U8, 0x55)],
            ),
        },
    )
    return [
        pytest.param(FpyValue(U8, 0xAB), id="u8"),
        pytest.param(FpyValue(I8, -1), id="i8-neg"),
        pytest.param(FpyValue(U16, 0x0102), id="u16"),
        pytest.param(FpyValue(I16, -2), id="i16-neg"),
        pytest.param(FpyValue(U32, 0x01020304), id="u32"),
        pytest.param(FpyValue(I32, -123456789), id="i32-neg"),
        pytest.param(FpyValue(U64, 0x0102030405060708), id="u64"),
        pytest.param(FpyValue(I64, -3_000_000_000), id="i64-neg"),
        pytest.param(FpyValue(F32, 1.5), id="f32"),
        pytest.param(FpyValue(F64, -2.5), id="f64-neg"),
        pytest.param(FpyValue(BOOL, True), id="bool-true"),
        pytest.param(FpyValue(BOOL, False), id="bool-false"),
        pytest.param(FpyValue(types["Fw.Enabled"], "ENABLED"), id="enum-u8-rep"),
        pytest.param(choice_v("RED"), id="enum-i32-rep"),
        pytest.param(scalars, id="struct-every-scalar"),
        pytest.param(many("TWO", "RED"), id="array-of-enums"),
        pytest.param(slurry_val, id="struct-nested-arrays"),
    ]


class TestWasmBigEndianSerialization:
    """_emit_store_big_endian / _emit_load_big_endian translate between LLVM
    values and the fprime wire format in linear memory. FpyValue.serialize()
    is the format oracle: storing a value must produce exactly its bytes
    (pinning big-endianness, tight packing, and the bool truth bytes -- a
    round trip alone couldn't catch both directions agreeing on the wrong
    endianness), and a value loaded from those bytes must store back to them
    (which pins the load as the store's inverse, since serialization is
    injective)."""

    def _emit_check_bytes(self, builder, buf, expected: bytes, exit_code: int):
        """Exit with *exit_code* unless the buffer holds exactly *expected*."""
        i8 = ir.IntType(8)
        i32 = ir.IntType(32)
        ok = ir.Constant(ir.IntType(1), 1)
        for i, byte in enumerate(expected):
            ptr = builder.gep(
                buf, [ir.Constant(i32, 0), ir.Constant(i32, i)], inbounds=True
            )
            ok = builder.and_(
                ok,
                builder.icmp_unsigned(
                    "==", builder.load(ptr, align=1), ir.Constant(i8, byte)
                ),
            )
        fail_block = builder.function.append_basic_block("bytes_bad")
        ok_block = builder.function.append_basic_block("bytes_ok")
        builder.cbranch(ok, ok_block, fail_block)
        builder.position_at_end(fail_block)
        builder.call(
            builder.module.globals[HOST_EXIT_FUNC_NAME],
            [ir.Constant(ERROR_CODE_TYPE, exit_code)],
        )
        builder.unreachable()
        builder.position_at_end(ok_block)

    def _build_module(self, value: FpyValue) -> ir.Module:
        module = ir.Module(name="be_test")
        module.triple = LLVM_TRIPLE
        declare_host_imports(module)
        func = ir.Function(
            module, ir.FunctionType(ir.VoidType(), []), name=FPY_ENTRY_POINT
        )
        builder = ir.IRBuilder(func.append_basic_block("entry"))
        emitter = EmitLlvmExpr(builder)

        expected = value.serialize()
        buf_type = ir.ArrayType(ir.IntType(8), len(expected))

        def byte_buffer(name, init):
            g = ir.GlobalVariable(module, buf_type, name=module.get_unique_name(name))
            g.linkage = "internal"
            g.initializer = ir.Constant(buf_type, bytearray(init))
            return g

        # Route the value through a mutable global and a runtime load, so the
        # walks execute on the interpreter instead of LLVM folding them away.
        src = ir.GlobalVariable(module, value.type.llvm_type, name="src")
        src.linkage = "internal"
        src.initializer = value.llvm_value

        # exit(1): storing the value must produce the oracle bytes.
        buf_store = byte_buffer("buf_store", bytes(len(expected)))
        written = emitter._emit_store_big_endian(
            builder.load(src), value.type, buf_store, 0
        )
        assert written == len(expected) == value.type.max_size
        self._emit_check_bytes(builder, buf_store, expected, exit_code=1)

        # exit(2): loading from the oracle bytes must store back to them.
        buf_in = byte_buffer("buf_in", expected)
        loaded = emitter._emit_load_big_endian(value.type, buf_in, 0)
        buf_out = byte_buffer("buf_out", bytes(len(expected)))
        emitter._emit_store_big_endian(loaded, value.type, buf_out, 0)
        self._emit_check_bytes(builder, buf_out, expected, exit_code=2)

        builder.ret_void()
        return module

    @pytest.mark.parametrize("value", _big_endian_cases())
    def test_round_trip_matches_serialize(self, value):
        code, _, _, _ = run_wasm(llvm_module_to_wasm(self._build_module(value)))
        assert code != 1, f"stored bytes diverged from serialize() for {value}"
        assert code != 2, f"load->store did not round-trip for {value}"
        assert code == NO_ERROR

    @pytest.mark.parametrize("byte", [b"\x01", b"\x7f", b"\xfe"])
    def test_invalid_bool_byte_faults(self, byte):
        # A bool byte that is neither truth value faults with
        # DESERIALIZE_ERROR_INVALID_BOOL, matching the C++ deserializer's
        # FW_DESERIALIZE_FORMAT_ERROR (and FpyValue.deserialize raising
        # DeserializeError).
        module = ir.Module(name="be_test")
        module.triple = LLVM_TRIPLE
        declare_host_imports(module)
        func = ir.Function(
            module, ir.FunctionType(ir.VoidType(), []), name=FPY_ENTRY_POINT
        )
        builder = ir.IRBuilder(func.append_basic_block("entry"))
        emitter = EmitLlvmExpr(builder)

        buf_type = ir.ArrayType(ir.IntType(8), 1)
        buf = ir.GlobalVariable(module, buf_type, name="buf_in")
        buf.linkage = "internal"
        buf.initializer = ir.Constant(buf_type, bytearray(byte))
        emitter._emit_load_big_endian(BOOL, buf, 0)
        builder.ret_void()

        code, _, _, _ = run_wasm(llvm_module_to_wasm(module))
        assert code == DirectiveErrorCode.DESERIALIZE_ERROR_INVALID_BOOL.value
