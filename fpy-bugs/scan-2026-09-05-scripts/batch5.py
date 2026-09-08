from drv import show
from fpy.types import FpyValue, U8, U32, I64, F64, F32, BOOL
from fpy.test_helpers import load_dictionary, default_dictionary

d = load_dictionary(default_dictionary)
P0 = "Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0"
choice = d["type_defs"]["Ref.Choice"]
print("Ref.Choice rep:", choice.rep_type, choice.enum_dict)

show(
    "enum/bool telemetry compares (correct sizes)",
    """
if Ref.typeDemo.ChoiceCh == Ref.Choice.RED:
    log("red")
if Ref.cmdSeq0.BreakpointInUse:
    log("bp in use")
if not Ref.cmdSeq0.Debug_ReachedEndOfFile:
    log("not eof")
x: Ref.Choice = Ref.typeDemo.ChoiceCh
write_to_port(%s, x)
""" % P0,
    tlm={
        "Ref.typeDemo.ChoiceCh": FpyValue(choice, "RED").serialize(),
        "Ref.cmdSeq0.BreakpointInUse": bytes([0xFF]),
        "Ref.cmdSeq0.Debug_ReachedEndOfFile": bytes([0]),
    },
)

show(
    "tlm value smaller than dictionary type (stale dictionary)",
    """
x: Ref.Choice = Ref.typeDemo.ChoiceCh
y: U8 = 7
write_to_port(%s, x)
write_to_port(%s, y)
""" % (P0, P0),
    tlm={"Ref.typeDemo.ChoiceCh": bytes([2])},
)

show(
    "tlm value larger than dictionary type (stale dictionary)",
    """
c: U32 = CdhCore.cmdDisp.CommandsDispatched
y: U8 = 7
write_to_port(%s, c)
write_to_port(%s, y)
""" % (P0, P0),
    tlm={"CdhCore.cmdDisp.CommandsDispatched": bytes([0, 0, 0, 5, 0, 0, 0, 0])},
)

show(
    "local array modified via runtime index, returned",
    """
def mk(k: U32) -> Ref.FpyExampleArray:
    a: Ref.FpyExampleArray = Ref.FpyExampleArray(0, 0, 0)
    for i in 0..3:
        a[i] = U32(k + U32(i))
    return a
x: Ref.FpyExampleArray = mk(10)
write_to_port(%s, x)
write_to_port(%s, mk(100)[1])
""" % (P0, P0),
)

show(
    "nested function calls as command args",
    """
def a(x: I32) -> I32:
    return I32(x + 1)
def b(x: F32) -> F32:
    return F32(x * 2.0)
u: U8 = 3
CdhCore.cmdDisp.CMD_TEST_CMD_1(a(a(5)), b(1.5), u)
CdhCore.cmdDisp.CMD_TEST_CMD_1(a(-1), 0.5, U8(u + U8(a(0))))
""",
)
