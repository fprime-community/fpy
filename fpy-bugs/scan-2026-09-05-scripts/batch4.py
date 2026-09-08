from drv import show
from fpy.types import FpyValue, U8, U32, I64, F64, F32, BOOL
from fpy.test_helpers import load_dictionary, default_dictionary

d = load_dictionary(default_dictionary)
P0 = "Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0"

show(
    "mixed-size params, nested call in arg list, param modification",
    """
def g(x: F64) -> F64:
    return x * 2.0
def f(a: U8, b: F64, c: bool, s: Fw.TimeInterval, e: I16) -> F64:
    a = U8(a + 1)
    s.seconds = 77
    write_to_port(%s, a)
    write_to_port(%s, s)
    write_to_port(%s, c)
    write_to_port(%s, e)
    return b + F64(a)
si: Fw.TimeInterval = Fw.TimeInterval(1, 2)
r: F64 = f(1, g(2.0), True, si, -5)
write_to_port(%s, r)
write_to_port(%s, si)
""" % ((P0,) * 6),
)

show(
    "recursive function with struct return",
    """
def fib_pair(n: U8) -> Fw.TimeInterval:
    if n == 0:
        return Fw.TimeInterval(0, 1)
    p: Fw.TimeInterval = fib_pair(U8(n - 1))
    return Fw.TimeInterval(p.useconds, U32(p.seconds + p.useconds))
r: Fw.TimeInterval = fib_pair(10)
write_to_port(%s, r)
""" % P0,
)

show(
    "local array modified via runtime index, returned",
    """
def mk(k: U32) -> Ref.FpyExampleArray:
    a: Ref.FpyExampleArray = Ref.FpyExampleArray(0, 0, 0)
    for i in 0..3:
        a[i] = U32(k + i)
    return a
x: Ref.FpyExampleArray = mk(10)
write_to_port(%s, x)
write_to_port(%s, mk(100)[1])
""" % (P0, P0),
)

show(
    "bool-returning function in and/or",
    """
calls: U8 = 0
def t() -> bool:
    calls = U8(calls + 1)
    return True
def fl() -> bool:
    calls = U8(calls + 1)
    return False
if fl() and t():
    log("bad1")
if t() or fl():
    log("ok1")
if not fl() and (t() or fl()):
    log("ok2")
write_to_port(%s, calls)
""" % P0,
)

show(
    "2D local array with two runtime indices (read) and struct-member array",
    """
i: I64 = 1
j: I64 = 0
tm: Ref.TooManyChoices = Ref.TooManyChoices(Ref.ManyChoices(Ref.Choice.ONE, Ref.Choice.TWO), Ref.ManyChoices(Ref.Choice.RED, Ref.Choice.BLUE))
write_to_port(%s, tm[i][j])
write_to_port(%s, tm[j][i])
write_to_port(%s, tm[i])
sl: Ref.ChoiceSlurry = Ref.ChoiceSlurry(tm, Ref.Choice.TWO, Ref.ChoicePair(Ref.Choice.ONE, Ref.Choice.RED), [4, 5])
write_to_port(%s, sl.choiceAsMemberArray[i])
write_to_port(%s, sl.tooManyChoices[i][j])
sl.choiceAsMemberArray[j] = 9
write_to_port(%s, sl)
""" % ((P0,) * 6),
)

show(
    "enum/bool telemetry compares",
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
        "Ref.typeDemo.ChoiceCh": bytes([2]),
        "Ref.cmdSeq0.BreakpointInUse": bytes([0xFF]),
        "Ref.cmdSeq0.Debug_ReachedEndOfFile": bytes([0]),
    },
)

show(
    "nested function calls as command args",
    """
def a(x: I32) -> I32:
    return x + 1
def b(x: F32) -> F32:
    return x * 2.0
u: U8 = 3
CdhCore.cmdDisp.CMD_TEST_CMD_1(a(a(5)), b(1.5), u)
CdhCore.cmdDisp.CMD_TEST_CMD_1(a(-1), 0.5, U8(u + a(0)))
""",
)

show(
    "check condition calls a function that runs a bare command",
    """
n: U8 = 0
def poll() -> bool:
    CdhCore.cmdDisp.CMD_NO_OP()
    n = U8(n + 1)
    return n >= 3
check poll() timeout {seconds: 5, useconds: 0} period {seconds: 0, useconds: 100000}:
    log("polled ok")
timeout:
    log("poll timed out")
write_to_port(%s, n)
""" % P0,
)
