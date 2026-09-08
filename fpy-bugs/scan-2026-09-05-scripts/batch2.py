from drv import show
from fpy.types import FpyValue, U8, U32, I64, F64
import tempfile, os

# command with runtime struct arg containing bool + floats
show(
    "cmd with runtime struct arg",
    """
s: Ref.FpyExampleStruct = Ref.FpyExampleStruct(True, 7, 1.5)
x: U32 = 0
x = 42
s.count = x
Ref.typeDemo.SEND_SCALARS(Ref.ScalarStruct(-1, -2, -3, -4, 1, 2, 3, 4, 1.5, 2.5))
Ref.typeDemo.SEND_SCALARS(Ref.ScalarStruct(-1, -2, -3, I64(x), 1, 2, 3, 4, 1.5, 2.5))
""",
)

# string command arg with quotes/backslashes -> check the bytes
show(
    "cmd string arg with escaped quote",
    r"""
CdhCore.cmdDisp.CMD_NO_OP_STRING("a\"b")
CdhCore.cmdDisp.CMD_NO_OP_STRING('x"y"z')
CdhCore.cmdDisp.CMD_NO_OP_STRING("path\\to")
""",
)

# element assignment: evaluation order of rhs vs indices with side effects
show(
    "element assign eval order",
    """
arr: Ref.FpyExampleArray = Ref.FpyExampleArray(0, 0, 0)
log_idx: I64 = 0
def idx() -> I64:
    log("idx called")
    return 1
def val() -> U32:
    log("val called")
    return 9
arr[idx()] = val()
assert arr[1] == 9, 20
""",
)

# param array element READ with runtime index inside function
show(
    "param array elem read runtime idx",
    """
def get(a: Ref.FpyExampleArray, i: I64) -> U32:
    return a[i]
arr: Ref.FpyExampleArray = Ref.FpyExampleArray(10, 20, 30)
assert get(arr, 2) == 30, 21
assert get(arr, 0) == 10, 22
""",
)

# nested: global array of structs elem member assign from function w/ runtime idx
show(
    "global array-of-struct member assign runtime idx in func",
    """
ph: Ref.SignalPairSet = Ref.SignalPairSet(Ref.SignalPair(0.0, 0.0), Ref.SignalPair(0.0, 0.0), Ref.SignalPair(0.0, 0.0), Ref.SignalPair(0.0, 0.0))
def setv(i: I64, v: F32):
    ph[i].value = v
setv(2, 3.5)
setv(0, 1.25)
assert ph[2].value == 3.5, 23
assert ph[0].value == 1.25, 24
assert ph[1].value == 0.0, 25
""",
)

# cast divergence runtime
show(
    "cast U8(300.0) runtime",
    """
f: F64 = 300.0
x: U8 = U8(f)
log("done")
assert x == 44, x
""",
)

# sleep_until with a different time context than the clock
show(
    "sleep_until different context",
    """
t: Fw.Time = Fw.Time(TimeBase.TB_NONE, 7, 100, 0)
sleep_until(t)
log("woke")
""",
    time_base=0,
    time_context=3,
    initial_time_us=5_000_000,
)

# exit inside nested function calls
show(
    "exit inside nested call",
    """
def inner(x: U8) -> U8:
    if x == 3:
        exit(42)
    return x
def outer(x: U8) -> U8:
    return inner(x)
y: U8 = outer(1)
y = outer(3)
log("not reached")
""",
)

# seq-run with a time-op argument (resolved_args staleness)
d = tempfile.mkdtemp(prefix="fpyseq-")
from fpy.test_helpers import write_child_seq

write_child_seq(
    """
sequence(d: Fw.TimeInterval, n: U32)
assert d.seconds == 5, 30
assert n == 2, 31
""",
    d,
    "child.bin",
)
show(
    "seq-run with time-op arg",
    """
t0: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 10, 0)
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 15, 0)
n: U32 = 2
Ref.seqDisp.RUN_ARGS("child.bin", Svc.BlockState.BLOCK, t1 - t0, n)
""",
    seq_dir=d,
)
