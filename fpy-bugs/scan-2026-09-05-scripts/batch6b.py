from drv import show

P0 = "Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0"


def S(src):
    return src.replace("P0", P0)


show(
    "long loop 20000 iterations",
    S("""
acc: I64 = 0
for i in 0..20000:
    acc = acc + (i % 7)
write_to_port(P0, acc)
"""),
)

show(
    "exit(0) inside function then nothing after",
    """
def f():
    exit(0)
    log("unreachable")
f()
log("unreachable2")
""",
)

show(
    "negative exit code",
    """
x: I64 = -5
exit(I32(x))
""",
)

show(
    "while True with break after command",
    S("""
i: I64 = 0
while True:
    CdhCore.cmdDisp.CMD_NO_OP()
    i = i + 1
    if i == 3:
        break
write_to_port(P0, i)
"""),
)

show(
    "struct/array/enum equality with runtime values",
    S("""
a: Fw.TimeInterval = Fw.TimeInterval(1, 2)
b: Fw.TimeInterval = Fw.TimeInterval(1, 2)
b.useconds = U32(b.useconds + 0)
p: Ref.SignalPairSet = Ref.SignalPairSet(Ref.SignalPair(1.0, 2.0), Ref.SignalPair(0.5, 0.25), Ref.SignalPair(3.0, 1.5), Ref.SignalPair(0.0, 0.75))
q: Ref.SignalPairSet = p
q[2].time = F32(3.0)
write_to_port(P0, a == b)
write_to_port(P0, p == q)
q[2].time = F32(3.5)
write_to_port(P0, p != q)
c: Ref.Choice = Ref.Choice.RED
write_to_port(P0, c == Ref.Choice.RED)
write_to_port(P0, c != Ref.Choice.BLUE)
"""),
)

show(
    "time comparison operators at runtime",
    S("""
t0: Fw.Time = now()
sleep(1)
t1: Fw.Time = now()
write_to_port(P0, t1 > t0)
write_to_port(P0, t0 >= t1)
write_to_port(P0, t1 - t0)
write_to_port(P0, (t1 - t0) == {seconds: 1, useconds: 0})
write_to_port(P0, t0 + (t1 - t0) == t1)
"""),
    initial_time_us=7_000_000,
)
