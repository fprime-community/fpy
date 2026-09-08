from drv import show
from fpy.error import WarningType

P0 = "Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0"


def S(src):
    return src.replace("P0", P0)


SV = {WarningType.SHADOW_VALUE}

show(
    "nested-block shadowing of same-named variables (top level and function)",
    S("""
x: I64 = 1
if True:
    x: I64 = 2
    write_to_port(P0, x)
    if True:
        x: I64 = 3
        write_to_port(P0, x)
    write_to_port(P0, x)
write_to_port(P0, x)
def f(x: I64) -> I64:
    if x > 0:
        x: I64 = x + 10
        write_to_port(P0, x)
    return x
write_to_port(P0, f(5))
"""),
    expected_warnings=SV,
)

show(
    "for loop var shadowing an outer variable (spec: reuse)",
    S("""
x: I64 = 100
for x in 0..3:
    write_to_port(P0, x)
write_to_port(P0, x)
"""),
    expected_warnings=SV,
)

show(
    "args passed to a sequence without sequence()",
    S("""
log("ran")
"""),
    args=b"\x01\x02",
)

show(
    "sequence() with empty parameter list",
    S("""
sequence()
log("ran")
"""),
)

show(
    "global used in function before declared at top level (declared later)",
    S("""
def f() -> I64:
    return g
g: I64 = 5
write_to_port(P0, f())
"""),
)
