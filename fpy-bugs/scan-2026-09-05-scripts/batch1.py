from drv import show
from fpy.error import WarningType

SH = {WarningType.SHADOW_CALLABLE}

show(
    "string escapes: single-quoted containing double quotes",
    """
log('"hi"')
""",
)
show(
    "string escapes: escaped double quotes",
    r"""
log("say \"hi\"")
""",
)
show(
    "string escapes: backslash n",
    r"""
log("a\nb")
""",
)
show(
    "float literal with underscore",
    """
x: F64 = 1_000.5
assert x == 1000.5, 3
""",
)
show(
    "log empty message",
    """
log("")
""",
)
show(
    "assert exit code 0",
    """
assert False, 0
log("after assert")
""",
)
show(
    "check with user def sleep",
    """
def sleep(seconds: U32 = 0, useconds: U32 = 0):
    log("user sleep called")
x: U8 = 0
check x == 1 timeout {seconds: 1, useconds: 0} period {seconds: 0, useconds: 500000}:
    log("cond true")
timeout:
    log("timed out")
""",
    expected_warnings=SH,
)
show(
    "check with user def now",
    """
def now() -> Fw.Time:
    log("user now called")
    return Fw.Time(TimeBase.TB_WORKSTATION_TIME, 0, 5, 0)
x: U8 = 0
check x == 1 timeout {seconds: 1, useconds: 0} period {seconds: 0, useconds: 500000}:
    log("cond true")
timeout:
    log("timed out")
""",
    expected_warnings=SH,
)
show(
    "check with user def time_cmp (always says GT)",
    """
def time_cmp(lhs: Fw.Time, rhs: Fw.Time) -> Fw.TimeComparison:
    log("user time_cmp called")
    return Fw.TimeComparison.GT
x: U8 = 1
check x == 1 timeout {seconds: 100, useconds: 0}:
    log("cond true")
timeout:
    log("timed out")
""",
    expected_warnings=SH,
)
show(
    "anon struct default arg",
    """
def f(t: Fw.TimeInterval = {seconds: 3, useconds: 0}) -> U32:
    return t.seconds
assert f() == 3, 5
assert f({seconds: 7, useconds: 0}) == 7, 6
""",
)
show(
    "member access on call result",
    """
def mk(a: U32) -> Fw.TimeInterval:
    return Fw.TimeInterval(a, U32(a + 1))
assert mk(4).useconds == 5, 9
assert mk(4).seconds == 4, 10
""",
)
show(
    "global struct modified in function",
    """
g: Fw.TimeInterval = Fw.TimeInterval(1, 2)
def bump():
    g.seconds = U32(g.seconds + 10)
    g.useconds = 99
bump()
assert g.seconds == 11, 11
assert g.useconds == 99, 12
""",
)
show(
    "return inside for loop in function",
    """
g: I64 = 0
def find(limit: I64) -> I64:
    for i in 0..10:
        g = g + 1
        if i == limit:
            return i
    return -1
assert find(3) == 3, 13
assert g == 4, 14
assert find(20) == -1, 15
assert g == 14, 16
""",
)
