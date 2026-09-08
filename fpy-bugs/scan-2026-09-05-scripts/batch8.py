from drv import show

P0 = "Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0"


def S(src):
    return src.replace("P0", P0)


show(
    "log message longer than FW_LOG_STRING_MAX_SIZE",
    """
log("%s")
""" % ("A" * 300),
)

show(
    "write_to_port with string literal and bool literal",
    S("""
write_to_port(P0, "hi")
write_to_port(P0, True)
write_to_port(P0, 5)
write_to_port(P0, 1.5)
"""),
)

show(
    "sleep_until with different time base than clock",
    S("""
t: Fw.Time = Fw.Time(TimeBase.TB_PROC_TIME, 0, 100, 0)
sleep_until(t)
log("woke")
"""),
    time_base=2,
    initial_time_us=5_000_000,
)

show(
    "assert exit code from tlm read (runtime) and code > 255",
    S("""
c: U32 = CdhCore.cmdDisp.CommandsDispatched
assert c == 3, I32(c)
"""),
    tlm={"CdhCore.cmdDisp.CommandsDispatched": bytes([0, 0, 1, 0x2C])},
)
