from drv import show
from fpy.types import FpyValue, U8, U32, I64, F64, F32, BOOL
import tempfile
from fpy.test_helpers import write_child_seq, load_dictionary, default_dictionary

d = load_dictionary(default_dictionary)
P0 = "Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0"

show(
    "cast U8(300.0) runtime",
    """
f: F64 = 300.0
x: U8 = U8(f)
write_to_port(%s, x)
y: I8 = I8(f)
write_to_port(%s, y)
g: F64 = -3.7
z: U8 = U8(g)
write_to_port(%s, z)
""" % (P0, P0, P0),
)

show(
    "write_to_port various runtime values",
    """
s: Ref.FpyExampleMulti = Ref.FpyExampleMulti(Ref.FpyExampleEnum.B, True, 2.5)
b: bool = s.boolVal
s.floatVal = 7.25
write_to_port(%s, s)
write_to_port(%s, s.enumVal)
write_to_port(%s, b)
write_to_port(%s, s.floatVal)
arr: Ref.SignalPairSet = Ref.SignalPairSet(Ref.SignalPair(1.0, 2.0), Ref.SignalPair(3.0, 4.0), Ref.SignalPair(5.0, 6.0), Ref.SignalPair(7.0, 8.0))
i: I64 = 2
write_to_port(%s, arr[i])
write_to_port(%s, arr[i].value)
write_to_port(%s, arr)
""" % ((P0,) * 7),
)

show(
    "check inside function called twice, and in a loop",
    """
n: U8 = 0
def waiter(lim: U8) -> U8:
    check n >= lim timeout {seconds: 1, useconds: 0} period {seconds: 0, useconds: 100000}:
        return 1
    timeout:
        return 0
n = 5
assert waiter(3) == 1, 40
assert waiter(9) == 0, 41
for i in 0..3:
    check n == 5 timeout {seconds: 1, useconds: 0}:
        n = U8(n + 0)
    timeout:
        exit(42)
log("checks done")
""",
)

show(
    "seq args used/modified in function",
    """
sequence(a: U8, s: Ref.FpyExampleStruct, t: Fw.Time)
def f() -> U32:
    s.count = U32(s.count + a)
    return s.count
assert f() == 12, 50
assert f() == 14, 51
assert s.flag == True, 52
assert t.seconds == 1234, 53
write_to_port(%s, s)
""" % P0,
    args=b"".join(
        [
            FpyValue(U8, 2).serialize(),
            FpyValue(
                d["type_defs"]["Ref.FpyExampleStruct"],
                {
                    "flag": FpyValue(BOOL, True),
                    "count": FpyValue(U32, 10),
                    "ratio": FpyValue(F32, 0.5),
                },
            ).serialize(),
            FpyValue(
                d["type_defs"]["Fw.TimeValue"],
                {
                    "timeBase": FpyValue(d["type_defs"]["TimeBase"], "TB_NONE"),
                    "timeContext": FpyValue(U8, 0),
                    "seconds": FpyValue(U32, 1234),
                    "useconds": FpyValue(U32, 5),
                },
            ).serialize(),
        ]
    ),
)

show(
    "cmd response captured and flags",
    """
r: Fw.CmdResponse = CdhCore.cmdDisp.CMD_NO_OP()
assert r == Fw.CmdResponse.EXECUTION_ERROR, 60
flags.assert_cmd_success = False
CdhCore.cmdDisp.CMD_NO_OP()
log("survived failing cmd")
def g():
    CdhCore.cmdDisp.CMD_CLEAR_TRACKING()
    log("in g after cmd")
g()
flags.assert_cmd_success = True
g()
log("not reached")
""",
    failing_opcodes={
        d["cmd_name_dict"]["CdhCore.cmdDisp.CMD_NO_OP"].opcode,
        d["cmd_name_dict"]["CdhCore.cmdDisp.CMD_CLEAR_TRACKING"].opcode,
    },
)

slurry = d["type_defs"]["Ref.ChoiceSlurry"]
choice = d["type_defs"]["Ref.Choice"]
many = d["type_defs"]["Ref.ManyChoices"]
toomany = d["type_defs"]["Ref.TooManyChoices"]
pair = d["type_defs"]["Ref.ChoicePair"]
arr2 = d["type_defs"]["Array_U8_2"]
sl = FpyValue(
    slurry,
    {
        "tooManyChoices": FpyValue(
            toomany,
            [
                FpyValue(many, [FpyValue(choice, "ONE"), FpyValue(choice, "TWO")]),
                FpyValue(many, [FpyValue(choice, "RED"), FpyValue(choice, "ONE")]),
            ],
        ),
        "separateChoice": FpyValue(choice, "TWO"),
        "choicePair": FpyValue(
            pair,
            {
                "firstChoice": FpyValue(choice, "RED"),
                "secondChoice": FpyValue(choice, "TWO"),
            },
        ),
        "choiceAsMemberArray": FpyValue(arr2, [FpyValue(U8, 9), FpyValue(U8, 11)]),
    },
)
print("Ref.Choice enum:", choice.enum_dict)
show(
    "tlm struct with member array + nested array; prm struct",
    """
i: I64 = 1
assert Ref.typeDemo.ChoiceSlurryCh.choiceAsMemberArray[1] == 11, 70
assert Ref.typeDemo.ChoiceSlurryCh.choiceAsMemberArray[i] == 11, 71
assert Ref.typeDemo.ChoiceSlurryCh.tooManyChoices[1][0] == Ref.Choice.RED, 72
assert Ref.typeDemo.ChoiceSlurryCh.tooManyChoices[i][i] == Ref.Choice.ONE, 73
assert Ref.typeDemo.ChoiceSlurryCh.choicePair.secondChoice == Ref.Choice.TWO, 74
x: Ref.ChoiceSlurry = Ref.typeDemo.ChoiceSlurryCh
x.tooManyChoices[i][0] = Ref.Choice.TWO
write_to_port(%s, x)
assert Ref.typeDemo.CHOICE_PAIR_PRM.firstChoice == Ref.Choice.RED, 75
write_to_port(%s, Ref.typeDemo.CHOICE_PAIR_PRM)
""" % (P0, P0),
    tlm={"Ref.typeDemo.ChoiceSlurryCh": sl.serialize()},
    prms={
        "Ref.typeDemo.CHOICE_PAIR_PRM": FpyValue(
            pair,
            {
                "firstChoice": FpyValue(choice, "RED"),
                "secondChoice": FpyValue(choice, "ONE"),
            },
        ).serialize()
    },
)

# seq-run with time-op arg, retry with materialized children
dd = tempfile.mkdtemp(prefix="fpyseq-")
write_child_seq(
    """
sequence(d: Fw.TimeInterval, n: U32)
assert d.seconds == 5, 30
assert n == 2, 31
""",
    dd,
    "child.bin",
)
show(
    "seq-run with time-op arg (retry)",
    """
t0: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 10, 0)
t1: Fw.Time = Fw.Time(TimeBase.TB_NONE, 0, 15, 0)
n: U32 = 2
Ref.seqDisp.RUN_ARGS("child.bin", Svc.BlockState.BLOCK, t1 - t0, n)
""",
    seq_dir=dd,
)
