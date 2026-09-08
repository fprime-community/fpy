import sys

sys.path.insert(0, "src")
exec(
    open(
        "/tmp/claude-1000/-home-threelambda-work-fpy/79b97fdc-98d8-46f3-ad72-705fc062bc77/scratchpad/diff.py"
    )
    .read()
    .split("W = lambda")[0]
)

CASES = {
    "all-const scalar cmd": "CdhCore.cmdDisp.CMD_TEST_CMD_1(I32(-5), F32(1.5), U8(7))\n",
    "all-runtime scalar cmd": "a: I32 = I32(-5)\nb: F32 = F32(1.5)\nc: U8 = U8(7)\nCdhCore.cmdDisp.CMD_TEST_CMD_1(a, b, c)\n",
    "mixed: const,runtime,const": "b: F32 = F32(1.5)\nCdhCore.cmdDisp.CMD_TEST_CMD_1(I32(-5), b, U8(7))\n",
    "string const + 2 const": 'CdhCore.health.HLTH_CHNG_PING("abc", U32(1), U32(2))\n',
    "string const + runtime u32": 'w: U32 = U32(11)\nCdhCore.health.HLTH_CHNG_PING("abc", w, U32(2))\n',
    "string const + 2 runtime": 'w: U32 = U32(11)\nf: U32 = U32(22)\nCdhCore.health.HLTH_CHNG_PING("abc", w, f)\n',
    "empty string arg + runtime": 'w: U32 = U32(11)\nCdhCore.health.HLTH_CHNG_PING("", w, U32(2))\n',
    "string + bool runtime": 'b: bool = True\nFileHandling.fileManager.RemoveFile("f.txt", b)\n',
    "array arg const": "Ref.typeDemo.CHOICES_WITH_FRIENDS(U8(1), [Ref.Choice.ONE, Ref.Choice.TWO], U8(2))\n",
    "array arg runtime": "a: Ref.ManyChoices = [Ref.Choice.ONE, Ref.Choice.TWO]\ni: I64 = 0\na[i] = Ref.Choice.TWO\nRef.typeDemo.CHOICES_WITH_FRIENDS(U8(1), a, U8(2))\n",
    "struct arg const": "Ref.typeDemo.CHOICE_PAIR_WITH_FRIENDS(U8(1), {choice_1: Ref.Choice.ONE, choice_2: Ref.Choice.TWO}, U8(2))\n",
    "enum arg runtime": "e: Ref.Choice = Ref.Choice.TWO\nRef.typeDemo.CHOICE(e)\n",
    "cmd response captured": "r: Fw.CmdResponse = CdhCore.cmdDisp.CMD_NO_OP()\nwrite_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, r)\n",
    "two cmds in a row": "CdhCore.cmdDisp.CMD_NO_OP()\nCdhCore.cmdDisp.CMD_TEST_CMD_1(I32(1), F32(2.0), U8(3))\n",
    "cmd inside function": "def f():\n    CdhCore.cmdDisp.CMD_NO_OP()\nf()\n",
    "cmd inside loop": "for i in 0 .. 2:\n    CdhCore.cmdDisp.CMD_NO_OP()\n",
    "cmd with runtime arg in fn": "def f(x: I32):\n    CdhCore.cmdDisp.CMD_TEST_CMD_1(x, F32(1.0), U8(2))\nf(I32(9))\n",
    "flags off then failing cmd": "flags.assert_cmd_success = False\nCdhCore.cmdDisp.CMD_NO_OP()\nwrite_to_port(Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0, U8(1))\n",
}
for name, seq in CASES.items():
    a, b = run_both(seq)
    flag = "" if a == b else "   <<<<<< DIVERGE"
    print(f"--- {name}\n    fpybc: {a}\n    wasm : {b}{flag}")
