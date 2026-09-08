"""Structural differential fuzzer: random programs over structs/arrays/functions/
loops with overflow-free arithmetic; compares serial dumps between fpybc and wasm."""

import random, sys, traceback, json, time

sys.path.insert(0, "/home/threelambda/work/fpy/src")
from drv import compile_both, run_fpybc_raw
from fpy.test_helpers import run_wasm, CompilationFailed
from fpy.harness import HarnessError
from fpy.bytecode.directives import DirectiveErrorCode

P0 = "Svc.Fpy.SerialPortIndex.EXAMPLE_PORT_0"
DYADIC = ["0.5", "1.5", "2.0", "0.25", "3.0", "1.0", "0.75"]


class Gen:
    def __init__(self, rng):
        self.r = rng
        self.uid = 0

    def name(self, p):
        self.uid += 1
        return f"{p}{self.uid}"

    # expression generators; `ivars` I64 var names, `fvars` F64 var names, `bvars` bool names,
    # `svars` TimeInterval names, `avars` FpyExampleArray names, funcs: list of (name, kind)
    def iexpr(self, env, depth=0):
        r = self.r
        opts = ["lit", "var", "narrow"]
        if depth < 2:
            opts += [
                "add",
                "sub",
                "mul",
                "div",
                "mod",
                "call",
                "member",
                "elem",
                "cast",
            ]
        c = r.choice(opts)
        if c == "lit" or (c == "var" and not env["i"]):
            return str(r.randint(0, 20))
        if c == "var":
            return r.choice(env["i"])
        if c == "narrow":
            k = r.choice(["n8", "n16", "n32", "u8", "u16"])
            if env.get(k):
                v = r.choice(env[k])
                return v if k.startswith("n") else f"I64({v})"
            return str(r.randint(0, 9))
        if c == "add":
            return f"({self.iexpr(env, depth+1)} + {self.iexpr(env, depth+1)})"
        if c == "sub":
            return f"({self.iexpr(env, depth+1)} - {self.iexpr(env, depth+1)})"
        if c == "mul":
            return f"({self.iexpr(env, depth+1)} * {r.randint(0, 5)})"
        if c == "div":
            return f"({self.iexpr(env, depth+1)} // {r.randint(1, 5)})"
        if c == "mod":
            return f"({self.iexpr(env, depth+1)} % {r.randint(1, 7)})"
        if c == "call" and env["ifuncs"]:
            f = r.choice(env["ifuncs"])
            return f"{f}({self.iexpr(env, depth+1)}, {self.sexpr(env, depth+1)}, {self.aexpr(env, depth+1)})"
        if c == "member" and env["s"]:
            return f"I64({r.choice(env['s'])}.{r.choice(['seconds','useconds'])})"
        if c == "elem" and env["a"]:
            return f"I64({r.choice(env['a'])}[{self.idx(env, depth+1)}])"
        if c == "cast" and env["f"]:
            return f"I64({r.choice(env['f'])} * 4.0)"  # dyadic float -> exact int
        return str(r.randint(0, 9))

    def idx(self, env, depth=0):
        # index in 0..2
        return f"(({self.iexpr(env, depth)}) % 3)"

    def fexpr(self, env, depth=0):
        r = self.r
        opts = ["lit", "var", "f32"]
        if depth < 2:
            opts += ["add", "mul", "sub", "cast", "pair"]
        c = r.choice(opts)
        if c == "lit" or (c == "var" and not env["f"]):
            return r.choice(DYADIC)
        if c == "var":
            return r.choice(env["f"])
        if c == "f32":
            if env.get("f32"):
                return r.choice(env["f32"])
            return r.choice(DYADIC)
        if c == "add":
            return f"({self.fexpr(env, depth+1)} + {self.fexpr(env, depth+1)})"
        if c == "sub":
            return f"({self.fexpr(env, depth+1)} - {self.fexpr(env, depth+1)})"
        if c == "mul":
            return f"({self.fexpr(env, depth+1)} * {r.choice(['0.5','2.0','1.5'])})"
        if c == "cast":
            return f"F64({self.iexpr(env, depth+1)} % 64)"
        if c == "pair" and env["p"]:
            return f"F64({r.choice(env['p'])}[{self.idx4(env, depth+1)}].{r.choice(['time','value'])})"
        return r.choice(DYADIC)

    def idx4(self, env, depth=0):
        return f"(({self.iexpr(env, depth)}) % 4)"

    def bexpr(self, env, depth=0):
        r = self.r
        c = r.choice(["cmp", "cmp", "fcmp", "var", "and", "or", "not", "lit", "bcall"])
        if c == "cmp":
            return f"({self.iexpr(env, depth+1)} {r.choice(['<','<=','==','!=','>','>='])} {self.iexpr(env, depth+1)})"
        if c == "fcmp":
            return f"({self.fexpr(env, depth+1)} {r.choice(['<','<=','==','!=','>'])} {self.fexpr(env, depth+1)})"
        if c == "var" and env["b"]:
            return r.choice(env["b"])
        if c == "and" and depth < 2:
            return f"({self.bexpr(env, depth+1)} and {self.bexpr(env, depth+1)})"
        if c == "or" and depth < 2:
            return f"({self.bexpr(env, depth+1)} or {self.bexpr(env, depth+1)})"
        if c == "not" and depth < 2:
            return f"(not {self.bexpr(env, depth+1)})"
        if c == "bcall" and env["bfuncs"]:
            return f"{r.choice(env['bfuncs'])}({self.iexpr(env, depth+1)})"
        return r.choice(["True", "False"])

    def sexpr(self, env, depth=0):
        r = self.r
        c = r.choice(["ctor", "var", "anon", "scall"])
        if c == "var" and env["s"]:
            return r.choice(env["s"])
        if c == "scall" and env["sfuncs"] and depth < 2:
            return f"{r.choice(env['sfuncs'])}({self.iexpr(env, depth+1)}, {self.sexpr(env, depth+1)})"
        if c == "anon" and depth < 2:
            return f"{{seconds: U32({self.iexpr(env, depth+1)} % 1000), useconds: U32({self.iexpr(env, depth+1)} % 1000)}}"
        return f"Fw.TimeInterval(U32({self.iexpr(env, depth+1)} % 1000), U32({self.iexpr(env, depth+1)} % 1000))"

    def aexpr(self, env, depth=0):
        r = self.r
        c = r.choice(["ctor", "var", "anon", "acall"])
        if c == "var" and env["a"]:
            return r.choice(env["a"])
        if c == "acall" and env["afuncs"] and depth < 2:
            return f"{r.choice(env['afuncs'])}({self.aexpr(env, depth+1)}, {self.iexpr(env, depth+1)})"
        if c == "anon" and depth < 2:
            return f"[U32({self.iexpr(env, depth+1)} % 500), U32({self.iexpr(env, depth+1)} % 500), U32({self.iexpr(env, depth+1)} % 500)]"
        return f"Ref.FpyExampleArray(U32({self.iexpr(env, depth+1)} % 500), U32({self.iexpr(env, depth+1)} % 500), U32({self.iexpr(env, depth+1)} % 500))"

    def pexpr(self, env, depth=0):
        r = self.r
        if env["p"] and r.random() < 0.5:
            return r.choice(env["p"])
        pairs = ", ".join(
            f"Ref.SignalPair(F32({self.fexpr(env, depth+1)}), F32({self.fexpr(env, depth+1)}))"
            for _ in range(4)
        )
        return f"Ref.SignalPairSet({pairs})"

    def stmt(self, env, ind, depth, in_func, ret_kind):
        r = self.r
        pad = "    " * ind
        opts = [
            "iassign",
            "fassign",
            "bassign",
            "sassign",
            "aassign",
            "smember",
            "aelem",
            "pelem",
            "dump",
            "if",
            "for",
            "while",
            "decl",
            "ndecl",
            "nassign",
            "ndump",
        ]
        if in_func and ret_kind and depth < 2:
            opts.append("ret")
        if depth >= 2:
            opts = [o for o in opts if o not in ("if", "for", "while")]
        c = r.choice(opts)
        if c == "iassign" and [v for v in env["i"] if v not in env.get("ro", [])]:
            return f"{pad}{r.choice([v for v in env['i'] if v not in env.get('ro', [])])} = {self.iexpr(env)}\n"
        if c == "fassign" and env["f"]:
            return f"{pad}{r.choice(env['f'])} = {self.fexpr(env)}\n"
        if c == "bassign" and env["b"]:
            return f"{pad}{r.choice(env['b'])} = {self.bexpr(env)}\n"
        if c == "sassign" and env["s"]:
            return f"{pad}{r.choice(env['s'])} = {self.sexpr(env)}\n"
        if c == "aassign" and env["a"]:
            return f"{pad}{r.choice(env['a'])} = {self.aexpr(env)}\n"
        if c == "smember" and env["s"]:
            return f"{pad}{r.choice(env['s'])}.{r.choice(['seconds','useconds'])} = U32({self.iexpr(env)} % 1000)\n"
        if c == "aelem" and [v for v in env["a"] if v != "a"]:
            return f"{pad}{r.choice([v for v in env['a'] if v != 'a'])}[{self.idx(env)}] = U32({self.iexpr(env)} % 500)\n"
        if c == "pelem" and env["p"]:
            if r.random() < 0.5:
                return f"{pad}{r.choice(env['p'])}[{self.idx4(env)}].{r.choice(['time','value'])} = F32({self.fexpr(env)})\n"
            return f"{pad}{r.choice(env['p'])}[{self.idx4(env)}] = Ref.SignalPair(F32({self.fexpr(env)}), F32({self.fexpr(env)}))\n"
        if c == "dump":
            kind = r.choice(["i", "f", "b", "s", "a", "p"])
            if env[kind]:
                return f"{pad}write_to_port({P0}, {r.choice(env[kind])})\n"
            return f"{pad}write_to_port({P0}, {self.iexpr(env)})\n"
        if c == "if":
            s = f"{pad}if {self.bexpr(env)}:\n" + self.block(
                env, ind + 1, depth + 1, in_func, ret_kind
            )
            if r.random() < 0.5:
                s += f"{pad}elif {self.bexpr(env)}:\n" + self.block(
                    env, ind + 1, depth + 1, in_func, ret_kind
                )
            if r.random() < 0.5:
                s += f"{pad}else:\n" + self.block(
                    env, ind + 1, depth + 1, in_func, ret_kind
                )
            return s
        if c == "for":
            lv = self.name("k")
            env2 = dict(env)
            env2["i"] = env["i"] + [lv]
            body = self.block(env2, ind + 1, depth + 1, in_func, ret_kind, loop=True)
            return f"{pad}for {lv} in {r.randint(0,2)}..{r.randint(1,4)}:\n" + body
        if c == "while":
            cv = self.name("w")
            env2 = dict(env)
            env2["i"] = env["i"] + [cv]
            env2["ro"] = env.get("ro", []) + [cv]
            body = self.block(env2, ind + 1, depth + 1, in_func, ret_kind, loop=True)
            return (
                f"{pad}{cv}: I64 = 0\n{pad}while {cv} < {r.randint(1,3)}:\n{pad}    {cv} = {cv} + 1\n"
                + body
            )
        if c == "decl":
            kind = r.choice(["i", "f", "b", "s", "a"])
            nm = self.name(kind + "v")
            if kind == "i":
                e = self.iexpr(env)
                env["i"].append(nm)
                return f"{pad}{nm}: I64 = {e}\n"
            if kind == "f":
                e = self.fexpr(env)
                env["f"].append(nm)
                return f"{pad}{nm}: F64 = {e}\n"
            if kind == "b":
                e = self.bexpr(env)
                env["b"].append(nm)
                return f"{pad}{nm}: bool = {e}\n"
            if kind == "s":
                e = self.sexpr(env)
                env["s"].append(nm)
                return f"{pad}{nm}: Fw.TimeInterval = {e}\n"
            e = self.aexpr(env)
            env["a"].append(nm)
            return f"{pad}{nm}: Ref.FpyExampleArray = {e}\n"
        if c == "ret":
            return f"{pad}return {self.retexpr(env, ret_kind)}\n"
        if c == "ndecl":
            k = r.choice(["n8", "n16", "n32", "u8", "u16", "f32"])
            nm = self.name(k + "v")
            e = self.nexpr(env, k)
            env.setdefault(k, [])
            env[k] = env[k] + [nm]
            return f"{pad}{nm}: {self.ntype(k)} = {e}\n"
        if c == "nassign":
            k = r.choice(["n8", "n16", "n32", "u8", "u16", "f32"])
            if env.get(k):
                return f"{pad}{r.choice(env[k])} = {self.nexpr(env, k)}\n"
            return f"{pad}pass\n"
        if c == "ndump":
            k = r.choice(["n8", "n16", "n32", "u8", "u16", "f32"])
            if env.get(k):
                return f"{pad}write_to_port({P0}, {r.choice(env[k])})\n"
            return f"{pad}pass\n"
        return f"{pad}pass\n"

    def ntype(self, k):
        return {
            "n8": "I8",
            "n16": "I16",
            "n32": "I32",
            "u8": "U8",
            "u16": "U16",
            "f32": "F32",
        }[k]

    def nexpr(self, env, k):
        t = self.ntype(k)
        if k == "f32":
            return f"F32({self.fexpr(env)})"
        if k.startswith("u"):
            return f"{t}(({self.iexpr(env)}) % 200)"
        return f"{t}((({self.iexpr(env)}) % 200) - 100)"

    def retexpr(self, env, kind):
        return {"i": self.iexpr, "b": self.bexpr, "s": self.sexpr, "a": self.aexpr}[
            kind
        ](env)

    def block(self, env, ind, depth, in_func, ret_kind, loop=False):
        # copy var lists so declarations inside the block stay block-scoped
        env = {k: (list(v) if isinstance(v, list) else v) for k, v in env.items()}
        n = self.r.randint(1, 3)
        s = "".join(self.stmt(env, ind, depth, in_func, ret_kind) for _ in range(n))
        if loop and self.r.random() < 0.3:
            s += "    " * ind + self.r.choice(["break", "continue"]) + "\n"
        return s

    def program(self):
        r = self.r
        env = {
            "i": [],
            "f": [],
            "b": [],
            "s": [],
            "a": [],
            "p": [],
            "ifuncs": [],
            "bfuncs": [],
            "sfuncs": [],
            "afuncs": [],
        }
        src = ""
        # globals
        for k in range(r.randint(1, 3)):
            nm = self.name("gi")
            env["i"].append(nm)
            src += f"{nm}: I64 = {r.randint(0, 30)}\n"
        for k in range(r.randint(1, 2)):
            nm = self.name("gf")
            env["f"].append(nm)
            src += f"{nm}: F64 = {r.choice(DYADIC)}\n"
        nm = self.name("gb")
        env["b"].append(nm)
        src += f"{nm}: bool = {r.choice(['True','False'])}\n"
        nm = self.name("gs")
        env["s"].append(nm)
        src += f"{nm}: Fw.TimeInterval = Fw.TimeInterval({r.randint(0,9)}, {r.randint(0,9)})\n"
        nm = self.name("ga")
        env["a"].append(nm)
        src += f"{nm}: Ref.FpyExampleArray = Ref.FpyExampleArray({r.randint(0,9)}, {r.randint(0,9)}, {r.randint(0,9)})\n"
        nm = self.name("gp")
        env["p"].append(nm)
        src += f"{nm}: Ref.SignalPairSet = Ref.SignalPairSet(Ref.SignalPair(1.0, 2.0), Ref.SignalPair(0.5, 0.25), Ref.SignalPair(3.0, 1.5), Ref.SignalPair(0.0, 0.75))\n"
        # functions (defined after globals; bodies may use globals)
        nf = r.randint(1, 3)
        funcs = []
        for k in range(nf):
            kind = r.choice(["i", "i", "b", "s", "a"])
            fname = self.name("fn")
            if kind == "i":
                params = "n: I64, t: Fw.TimeInterval, a: Ref.FpyExampleArray"
                ret = "I64"
                fenv = dict(env)
                fenv["i"] = env["i"] + ["n"]
                fenv["s"] = env["s"] + ["t"]
                fenv["a"] = env["a"] + ["a"]
            elif kind == "b":
                params = "n: I64"
                ret = "bool"
                fenv = dict(env)
                fenv["i"] = env["i"] + ["n"]
            elif kind == "s":
                params = "n: I64, t: Fw.TimeInterval"
                ret = "Fw.TimeInterval"
                fenv = dict(env)
                fenv["i"] = env["i"] + ["n"]
                fenv["s"] = env["s"] + ["t"]
            else:
                params = "a: Ref.FpyExampleArray, n: I64"
                ret = "Ref.FpyExampleArray"
                fenv = dict(env)
                fenv["i"] = env["i"] + ["n"]
                fenv["a"] = env["a"] + ["a"]
            # functions may call previously defined ones only (avoid recursion blowups)
            body = self.block(fenv, 1, 1, True, kind)
            body += "    return " + self.retexpr(fenv, kind) + "\n"
            src += f"def {fname}({params}) -> {ret}:\n{body}"
            env[
                {"i": "ifuncs", "b": "bfuncs", "s": "sfuncs", "a": "afuncs"}[kind]
            ].append(fname)
        # main
        for k in range(r.randint(3, 8)):
            src += self.stmt(env, 0, 0, False, None)
        # final dumps of all globals
        for kind in ("i", "f", "b", "s", "a", "p"):
            for v in env[kind]:
                if v.startswith("g"):
                    src += f"write_to_port({P0}, {v})\n"
        return src


def run_one(seed):
    rng = random.Random(seed)
    src = Gen(rng).program()
    try:
        state, out = compile_both(src)
    except CompilationFailed as e:
        return ("compile_failed", src, str(e)[:300])
    except Exception as e:
        return ("compile_crash", src, traceback.format_exc()[-1500:])
    res = {}
    for be in ("fpybc", "wasm"):
        if isinstance(out[be], Exception):
            return (
                "codegen_crash_" + be,
                src,
                "".join(traceback.format_exception(out[be]))[-1500:],
            )
    try:
        dirs, arg_types = out["fpybc"]
        r = run_fpybc_raw(dirs, arg_types)
        if r.get("error"):
            res["fpybc"] = ("harness_error", r["error"])
        elif r.get("cmdResponse") == 0:
            res["fpybc"] = ("ok", [s["data"] for s in r.get("serial", [])])
        else:
            res["fpybc"] = (
                "fail",
                DirectiveErrorCode(r["lastDirectiveError"]).name,
                r.get("exitCode"),
                [s["data"] for s in r.get("serial", [])],
            )
    except HarnessError as e:
        res["fpybc"] = ("harness_error", str(e)[:500])
    try:
        code, events, cmds, serial = run_wasm(out["wasm"])
        if code == 0:
            res["wasm"] = ("ok", [s[1].hex() for s in serial])
        else:
            res["wasm"] = ("fail", code, [s[1].hex() for s in serial])
    except HarnessError as e:
        res["wasm"] = ("harness_error", str(e)[:500])
    a, b = res["fpybc"], res["wasm"]
    if a[0] == "ok" and b[0] == "ok" and a[1] == b[1]:
        return ("match", src, None)
    return ("MISMATCH", src, json.dumps(res, default=str)[:3000])


if __name__ == "__main__":
    start, n = int(sys.argv[1]), int(sys.argv[2])
    counts = {}
    with open(
        f"/tmp/claude-1000/-home-threelambda-work-fpy/a25d082e-2182-454f-b015-144536e04ae2/scratchpad/fuzz_{start}.log",
        "w",
    ) as log:
        for seed in range(start, start + n):
            kind, src, info = run_one(seed)
            counts[kind] = counts.get(kind, 0) + 1
            if kind not in ("match",):
                log.write(
                    f"\n########## seed={seed} {kind}\n{src}\n---- info ----\n{info}\n"
                )
                log.flush()
            if seed % 25 == 0:
                log.write(f"\n[progress seed={seed} counts={counts}]\n")
                log.flush()
        log.write(f"\nFINAL counts={counts}\n")
    print(counts)
