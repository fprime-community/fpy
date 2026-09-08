"""Random differential fuzzer: generates programs in a subset of Fpy whose
semantics coincide with Python (no overflow, positive divisors, value-copy
aggregates never aliased), executes the Python rendering as the reference,
then compiles the Fpy rendering for fpybc and wasm and compares outcomes.

Usage: fuzz.py <seed_start> <count> [--keep DIR]
"""

import sys, random, traceback, os, json

sys.path.insert(0, "src")
import fpy.harness as H
from fpy.harness import SequencerHarness, FPY_HARNESS_BINARY, WASM_HARNESS_BINARY

H._fpy_harness = SequencerHarness(FPY_HARNESS_BINARY)
H._wasm_harness = SequencerHarness(WASM_HARNESS_BINARY)
import fpy.test_helpers as T
from fpy.test_helpers import (
    analyze_seq,
    _fpybc_codegen,
    _wasm_codegen,
    run_seq,
    run_wasm,
    CompilationFailed,
    ValidationError,
)

ARR_LEN = 3  # Ref.FpyExampleArray is [3] U32
MOD = 1000


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)
        self.ints = []  # global int var names
        self.bools = []
        self.arrs = []
        self.structs = []  # Ref.ScalarStruct vars: members i64, i32, u8 used
        self.u8s = []  # U8 vars
        self.enums = []  # Ref.Choice vars
        self.slurries = []  # Ref.ChoiceSlurry vars
        self.funcs = []  # (name, [param types], returns_int)
        self.n = 0
        self.oob_prob = 0.03
        self.assert_prob = 0.04

    def fresh(self, p):
        self.n += 1
        return f"{p}{self.n}"

    # ---------------- expressions: return (fpy_text, py_text) ----------------
    def int_expr(self, depth, scope):
        r = self.r
        ints = scope["ints"]
        choices = ["lit", "var"]
        if depth > 0:
            choices += [
                "bin",
                "bin",
                "mod",
                "arr",
                "struct",
                "call",
                "call",
                "u8",
                "slurry",
            ]
        c = r.choice(choices)
        if c == "lit" or (c == "var" and not ints):
            v = r.randint(0, 20)
            return str(v), str(v)
        if c == "var":
            v = r.choice(ints)
            return v, v
        if c == "bin":
            a = self.int_expr(depth - 1, scope)
            b = self.int_expr(depth - 1, scope)
            op = r.choice(["+", "-", "*"])
            return f"(({a[0]} {op} {b[0]}) % {MOD})", f"(({a[1]} {op} {b[1]}) % {MOD})"
        if c == "mod":
            a = self.int_expr(depth - 1, scope)
            b = self.int_expr(depth - 1, scope)
            op = r.choice(["%", "//"])
            return f"({a[0]} {op} ({b[0]} % 9 + 1))", f"({a[1]} {op} ({b[1]} % 9 + 1))"
        if c == "arr" and (scope["arrs"]):
            a = r.choice(scope["arrs"])
            i = self.index_expr(depth - 1, scope)
            return f"I64({a}[{i[0]}])", f"I64({a}[chk({i[1]})])"
        if c == "struct" and scope["structs"]:
            s = r.choice(scope["structs"])
            m = r.choice(["i64", "i32", "u8"])
            if m == "i64":
                return f"{s}.i64", f"{s}.i64"
            return f"I64({s}.{m})", f"I64({s}.{m})"
        if c == "u8" and scope["u8s"]:
            u = r.choice(scope["u8s"])
            return f"I64({u})", f"I64({u})"
        if c == "slurry" and scope["slurries"]:
            s = r.choice(scope["slurries"])
            i = self.int_expr(depth - 1, scope)
            return (
                f"I64({s}.choiceAsMemberArray[({i[0]} % 2)])",
                f"I64({s}.choiceAsMemberArray[({i[1]} % 2)])",
            )
        if c == "call" and scope["callable_funcs"]:
            fs = [f for f in scope["callable_funcs"] if f[2]]
            if fs:
                f = r.choice(fs)
                args = [
                    (
                        self.int_expr(depth - 1, scope)
                        if t == "int"
                        else self.bool_expr(depth - 1, scope)
                    )
                    for t in f[1]
                ]
                return (
                    f"{f[0]}({', '.join(a[0] for a in args)})",
                    f"{f[0]}({', '.join(a[1] for a in args)})",
                )
        v = r.randint(0, 20)
        return str(v), str(v)

    def index_expr(self, depth, scope):
        e = self.int_expr(depth, scope)
        if self.r.random() < self.oob_prob:
            return e  # may go out of bounds
        return f"({e[0]} % {ARR_LEN})", f"({e[1]} % {ARR_LEN})"

    def bool_expr(self, depth, scope):
        r = self.r
        choices = ["lit", "var", "cmp", "cmp", "enumcmp"]
        if depth > 0:
            choices += ["not", "and", "or", "cmp", "enumcmp"]
        c = r.choice(choices)
        if c == "lit" or (c == "var" and not scope["bools"]):
            v = r.choice(["True", "False"])
            return v, v
        if c == "var":
            v = r.choice(scope["bools"])
            return v, v
        if c == "cmp":
            a = self.int_expr(depth - 1, scope)
            b = self.int_expr(depth - 1, scope)
            op = r.choice(["<", "<=", ">", ">=", "==", "!="])
            return f"({a[0]} {op} {b[0]})", f"({a[1]} {op} {b[1]})"
        if c == "not":
            a = self.bool_expr(depth - 1, scope)
            return f"(not {a[0]})", f"(not {a[1]})"
        if c == "enumcmp" and (scope["enums"] or scope["slurries"]):
            lhs = self.enum_expr(depth - 1, scope)
            rhs = self.enum_expr(depth - 1, scope)
            op = r.choice(["==", "!="])
            return f"({lhs[0]} {op} {rhs[0]})", f"({lhs[1]} {op} {rhs[1]})"
        a = self.bool_expr(depth - 1, scope)
        b = self.bool_expr(depth - 1, scope)
        return f"({a[0]} {c} {b[0]})", f"({a[1]} {c} {b[1]})"

    ENUMS = ["ONE", "TWO", "RED", "BLUE"]

    def enum_expr(self, depth, scope):
        r = self.r
        c = r.choice(["lit", "var", "sl1", "sl2", "sl3"])
        if c == "var" and scope["enums"]:
            v = r.choice(scope["enums"])
            return v, v
        if c.startswith("sl") and scope["slurries"]:
            s = r.choice(scope["slurries"])
            if c == "sl1":
                return f"{s}.separateChoice", f"{s}.separateChoice"
            if c == "sl2":
                m = r.choice(["firstChoice", "secondChoice"])
                return f"{s}.choicePair.{m}", f"{s}.choicePair.{m}"
            i = self.int_expr(max(depth - 1, 0), scope)
            j = self.int_expr(max(depth - 1, 0), scope)
            return (
                f"{s}.tooManyChoices[({i[0]} % 2)][({j[0]} % 2)]",
                f"{s}.tooManyChoices[({i[1]} % 2)][({j[1]} % 2)]",
            )
        e = r.choice(self.ENUMS)
        return f"Ref.Choice.{e}", f"Ref.Choice.{e}"

    # ---------------- statements: return (fpy_lines, py_lines) ----------------
    def block(self, depth, scope, ind, in_loop, in_func_ret):
        n = self.r.randint(1, 4)
        fl, pl = [], []
        for _ in range(n):
            a, b = self.stmt(depth, scope, ind, in_loop, in_func_ret)
            fl += a
            pl += b
        if not fl:
            fl, pl = [ind + "pass"], [ind + "pass"]
        return fl, pl

    def stmt(self, depth, scope, ind, in_loop, in_func_ret):
        r = self.r
        choices = [
            "assign_int",
            "assign_int",
            "assign_bool",
            "arr_store",
            "struct_store",
            "call_stmt",
            "u8_store",
            "enum_store",
            "slurry_store",
            "slurry_store",
            "aug",
            "arr_copy",
            "struct_copy",
        ]
        if depth > 0:
            choices += ["if", "if", "while", "for", "while_true"]
        if in_loop:
            choices += ["break", "continue"]
        if in_func_ret is not None:
            choices += ["return"]
        if r.random() < self.assert_prob:
            choices = ["assert"]
        c = r.choice(choices)
        if c == "assign_int" and scope["ints"]:
            v = r.choice(scope["ints"])
            e = self.int_expr(2, scope)
            return [f"{ind}{v} = {e[0]}"], [f"{ind}{v} = {e[1]}"]
        if c == "assign_bool" and scope["bools"]:
            v = r.choice(scope["bools"])
            e = self.bool_expr(2, scope)
            return [f"{ind}{v} = {e[0]}"], [f"{ind}{v} = {e[1]}"]
        if c == "arr_store" and scope["arrs"]:
            a = r.choice(scope["arrs"])
            i = self.index_expr(1, scope)
            e = self.int_expr(2, scope)
            return [f"{ind}{a}[{i[0]}] = U32({e[0]})"], [
                f"{ind}{a}[chk({i[1]})] = U32({e[1]})"
            ]
        if c == "struct_store" and scope["structs"]:
            s = r.choice(scope["structs"])
            m = r.choice(["i64", "i32", "u8"])
            e = self.int_expr(2, scope)
            if m == "i64":
                return [f"{ind}{s}.i64 = {e[0]}"], [f"{ind}{s}.i64 = {e[1]}"]
            if m == "i32":
                return [f"{ind}{s}.i32 = I32({e[0]})"], [f"{ind}{s}.i32 = I32({e[1]})"]
            return [f"{ind}{s}.u8 = U8({e[0]} % 256)"], [
                f"{ind}{s}.u8 = U8({e[1]} % 256)"
            ]
        if c == "u8_store" and scope["u8s"]:
            u = r.choice(scope["u8s"])
            e = self.int_expr(2, scope)
            return [f"{ind}{u} = U8({e[0]} % 256)"], [f"{ind}{u} = U8({e[1]} % 256)"]
        if c == "enum_store" and scope["enums"]:
            v = r.choice(scope["enums"])
            e = self.enum_expr(1, scope)
            return [f"{ind}{v} = {e[0]}"], [f"{ind}{v} = {e[1]}"]
        if c == "slurry_store" and scope["slurries"]:
            s = r.choice(scope["slurries"])
            k = r.choice(["sep", "pair", "tmc", "u8arr"])
            if k == "sep":
                e = self.enum_expr(1, scope)
                return [f"{ind}{s}.separateChoice = {e[0]}"], [
                    f"{ind}{s}.separateChoice = {e[1]}"
                ]
            if k == "pair":
                m = r.choice(["firstChoice", "secondChoice"])
                e = self.enum_expr(1, scope)
                return [f"{ind}{s}.choicePair.{m} = {e[0]}"], [
                    f"{ind}{s}.choicePair.{m} = {e[1]}"
                ]
            if k == "tmc":
                # one constant index and one runtime index (two runtime indices in a store is a known fpybc bug)
                K = r.randint(0, 1)
                j = self.int_expr(1, scope)
                e = self.enum_expr(1, scope)
                if r.random() < 0.5:
                    return (
                        [f"{ind}{s}.tooManyChoices[{K}][({j[0]} % 2)] = {e[0]}"],
                        [f"{ind}{s}.tooManyChoices[{K}][({j[1]} % 2)] = {e[1]}"],
                    )
                return (
                    [f"{ind}{s}.tooManyChoices[({j[0]} % 2)][{K}] = {e[0]}"],
                    [f"{ind}{s}.tooManyChoices[({j[1]} % 2)][{K}] = {e[1]}"],
                )
            i = self.int_expr(1, scope)
            e = self.int_expr(2, scope)
            return (
                [f"{ind}{s}.choiceAsMemberArray[({i[0]} % 2)] = U8({e[0]} % 256)"],
                [f"{ind}{s}.choiceAsMemberArray[({i[1]} % 2)] = U8({e[1]} % 256)"],
            )
        if c == "aug" and scope["ints"]:
            v = r.choice(scope["ints"])
            e = self.int_expr(1, scope)
            op = r.choice(["+=", "-="])
            return [f"{ind}{v} {op} ({e[0]} % 50)"], [f"{ind}{v} {op} ({e[1]} % 50)"]
        if c == "arr_copy" and len(scope["arrs"]) >= 2:
            a, b = r.sample(scope["arrs"], 2)
            return [f"{ind}{a} = {b}"], [f"{ind}{a} = copy.deepcopy({b})"]
        if c == "struct_copy" and len(scope["slurries"]) >= 2:
            a, b = r.sample(scope["slurries"], 2)
            return [f"{ind}{a} = {b}"], [f"{ind}{a} = copy.deepcopy({b})"]
        if c == "while_true":
            if not scope["counters"]:
                return [f"{ind}pass"], [f"{ind}pass"]
            cnt = scope["counters"].pop()
            k = r.randint(1, 4)
            fl = [
                f"{ind}{cnt} = 0",
                f"{ind}while True:",
                f"{ind}    {cnt} = {cnt} + 1",
                f"{ind}    if {cnt} > {k}:",
                f"{ind}        break",
            ]
            pl = list(fl)
            a, b = self.block(depth - 1, scope, ind + "    ", True, in_func_ret)
            return fl + a, pl + b
        if c == "call_stmt" and scope["callable_funcs"]:
            f = r.choice(scope["callable_funcs"])
            args = [
                self.int_expr(1, scope) if t == "int" else self.bool_expr(1, scope)
                for t in f[1]
            ]
            return (
                [f"{ind}{f[0]}({', '.join(a[0] for a in args)})"],
                [f"{ind}{f[0]}({', '.join(a[1] for a in args)})"],
            )
        if c == "if":
            cond = self.bool_expr(2, scope)
            fl = [f"{ind}if {cond[0]}:"]
            pl = [f"{ind}if {cond[1]}:"]
            a, b = self.block(depth - 1, scope, ind + "    ", in_loop, in_func_ret)
            fl += a
            pl += b
            for _ in range(r.randint(0, 2)):
                cond = self.bool_expr(1, scope)
                fl.append(f"{ind}elif {cond[0]}:")
                pl.append(f"{ind}elif {cond[1]}:")
                a, b = self.block(depth - 1, scope, ind + "    ", in_loop, in_func_ret)
                fl += a
                pl += b
            if r.random() < 0.5:
                fl.append(f"{ind}else:")
                pl.append(f"{ind}else:")
                a, b = self.block(depth - 1, scope, ind + "    ", in_loop, in_func_ret)
                fl += a
                pl += b
            return fl, pl
        if c == "while":
            # bounded by a counter that is a pre-declared int var of this frame; never reuse one
            if not scope["counters"]:
                return [f"{ind}pass"], [f"{ind}pass"]
            cnt = scope["counters"].pop()
            k = r.randint(0, 4)
            fl = [
                f"{ind}{cnt} = 0",
                f"{ind}while {cnt} < {k}:",
                f"{ind}    {cnt} = {cnt} + 1",
            ]
            pl = list(fl)
            a, b = self.block(depth - 1, scope, ind + "    ", True, in_func_ret)
            return fl + a, pl + b
        if c == "for":
            lv = self.fresh("i")
            lo = self.int_expr(1, scope)
            hi = self.int_expr(1, scope)
            fl = [f"{ind}for {lv} in ({lo[0]} % 5)..({hi[0]} % 7):"]
            pl = [f"{ind}for {lv} in range(({lo[1]} % 5), ({hi[1]} % 7)):"]
            inner = dict(scope)
            inner["ints"] = scope["ints"] + [lv]
            a, b = self.block(depth - 1, inner, ind + "    ", True, in_func_ret)
            return fl + a, pl + b
        if c == "break":
            return [f"{ind}break"], [f"{ind}break"]
        if c == "continue":
            return [f"{ind}continue"], [f"{ind}continue"]
        if c == "return":
            if in_func_ret == "int":
                e = self.int_expr(2, scope)
                return [f"{ind}return {e[0]}"], [f"{ind}return {e[1]}"]
            return [f"{ind}return"], [f"{ind}return"]
        if c == "assert":
            cond = self.bool_expr(2, scope)
            code = r.randint(100, 200)
            return [f"{ind}assert {cond[0]}, {code}"], [
                f"{ind}assert {cond[1]}, {code}"
            ]
        return [f"{ind}pass"], [f"{ind}pass"]

    # ---------------- program ----------------
    def program(self):
        r = self.r
        fpy, py = [], []
        py += [
            "import copy",
            "class _Fail(Exception):",
            "    def __init__(self, code): self.code = code",
            f"def chk(i):",
            f"    if not (0 <= i < {ARR_LEN}): raise _Fail(11)",
            "    return i",
            "I64 = lambda v: v",
            "U32 = lambda v: v & 0xFFFFFFFF",
            "U8 = lambda v: v & 0xFF",
            "I32 = lambda v: ((v + 2**31) % 2**32) - 2**31",
            "class _S:",
            "    def __init__(self): self.i64 = 0; self.i32 = 0; self.u8 = 0",
            "class _P:",
            "    def __init__(self): self.firstChoice = 'Ref.Choice.ONE'; self.secondChoice = 'Ref.Choice.ONE'",
            "class _Sl:",
            "    def __init__(self):",
            "        self.tooManyChoices = [['Ref.Choice.ONE']*2 for _ in range(2)]; self.separateChoice = 'Ref.Choice.ONE'",
            "        self.choicePair = _P(); self.choiceAsMemberArray = [0, 0]",
        ]
        # enum constants are modelled as their own names (strings) in python
        for e in self.ENUMS:
            pass
        py.append("class Ref:")
        py.append("    class Choice:")
        for e in self.ENUMS:
            py.append(f"        {e} = 'Ref.Choice.{e}'")
        # globals
        for _ in range(r.randint(1, 4)):
            v = self.fresh("g")
            self.ints.append(v)
            x = r.randint(0, 20)
            fpy.append(f"{v}: I64 = {x}")
            py.append(f"{v} = {x}")
        for _ in range(r.randint(0, 2)):
            v = self.fresh("b")
            self.bools.append(v)
            x = r.choice(["True", "False"])
            fpy.append(f"{v}: bool = {x}")
            py.append(f"{v} = {x}")
        for _ in range(r.randint(0, 2)):
            v = self.fresh("a")
            self.arrs.append(v)
            xs = [r.randint(0, 20) for _ in range(ARR_LEN)]
            fpy.append(
                f"{v}: Ref.FpyExampleArray = Ref.FpyExampleArray({', '.join(map(str, xs))})"
            )
            py.append(f"{v} = [{', '.join(map(str, xs))}]")
        for _ in range(r.randint(0, 2)):
            v = self.fresh("s")
            self.structs.append(v)
            fpy.append(
                f"{v}: Ref.ScalarStruct = Ref.ScalarStruct(i64={r.randint(0,20)}, i32={r.randint(0,20)}, u8={r.randint(0,20)})"
            )
            py.append(f"{v} = _S()")
            # python needs member init matching fpy ctor args
            i64, i32, u8 = (
                fpy[-1].split("i64=")[1].split(",")[0],
                fpy[-1].split("i32=")[1].split(",")[0],
                fpy[-1].split("u8=")[1].split(")")[0],
            )
            py.append(f"{v}.i64 = {i64}; {v}.i32 = {i32}; {v}.u8 = {u8}")
        for _ in range(r.randint(0, 2)):
            v = self.fresh("u")
            self.u8s.append(v)
            x = r.randint(0, 255)
            fpy.append(f"{v}: U8 = {x}")
            py.append(f"{v} = {x}")
        for _ in range(r.randint(0, 2)):
            v = self.fresh("e")
            self.enums.append(v)
            x = r.choice(self.ENUMS)
            fpy.append(f"{v}: Ref.Choice = Ref.Choice.{x}")
            py.append(f"{v} = Ref.Choice.{x}")
        for _ in range(r.randint(0, 2)):
            v = self.fresh("sl")
            self.slurries.append(v)
            fpy.append(f"{v}: Ref.ChoiceSlurry = Ref.ChoiceSlurry()")
            py.append(f"{v} = _Sl()")
        # global counters for while loops at top level
        counters = []
        for _ in range(6):
            v = self.fresh("w")
            counters.append(v)
            fpy.append(f"{v}: I64 = 0")
            py.append(f"{v} = 0")
        gscope = {
            "ints": self.ints,
            "bools": self.bools,
            "arrs": self.arrs,
            "structs": self.structs,
            "u8s": self.u8s,
            "enums": self.enums,
            "slurries": self.slurries,
            "counters": list(counters),
            "callable_funcs": [],
        }
        # functions
        for _ in range(r.randint(0, 3)):
            name = self.fresh("f")
            ptypes = [r.choice(["int", "bool"]) for _ in range(r.randint(0, 3))]
            returns_int = r.random() < 0.7
            params = [(self.fresh("p"), t) for t in ptypes]
            fsig = ", ".join(
                f"{p}: {'I64' if t == 'int' else 'bool'}" for p, t in params
            )
            fpy.append(f"def {name}({fsig}){' -> I64' if returns_int else ''}:")
            py.append(f"def {name}({', '.join(p for p, _ in params)}):")
            allg = (
                self.ints
                + self.bools
                + self.arrs
                + self.structs
                + counters
                + self.u8s
                + self.enums
                + self.slurries
            )
            py.append(f"    global {', '.join(allg)}")
            fscope = dict(gscope)
            fscope["ints"] = self.ints + [p for p, t in params if t == "int"]
            fscope["bools"] = self.bools + [p for p, t in params if t == "bool"]
            fscope["callable_funcs"] = list(
                self.funcs
            )  # only earlier funcs: no recursion
            # locals
            locs = []
            for _ in range(r.randint(0, 2)):
                v = self.fresh("l")
                locs.append(v)
                x = r.randint(0, 20)
                fpy.append(f"    {v}: I64 = {x}")
                py.append(f"    {v} = {x}")
            lcs = []
            for _ in range(4):
                lc = self.fresh("c")
                lcs.append(lc)
                fpy.append(f"    {lc}: I64 = 0")
                py.append(f"    {lc} = 0")
            fscope["ints"] = fscope["ints"] + locs
            fscope["counters"] = lcs
            a, b = self.block(
                2, fscope, "    ", False, "int" if returns_int else "void"
            )
            fpy += a
            py += b
            if returns_int:
                e = self.int_expr(2, fscope)
                fpy.append(f"    return {e[0]}")
                py.append(f"    return {e[1]}")
            self.funcs.append((name, ptypes, returns_int))
        gscope["callable_funcs"] = list(self.funcs)
        a, b = self.block(3, gscope, "", False, None)
        fpy += a
        py += b
        return "\n".join(fpy) + "\n", "\n".join(py) + "\n"


def reference(py_src, gen):
    """Run the Python rendering; return ("ok", asserts) or ("fail", code)."""
    ns = {}
    import signal

    def _alarm(*a):
        raise TimeoutError("reference hung")

    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(5)
    try:
        exec(py_src, ns)
    except AssertionError as e:
        return "fail", int(e.args[0])
    except Exception as e:
        if type(e).__name__ == "_Fail":
            return "fail", e.code
        raise
    finally:
        signal.alarm(0)
    asserts = []
    for v in gen.ints:
        asserts.append(f"assert {v} == {ns[v]}, 250")
    for v in gen.bools:
        asserts.append(f"assert {v} == {ns[v]}, 251")
    for v in gen.arrs:
        for k in range(ARR_LEN):
            asserts.append(f"assert {v}[{k}] == {ns[v][k]}, 252")
    for v in gen.u8s:
        asserts.append(f"assert {v} == {ns[v]}, 256")
    for v in gen.enums:
        asserts.append(f"assert {v} == {ns[v]}, 257")
    for v in gen.slurries:
        s = ns[v]
        asserts.append(f"assert {v}.separateChoice == {s.separateChoice}, 258")
        asserts.append(
            f"assert {v}.choicePair.firstChoice == {s.choicePair.firstChoice}, 259"
        )
        asserts.append(
            f"assert {v}.choicePair.secondChoice == {s.choicePair.secondChoice}, 260"
        )
        for i in range(2):
            for j in range(2):
                asserts.append(
                    f"assert {v}.tooManyChoices[{i}][{j}] == {s.tooManyChoices[i][j]}, 261"
                )
            asserts.append(
                f"assert {v}.choiceAsMemberArray[{i}] == {s.choiceAsMemberArray[i]}, 262"
            )
    for v in gen.structs:
        asserts.append(f"assert {v}.i64 == {ns[v].i64}, 253")
        asserts.append(f"assert {v}.i32 == {ns[v].i32}, 254")
        asserts.append(f"assert {v}.u8 == {ns[v].u8}, 255")
    return "ok", asserts


def run_backends(src):
    res = {}
    try:
        state = analyze_seq(src, ignored_warnings=set(T.ALL_WARNINGS))
    except CompilationFailed as e:
        return {"compile": "fail:" + "|".join(str(e).splitlines()[1:2])}
    except Exception as e:
        return {"compile": f"CRASH:{type(e).__name__}:{e}"}
    try:
        dirs, arg_types = _fpybc_codegen(state)
        try:
            run_seq(None, dirs, arg_name_types=arg_types)
            res["fpybc"] = ("ok", 0)
        except ValidationError as e:
            res["fpybc"] = ("validation", str(e)[:200])
        except RuntimeError as e:
            a = e.args[0]
            res["fpybc"] = ("fail", a.value if hasattr(a, "value") else a)
        except Exception as e:
            res["fpybc"] = ("harness", f"{type(e).__name__}:{str(e)[:200]}")
    except CompilationFailed as e:
        res["fpybc"] = ("codegen-fail", "|".join(str(e).splitlines()[1:2]))
    except Exception as e:
        res["fpybc"] = ("codegen-CRASH", f"{type(e).__name__}:{e}")
    try:
        wasm = _wasm_codegen(state)
        try:
            code, _, _, _ = run_wasm(wasm)
            res["wasm"] = ("ok", 0) if code == 0 else ("fail", code)
        except Exception as e:
            res["wasm"] = ("harness", f"{type(e).__name__}:{str(e)[:200]}")
    except CompilationFailed as e:
        res["wasm"] = ("codegen-fail", "|".join(str(e).splitlines()[1:2]))
    except Exception as e:
        res["wasm"] = ("codegen-CRASH", f"{type(e).__name__}:{e}")
    return res


def main():
    start, count = int(sys.argv[1]), int(sys.argv[2])
    keep = None
    if "--keep" in sys.argv:
        keep = sys.argv[sys.argv.index("--keep") + 1]
        os.makedirs(keep, exist_ok=True)
    stats = {"ok": 0, "compile_fail": 0, "mismatch": 0, "crash": 0, "expected_fail": 0}
    for seed in range(start, start + count):
        gen = Gen(seed)
        fpy_src, py_src = gen.program()
        try:
            kind, info = reference(py_src, gen)
        except Exception as e:
            print(f"seed {seed}: reference error {type(e).__name__}: {e}")
            continue
        if kind == "ok":
            fpy_src += "\n".join(info) + "\n"
            expected = ("ok", 0)
        else:
            expected = ("fail", info)
            stats["expected_fail"] += 1
        print(f"seed {seed} ...", flush=True)
        res = run_backends(fpy_src)
        if "compile" in res:
            stats["compile_fail"] += 1
            if res["compile"].startswith("CRASH"):
                stats["crash"] += 1
                print(f"seed {seed}: COMPILER CRASH {res['compile']}")
                if keep:
                    open(f"{keep}/crash_{seed}.fpy", "w").write(fpy_src)
            elif "--verbose" in sys.argv:
                print(f"seed {seed}: compile fail {res['compile']}")
            continue
        bad = False
        if res.get("wasm", ("", ""))[0] == "harness" and "ERR_OUT_OF_MEMORY" in str(
            res["wasm"][1]
        ):
            stats["wasm_oom"] = stats.get("wasm_oom", 0) + 1
            res["wasm"] = expected
        for b in ("fpybc", "wasm"):
            got = res[b]
            if got != expected:
                bad = True
        if bad:
            stats["mismatch"] += 1
            print(
                f"seed {seed}: MISMATCH expected={expected} fpybc={res['fpybc']} wasm={res['wasm']}"
            )
            if keep:
                open(f"{keep}/mismatch_{seed}.fpy", "w").write(fpy_src)
                open(f"{keep}/mismatch_{seed}.py", "w").write(py_src)
        else:
            stats["ok"] += 1
    print("stats", stats)
    H.close_all()


if __name__ == "__main__":
    main()
