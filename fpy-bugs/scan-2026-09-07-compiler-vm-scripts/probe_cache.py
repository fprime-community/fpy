import sys, json, tempfile, os

sys.path.insert(0, "src")
from fpy.state import get_base_compile_state
from fpy.compiler import text_to_ast, analyze_ast, analysis_to_fpybc_directives
from fpy.types import SEQ_ARGS, FwSizeStoreType, TIME
import fpy.bytecode.directives as D

src_dict = "test/fpy/RefTopologyDictionary.json"
d = json.load(open(src_dict))
for t in d["typeDefinitions"]:
    if t["qualifiedName"] == "Svc.SeqArgs":
        t["members"]["buffer"]["size"] = 100
    if t["qualifiedName"] == "FwSizeStoreType":
        t["underlyingType"] = {
            "name": "U32",
            "kind": "integer",
            "size": 32,
            "signed": False,
        }
    if False:
        t["underlyingType"] = {
            "name": "U16",
            "kind": "integer",
            "size": 16,
            "signed": False,
        }
mod_dict = "/tmp/claude-1000/ModDictionary.json"
json.dump(d, open(mod_dict, "w"))

prog = 'CdhCore.cmdDisp.CMD_NO_OP_STRING("hi")\n'


def compile_with(dictionary):
    st = get_base_compile_state(dictionary)
    body = text_to_ast(prog)
    analyze_ast(body, st)
    dirs, _ = analysis_to_fpybc_directives(st)
    return dirs


for name, dic in (("A", src_dict), ("B", mod_dict), ("A again", src_dict)):
    dirs = compile_with(dic)
    print(
        f"{name}: SeqArgs buffer len={SEQ_ARGS.members[1].type.length} FwSizeStoreType={FwSizeStoreType.name} FwOpcodeType={D.FwOpcodeType.name} "
        f"cmd bytes={dirs[-1].serialize().hex()}"
    )
