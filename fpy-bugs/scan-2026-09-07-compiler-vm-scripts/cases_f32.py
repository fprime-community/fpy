CASES = {}
CASES["f32_literal_cmp"] = """
x: F32 = 0.1
assert x == 0.1, 1
"""
CASES["f32_var_cmp_f32_var"] = """
x: F32 = 0.1
y: F32 = 0.1
assert x == y, 2
z: F64 = x
assert z == x, 3
"""
CASES["f32_struct_member_cmp"] = """
s: Ref.SignalPair = {time: 0.1, value: 0.2}
assert s.time == 0.1, 4
"""
CASES["f32_widen_narrow_roundtrip"] = """
x: F32 = 1.5
y: F64 = x
assert y == 1.5, 5
z: F32 = F32(y * 2)
assert z == 3.0, 6
w: I64 = I64(z)
assert w == 3, 7
u: U8 = U8(z + 0.5)
assert u == 3, 8
"""
