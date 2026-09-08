CASES = {}
# author edits with tab=4: line 4 (8 spaces) looks nested inside `if b`; fpy reads tab=8 so it belongs to `if a`
CASES["mixed_tabs_spaces_silent_renest"] = (
    "a: bool = True\nb: bool = False\nn: I64 = 0\nif a:\n\tif b:\n\t\tn = n + 1\n        n = n + 10\nassert n == 0, 1\n"
)
CASES["mixed_tabs_spaces_silent_renest2"] = (
    "a: bool = True\nb: bool = False\nn: I64 = 0\nif a:\n\tif b:\n\t\tn = n + 1\n        n = n + 10\nassert n == 10, 2\n"
)
