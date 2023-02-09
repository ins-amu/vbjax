from . import test_loops, test_regmap, test_shtlc, test_coupling

test_mods = [test_loops, test_regmap, test_shtlc, test_coupling]

for mod in test_mods:
    for key in dir(mod):
        if key.startswith('test_'):
            test_fn = getattr(mod, key)
            test_fn()

print('vbjax tests passed')