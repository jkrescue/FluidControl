# Bounded K4 maintenance receipt extension

Preparation only; no additional cache pass is authorized by this change.
K4 now permits the explicit fixed r1-r12 receipt names in its existing output
directory. K1 remains restricted to its original r1-r4 names. Defaults and
exclusive creation remain unchanged; existing evidence cannot be overwritten.

Only a constant and argument validation changed. The exact44 approved training
file map, no-symlink confinement, same-descriptor hash/stat/advice sequence,
20/20.5/20.75 GiB memory checks and 300-second deadline are unchanged.
There is no timer, automatic execution, global cache clearing, data write,
training restart, numerical change or scientific-admission implication.

Reviewed helper SHA:
`d116b55b67c50d2e618ec4188fe4fc8a4bc8b732deda8ca89f5b942d4b654e25`.
Reviewed tests SHA:
`fc5d53a07c34baa2d90a4b31d2c24e5fa1189b025426241fa37442c8dac4942e`.
Implementation and independent review each passed54 mocked CPU tests in0.04s.
These tests do not read real HDF or issue cache advice. Root must separately
authorize each later pass after inspecting the same live training invocation
and fresh memory readings. Earlier r1-r4 actual receipts are preserved.
