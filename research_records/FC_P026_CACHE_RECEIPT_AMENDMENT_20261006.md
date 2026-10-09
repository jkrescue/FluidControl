# P026 K1 clean-cache receipt-name amendment

Engineering-only amendment, reviewed by Lead before integration. The original
r1 helper SHA `5ba8916f0f3fbbfbfd9ac538e3990b8068ffd526701b74278e849280de3ab8d5`
and actual r1 receipt SHA
`792d0402237346b6f22b114f895f68e0bae9e3210c99576d2657b55247399537`
remain historical evidence, unchanged.

The amended helper SHA is
`94d3c43b5b4341f4fa541dd9c630b5dad83785720f65ca092a1728b2fa4c7cb8`.
Its sole behavior change is optional `--receipt-name` selecting exactly
`cache_advice_20261006_r1.jsonl`, `cache_advice_20261006_r2.jsonl`, or
`cache_advice_20261006_r3.jsonl`, always inside the existing exact K1 output
directory. Default remains r1; each selected output is created exclusively.
An existing receipt aborts before any data/cache operation.

All file-path constraints, exact44 audit SHA, read-only same-descriptor hashing,
1 MiB chunks, before/after file identities, immediate clean-cache advice,
300-second bound, initial20.75 GiB free, pre-file20.5 GiB free, and continuous
dual20 GiB checks are unchanged. There is no loop, scheduling, automatic trigger,
or permission to change the running trainer or its numerical protocol.

Implementation and independent Lead runs each passed11 tiny CPU tests in0.03s.
Tests use software metadata fixtures only, reject arbitrary paths and r4, preserve
the default, and prove each existing receipt cannot be repeated. Test SHA
`5a7ae9a2cb0923f1da75a3d8c41b3b83d77d66521a5152425224733f2fc25624`.

Lead separately authorized one r2 pass to be executed by Lead after canonical
hash confirmation. This document is not an r2 completion receipt. The implementer
did not execute a pass. r3 remains unauthorized; all future execution requires
separate explicit approval. No scientific threshold, model, data content, PPO,
or admission changes are made.
