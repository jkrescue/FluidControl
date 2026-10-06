# FC-P064 formal resource revision — preparation only

This is an engineering resource proposal, not execution approval and not a
scientific protocol change.  The unchanged numerical runner remains SHA
`03c5862e34a648a1254284d1709bd74c3b995d3a91ae06c4a6f92a945029c0f3`;
all data, split, horizons, windows, 75 tests and gates remain unchanged.

## Why an explicit revision is required

The historical runner admits and monitors with
`min(MemFree, MemAvailable) >= 20 GiB`.  On the UMA host, `MemFree` excludes
reclaimable page cache and was about 5.65 GiB while `MemAvailable` was about
112.99 GiB during the concurrently running CPU CFD.  Treating that cache as
unavailable would reject before science without showing physical pressure.
This proposal does not mutate the historical runner, fake `MemFree`, call
`drop_caches`, or perform automatic cache advice.

## Proposed P064-only outer resource contract

- outer systemd cgroup: fixed memory maximum 72 GiB, swap maximum 0, CPU quota
  800%, TasksMax 1024, RuntimeMaxSec 10800 and TimeoutStopSec at least 120;
- startup: two fresh samples two seconds apart, each with physical
  `MemAvailable >= 50 GiB`;
- continuous two-second monitoring through every container: physical
  `MemAvailable >= 22 GiB`, preserving at least the existing 20 GiB user
  reserve plus 2 GiB operational margin;
- GPU stages retain allocator fraction `.06`, GPU 0 only, and the existing
  no-competing-GPU check; CPU stages retain no GPU visibility;
- each reviewed container remains network-none/read-only, exact read-only
  inputs, one exclusive output mount, fixed image, bounded pids, exact CID
  inspection and owned cleanup;
- persist every resource observation and the exact cgroup/container terminal
  evidence before claiming completion.

The historical inner `min(MemFree, MemAvailable)` check must not be silently
bypassed.  Before execution, a separately reviewed P064 orchestration adapter
must replace that check with the above explicit `MemAvailable` contract while
preserving the command plan and numerical runner bytes.  Until that adapter
and a final approval exist, source/proof preparation cannot launch formal.

## Concurrency and budget

The observed paired-CFD controller cgroup used under 1 GiB at the read-only
checkpoint, but its two solver containers and I/O remain external consumers.
Formal is expected to require roughly 36–37 minutes from prior matched runs.
Concurrent launch is allowed only if the new startup samples, cgroup limits,
GPU-idle check and exclusive output all pass at launch time; the current high
`MemAvailable` alone is not execution authorization.
