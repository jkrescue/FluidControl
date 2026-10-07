# B04 VTK terminal review

Limited independent review PASS; no HDF curation or training was executed by this review.

Actual unit `fluid-control-p064-b04-long-excitation-vtk-20261007.service`, invocation `81fbe9225839460f9501d880ac56e6e5`, was independently observed at PID 0 / exited / success / exit 0. Approval SHA256: `8bcd7fe0dbcac80b78262d7179e456ad99a26e42cd9eea61753cff7c39396308`.

Result: `artifacts/p064_b04_long_excitation_vtk_view_20261007/vtk_result.json`, SHA256 `4079b1d9365d30f03dfa994be55202289e4989b351d6024ae501d94a771f1a35`.

Independent read-only checks decoded the binary TimeValue header in all 801 internal.vtu files. Values equal the saved result exactly and match 120:0.1:200 within the declared 2e-5 float32 clock tolerance. All 801 complete-file hashes reproduce the saved inventory digest `a5ce1b6f5763e2163927a74a8657382fd448125843b782564080910cfd5e0ac5`. Representative first/middle/last files contain U and p arrays. This is header/inventory validation, not repeated full-mesh sampling or complete field-finiteness validation.

The raw result, both coefficient tables and all 801 force-component property hashes remain equal to the prior independent raw audit. No pre-export complete raw U/p hash inventory existed; this review does not claim independent bytewise before/after proof for every raw field. Export used the reviewed read-only raw mount and isolated writable view. The exact CID recorded in foam.cid is absent. Solver data were not regenerated.

The first lightweight inspection failed before traversal because shell quoting changed a regular expression; correcting that read-only expression produced the successful check in 2.43 seconds. Neither attempt reran export or CFD.

The separately reviewed Curator-only path is ready for explicit approval: source `fc8caf70a797e65e982c99663c7cf6b32886743cf27742014cd1c13f38dc12d3`, supervisor `a3e3a18a359d806bc1b58e631335dc9f855eee20584455da020683a0f4163df2`. It must preserve original VTK clock values for force/action interpolation, reuse normalization bytes, and actually run the official Reader/project DataPipe first/last H100 checks on its output. Existing synthetic fixtures do not substitute for that future evidence.
