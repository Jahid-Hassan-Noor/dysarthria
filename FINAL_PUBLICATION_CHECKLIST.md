# Final publication checklist

Audit date: 2026-11-14. Checked items have evidence in [REPRODUCIBILITY_AUDIT.md](REPRODUCIBILITY_AUDIT.md). A local PASS does not imply a public-clone or remote-CI PASS.

- [x] Repository paths validated: `common.ROOT`, documented V10.4 commands and local preflight passed.
- [x] Dependency versions checked: 16 exact pins installed locally; `pip check` passed. A fresh full install remains untested.
- [x] Test suite executed: 16 local tests passed; 10 synthetic tests passed in a fresh lightweight environment.
- [x] CI status accurately documented: remote CI is NOT TESTED; the workflow is local because the GitHub token lacks `workflow` scope.
- [x] Frozen artifact hashes verified where accessible: 159 manifest entries and representative V10 hashes matched.
- [x] V10 reference pipeline status documented: saved results validated read-only; fresh training/evaluation NOT TESTED.
- [x] V10.4 pipeline status documented: `run_v10_4.py all` passed locally; fresh public clone BLOCKED by missing artifacts.
- [x] Abstract interpretation corrected in proposed replacement text; PDF integration pending.
- [x] Table 7 reviewed and proposed footnote corrected; PDF integration pending.
- [x] Table 10 reviewed and proposed compact table corrected; PDF integration pending.
- [x] Conclusion interpretation corrected in proposed replacement text; PDF integration pending.
- [x] Primary manuscript and supplementary results checked against saved outputs; one versioned TORGO interval difference documented. Full line-by-line check is pending.
- [ ] Final manuscript generated: matching editable source is unavailable.
- [x] Remaining limitations reported, including licensed data, missing public artifacts, remote CI and manuscript source.

No original model weights, frozen reference results, seeds or speaker partitions were changed.
