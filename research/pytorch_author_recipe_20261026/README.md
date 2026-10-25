# RealMLP-TD-S: frozen CPU study

Read `docs/research/2026-10-26-pytorch-conclusion.md` for the conclusions and
`2026-10-26-author-recipe-audit.md` for fidelity and explicit adaptations.
This is research tooling, not a promoted contest release.

`source_provenance.json` pins the audited public PyTabKit sources and their
Apache-2.0 license. They are not imported as a runtime dependency. The port uses
the organizer's existing torch==2.5.1 and other listed libraries only.

The evidence directory retains the original freeze, per-arm cost and fit ledgers,
predictions, row lineage, independently recomputed metrics and test receipts.
Trained weights and raw datasets are excluded. Original `outputs/` remains the
working record, including unsuccessful earlier technical attempts.

To repeat in a compatible environment with the provided organizer datasets and
the prepared extra task directories under `outputs/pytorch_inputs`:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/realmlp_equal_budget.py --freeze-new outputs/NEW_UNIQUE_SERIES
python tools/realmlp_equal_budget.py --frozen outputs/NEW_UNIQUE_SERIES/freeze.json --tasks regression classification --phases development control
python tools/realmlp_equal_budget.py --frozen outputs/NEW_UNIQUE_SERIES/freeze.json --tasks adult bank bike har --phases development
```

Input hashes and role metadata are recorded in the freeze/provenance. Extra data
are development evaluation sets; no external test labels are loaded. The loader
does not download data. Use a new output directory; prior arms are not overwritten.
Each complete fresh-process arm has 900 seconds, one thread and a 4096 MiB sampled
RSS watchdog. The final evaluation used 27 arms: organizer development/control,
four extra development tasks and the subsequently admitted HAR control.

`charged_protocol.json`, the freeze and `result.json` are authoritative for this
study. The underlying shared `prepare()` also writes a legacy screening protocol
with old-model configuration fields; those fields are not this port's recipe.
Historical reference replay requires the original saved reference artifacts and
is separate from the timed freshly trained comparison.

No organizer API calls are made. Controls were already viewed or their project-wide
viewing status was uncertain as disclosed in the freeze; they are not new blind
tests. Same-split seed replicas are never counted as independent transfer evidence.
