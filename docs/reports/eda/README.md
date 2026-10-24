# EDA verification reports

These reports distinguish completed execution, measured model behavior, and unverified model interpretation. External development datasets, credentials, full network logs and model artifacts are excluded from Git.

- [Initial four-task autonomous review](2026-10-24-initial.md)
- [Completion and transport regression review](2026-10-24-completion.md)
- [Historical temporal v2 and negative results](2026-10-24-temporal-v2.md)
- [Universal research: final assessment and requirements](2026-10-24-universal.md)
- [Sanitized machine-readable evidence](2026-10-24-universal-evidence.json)
- [Test artifacts, failures and resolutions](2026-10-24-test-summary.json)

Every current EDA run also writes REPORT.md and evaluation_report.json into its own output directory. Published evidence is a sanitized summary of those local artifacts, not a copy of the training data or a claim of competition acceptance.

The separate [participant-example reference study](../examples/2026-10-25-reference.md) solves both supplied ML examples and publishes predictions. It measures expert-selected model recipes, not autonomous EDA decisions.
