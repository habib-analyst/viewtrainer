# Changelog

## 0.1.0 — 2026-10-28

Initial release.

- `ConsensusTrainer`: wraps any PyTorch classifier with consensus
  training behind `fit` / `evaluate` / `predict`.
- Training objective: cross-entropy plus a KL consensus loss between
  each view's probabilities and the detached panel consensus, masked by
  default to samples the consensus already classifies correctly
  (`consensus_mask="correct"`; `"none"` applies it everywhere).
  Default `consensus_weight` is a gentle 0.1, set from the measured
  demo sweep in REPORT.md (1.0 cost accuracy on the demo task).
- Default six-view panel of pure-PyTorch label-preserving views
  (identity, flip, shift, brightness, contrast, rotation) plus a
  deterministic seeded noise view; custom callables accepted.
- `Prediction` with consensus probability, winning-class view-spread
  uncertainty, view agreement, retained per-view votes, and a
  confident / unsure defer verdict; `explain()` renders the votes.
- `evaluate` with risk–coverage points and AURC; `fit_defer_threshold`
  fits the defer threshold on held-out data to a target accepted
  accuracy and honestly reports unmet targets.
- `plain_trainer` ablation helper (identity view, zero consensus
  weight) and `consensus_from_probabilities` / `consensus_batch` core
  with semantics matched to ViewConsensus, the inference-time sibling.
- `save` / `load` with a panel-mismatch guard.
