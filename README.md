# ViewTrainer

**Train any PyTorch classifier so its augmented views must agree — then predict with the same panel: consensus probability, view-spread uncertainty, per-view votes, and a defer-to-human verdict, behind one small `fit` / `evaluate` / `predict` Trainer.**

ViewTrainer is the train-time sibling of [ViewConsensus](https://github.com/habib-analyst/viewconsensus), which wraps an already-trained classifier at inference time. ViewConsensus models guess how views *would* vote; ViewTrainer models are *trained* so the votes mean something: the training loss includes the panel's agreement, and prediction reuses the identical panel and the identical consensus semantics (mean probability, winning-class spread as uncertainty, retained per-view votes).

## Install

```bash
pip install viewtrainer
```

Requires Python 3.10+, PyTorch 2+, and NumPy. Nothing else — no Lightning, no training framework, no experiment tracker.

## Quickstart

```python
import torch
from torch import nn
from viewtrainer import ConsensusTrainer

model = nn.Sequential(  # any nn.Module that returns class logits
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(16 * 12 * 12, 4),
)

trainer = ConsensusTrainer(model, labels=["disc", "square", "triangle", "ring"])
history = trainer.fit(train_loader, val_loader=val_loader, epochs=8)

metrics = trainer.evaluate(test_loader)
print(metrics.summary())
# one line: accuracy | loss | mean uncertainty |
#           defer rate -> accepted accuracy | AURC

# Fit the defer threshold on held-out data, then predict with verdicts.
trainer.fit_defer_threshold(val_loader, target_accepted_accuracy=0.95)
pred = trainer.predict(image)          # one (C, H, W) tensor
print(pred.label, pred.probability, pred.verdict)
print(pred.explain())                  # every view's vote, kept
```

`predict` returns a `Prediction` with `label`, `probability` (consensus), `uncertainty` (winning-class spread across views), `agreement`, `votes`, `verdict` (`"confident"` / `"unsure"`), and `should_defer`. A batch returns a list of them.

## How it works

For every training input, each view in the panel (default: identity, flip, shift, brightness, contrast, small rotation — all pure PyTorch, all label-preserving for typical shape/object tasks) is forwarded through your model. The loss is:

```
loss = cross-entropy  +  consensus_weight × consensus loss
```

The consensus loss is the KL divergence between each view's probabilities and the panel consensus (mean probability, detached), **masked by default to samples the consensus already gets right** — the confidence-masking idea FixMatch uses — so agreement is tightened where the panel is right instead of forcing confident agreement on ambiguous samples, which would destroy the spread signal the defer rule depends on. Set `consensus_mask="none"` to apply it everywhere, `consensus_weight=0` to ablate it entirely, and `task_loss="consensus"` to put cross-entropy on the mean probability instead of averaging it per view. The default weight is a gentle `0.1` on purpose: in the demo sweep (see the build report), a weight of 1.0 measurably cost accuracy on a small task. The term is a dial for tasks where view agreement itself matters, not a guaranteed accuracy win — the demo prints the panel-only arm separately so you can see exactly what it adds on the demo task.

At predict time the same panel votes again. Samples whose winning class wobbles across views get high uncertainty; `fit_defer_threshold` picks, on held-out data, the threshold with the lowest defer rate that reaches your target accepted accuracy, and `evaluate` reports the full risk–coverage curve and AURC, not just accuracy.

Already have a trained model? Wrap it for inference only with [ViewConsensus](https://github.com/habib-analyst/viewconsensus) — `trainer.predict_view_probabilities(x)` emits exactly the `[N, V, C]` panel output it consumes, and both libraries define uncertainty identically.

## Measured demo (synthetic, reproducible)

`examples/consensus_training_demo.py` trains three identical CNNs from the same initial weights on a synthetic four-shape task with a deliberately faint, hard subset: plain training, the view panel alone (`consensus_weight=0`), and full consensus training — then measures accuracy, AURC, risk at 80% coverage, error separation by view-spread, and deferral. Run it and read the real numbers it prints; the summary JSON lands in `examples/output/demo_summary.json`. It is a mechanics demo on synthetic data with fixed seeds, not a benchmark, and the script prints whatever it measures.

## What it is not

- **Not a new consistency-regularization method.** The objective is in the Π-model / FixMatch lineage and says so. The product is the unified contract — one panel, one loss, one prediction object with votes, spread, and a fitted defer rule — not a claim that consistency training was invented here.
- **Not a semi-supervised learning framework.** ViewTrainer trains on labeled data only. For unlabeled-data consistency methods (FixMatch, Mean Teacher, …) use TorchSSL.
- **Not an uncertainty-quantification suite.** No ensembles, Bayesian layers, calibration, or conformal coverage. For those, use TorchUncertainty or MAPIE. ViewTrainer's uncertainty is one specific, inspectable signal: how much the winning class moves across augmented views of the same input.
- **Not a training platform.** No distributed training, mixed precision, checkpoint scheduling, or logging backends. It is a small wrapper around a plain PyTorch loop you can read in one sitting; bring your own infrastructure if you need it.
- **Not a deferral guarantee.** `fit_defer_threshold` is defer-threshold fitting on held-out data, reported honestly (including when the target is not met). It is not a learned rejector in the SelectiveNet sense and carries no coverage guarantee.

## Panels are your responsibility

Views must be label-preserving for your task. A horizontal flip is a fine view for shapes and a terrible one for left/right classes or text. The default panel targets float image tensors in roughly `[0, 1]`; for anything else, pass your own callables:

```python
trainer = ConsensusTrainer(model, views=[identity_view(), my_domain_view])
```

## License

MIT. Issues and pull requests are welcome on GitHub.
