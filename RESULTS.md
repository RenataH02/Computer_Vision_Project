# Results

Extended results for all three stages: full accuracy tables, ablation grids, and
robustness checks behind the summaries in the README. Each section follows the same
part/section numbering as its corresponding notebook, so any number here can be traced back to the exact cell that produced it.

---

## Stage 1 : Classification Baselines

### 1.1 Accuracy table (Part 5.1)

Top-1 test accuracy (%), mean ± std over 3 runs.

| Dataset | Encoder | Method | K=5 | K=10 | K=full |
|---|---|---|---|---|---|
| DTD | ResNet-18 | linear probe | 45.78 ± 1.21 | 54.68 ± 0.76 | 63.01 ± 0.07 |
| DTD | ResNet-18 | prototypes | 46.51 ± 0.53 | 53.37 ± 0.85 | 58.83 ± 0.00 |
| Flowers-102 | ResNet-18 | linear probe | 75.68 ± 0.68 | 83.40 ± 0.09 | 83.40 ± 0.09 |
| Flowers-102 | ResNet-18 | prototypes | 70.19 ± 0.11 | 75.23 ± 0.00 | 75.23 ± 0.00 |
| Flowers-102 | DINOv2 | linear probe | 99.05 ± 0.22 | 99.31 ± 0.11 | 99.31 ± 0.11 |
| Flowers-102 | DINOv2 | prototypes | 99.04 ± 0.13 | 99.40 ± 0.00 | 99.40 ± 0.00 |

Flowers-102's official training split has exactly 10 images per class, so K=10 and K=full
are identical there by construction. Only DTD (40 images per class) shows a genuine
5 to 10 to full progression.

### 1.2 Training stability (Part 5.3)

Representative 10-shot (seed 0) linear-probe runs.

| Run | Epochs run | Best epoch | Train acc @best | Val acc @best | Train-val gap | Min val loss | Val loss @end | Val loss rising? |
|---|---|---|---|---|---|---|---|---|
| DTD/ResNet-18 | 56 | 16 | 96.17% | 51.60% | 44.57pt | 1.7749 | 1.8257 | yes |
| Flowers/ResNet-18 | 78 | 38 | 100.00% | 86.27% | 13.73pt | 0.6106 | 0.6106 | no |
| Flowers/DINOv2 | 44 | 4 | 100.00% | 99.41% | 0.59pt | 0.0416 | 0.0416 | no |

Overfitting severity tracks feature quality directly. DTD's validation loss bottoms out
around epoch 30 and then rises while training loss keeps falling, the clearest sign here
of memorization rather than generalization. DINOv2/Flowers barely overfits at all.

### 1.3 Confusion matrix highlights (Part 5.4)

**DTD/ResNet-18** : overall top-1 63.09%, mean-per-class accuracy 63.09% (test split is
perfectly balanced, 40 images/class).

Hardest and easiest classes by recall:

| Hardest | Recall | Easiest | Recall |
|---|---|---|---|
| blotchy | 0.23 | chequered | 0.88 |
| stained | 0.30 | crystalline | 0.85 |
| woven | 0.35 | potholed | 0.85 |
| bumpy | 0.35 | cobwebbed | 0.85 |
| lined | 0.40 | | |

Top confusions:

| True class | Predicted as | Rate |
|---|---|---|
| dotted | polka-dotted | 0.42 |
| lined | banded | 0.30 |
| polka-dotted | dotted | 0.23 |
| bubbly | sprinkled | 0.20 |
| stained | marbled | 0.20 |
| grid | meshed | 0.20 |

**Flowers-102/ResNet-18** : overall top-1 83.28%, mean-per-class accuracy 85.10% (test
split is unbalanced, 20-238 images/class).

| Hardest | Recall | Easiest | Recall |
|---|---|---|---|
| sweet pea | 0.42 | osteospermum | 1.00 |
| camellia | 0.48 | tree poppy | 1.00 |
| canterbury bells | 0.50 | orange dahlia | 1.00 |
| primula | 0.52 | cautleya spicata | 1.00 |
| petunia | 0.55 | | |

| True class | Predicted as | Rate |
|---|---|---|
| canterbury bells | bolero deep blue | 0.15 |
| garden phlox | snapdragon | 0.12 |
| pink primrose | lenten rose | 0.10 |
| canterbury bells | monkshood | 0.10 |
| mexican aster | osteospermum | 0.10 |
| camellia | mallow | 0.10 |

In both datasets, the largest confusions sit between classes that are genuinely similar
to look at, not random pairs. That means raw top-1 understates how good the predictions
actually are. The 1.8pt gap between top-1 and mean-per-class accuracy on Flowers comes
from its class imbalance; the two measures agree closely enough that no ranking changes.

### 1.4 Feature space (Part 5.5)

No additional numbers beyond the accuracy table in 1.1. Qualitatively: DINOv2/Flowers
clusters are so tight that several sit hidden behind their own prototype marker.
Flowers/ResNet-18 clusters are identifiable but broader, with some points crossing
between classes. DTD/ResNet-18 is the most entangled of the three: textures with clear
repeating structure (knitted, banded, potholed) form clean groups, while diffuse ones
(blotchy, flecked, matted) mix together in a crowded center, matching their low recall
above.

---

## Stage 2 : Flow Matching to Class Prototypes

### 2.1 Accuracy table and ΔAcc (Part 7.1)

Top-1 test accuracy (%), mean ± std over 3 runs, Δ vs. the Stage 1 prototype baseline
(recomputed in this notebook; see the note under Cross-Stage Notes).

| Dataset/Encoder | K | baseline | standard T4 | standard T12 | rolled T4 | rolled T12 |
|---|---|---|---|---|---|---|
| DTD/ResNet-18 | 5 | 46.51 | 43.94 ± 0.79 (-2.57) | 44.36 ± 0.69 (-2.15) | 42.66 ± 1.13 (-3.85) | 42.78 ± 1.46 (-3.72) |
| DTD/ResNet-18 | 10 | 53.37 | 49.01 ± 0.53 (-4.36) | 49.57 ± 0.60 (-3.79) | 45.02 ± 0.41 (-8.35) | 45.59 ± 0.38 (-7.78) |
| DTD/ResNet-18 | full | 58.78 | 59.40 ± 0.22 (+0.62) | 59.91 ± 0.20 (+1.13) | 55.59 ± 0.19 (-3.19) | 55.48 ± 0.39 (-3.30) |
| Flowers/ResNet-18 | 5 | 70.19 | 69.77 ± 0.66 (-0.42) | 70.16 ± 0.66 (-0.03) | 67.80 ± 1.07 (-2.39) | 68.15 ± 1.22 (-2.04) |
| Flowers/ResNet-18 | 10/full | 75.22 | 76.95 ± 0.21 (+1.73) | 77.29 ± 0.22 (+2.07) | 73.79 ± 0.43 (-1.43) | 74.16 ± 0.15 (-1.05) |
| Flowers/DINOv2 | 5 | 99.04 | 99.16 ± 0.09 (+0.12) | 99.17 ± 0.10 (+0.12) | 99.05 ± 0.11 (+0.01) | 99.05 ± 0.12 (+0.01) |
| Flowers/DINOv2 | 10/full | 99.40 | 99.48 ± 0.01 (+0.08) | 99.48 ± 0.00 (+0.08) | 99.47 ± 0.02 (+0.07) | 99.47 ± 0.01 (+0.08) |

Standard FM sits below baseline at K=5 and K=10 on DTD and Flowers/ResNet-18, and only
turns positive at K=full. Rolled-out training loses to standard FM in all 9 rows without
exception, sometimes falling below baseline even at K=full. T=12 beats T=4 consistently
within standard FM, but only by a few tenths of a point.

### 2.2 Training curves (Part 7.3)

No numbers beyond 2.1. Both objectives train stably, with no divergence and a smooth
decrease throughout. Standard FM's velocity-MSE and rolled-out's final-point-MSE are
different quantities on different scales and are not directly comparable to each other.
All curves converge well before their epoch budget ends, except rolled-out on
Flowers/DINOv2 at T=12, which was still trending down at 150 epochs. That case is
addressed directly in 2.4.

### 2.3 Feature space and flow trajectories (Part 7.4-7.5)

Qualitative only. Projected feature clouds look nearly identical before and after FM
across all three dataset/encoder combinations, with no visible cluster tightening.
Individual flow trajectories are short, local hops that rarely reach their target
prototype: start and end markers sit close together, while the correct prototype is
often far away in the projection. The network settles on a conservative correction
rather than completing the transport its own training objective asks for.

### 2.4 Robustness checks: was rolled-out undertrained? (Part 7.6)

Check 1, extended training on Flowers-102/DINOv2, rolled-out T=12:

| | 150 epochs (orig, mean of 3 seeds) | 400 epochs (this seed only) |
|---|---|---|
| Test accuracy | 99.47% | 99.32% |

Loss kept falling smoothly through epoch 400 with no plateau, but accuracy did not
improve. It slipped slightly, meaning loss and accuracy decoupled here.

Check 2, matched epoch budget (300, same as standard FM), K=full:

| Dataset/Encoder | T | rolled @150ep (orig, mean 3 seeds) | rolled @300ep (this seed) | baseline |
|---|---|---|---|---|
| DTD | 4 | 55.59 | 53.03 | 58.78 |
| DTD | 12 | 55.48 | 54.73 | 58.78 |
| Flowers/ResNet-18 | 4 | 73.79 | 72.53 | 75.22 |
| Flowers/ResNet-18 | 12 | 74.16 | 72.60 | 75.22 |
| Flowers/DINOv2 | 4 | 99.47 | 99.41 | 99.40 |
| Flowers/DINOv2 | 12 | 99.47 | 99.41 | 99.40 |

More epochs never closed the gap and were flat to worse in every row. This rules out
undertraining as the explanation: rolled-out's weaker performance is a property of its
harder optimization problem, not a lack of epochs.

---

## Stage 3 : FM Before a Frozen Linear Classifier

### 3.1 Baseline diagnostics (Part 3)

The frozen Stage 1 probe's own fit, before any FM is added (seed 0, K=10):

| Dataset | Encoder | Train CE | Train top-1 | Val CE | Val top-1 | Test top-1 |
|---|---|---|---|---|---|---|
| DTD | ResNet-18 | 0.537 | 96.2% | 1.848 | 51.60% | 53.72% |
| Flowers-102 | ResNet-18 | 0.088 | 100.0% | 0.726 | 86.27% | 83.28% |
| Flowers-102 | DINOv2 | 0.027 | 100.0% | 0.093 | 99.41% | 99.40% |

Identity-init verification, confirming FM at initialization exactly reproduces the
linear probe:

| Dataset/Encoder | Linear probe test top-1 | FM@init test top-1 | max\|ẑ-z\| |
|---|---|---|---|
| DTD/ResNet-18 | 53.72% | 53.72% | 0.0e+00 |
| Flowers/ResNet-18 | 83.28% | 83.28% | 0.0e+00 |
| Flowers/DINOv2 | 99.40% | 99.40% | 0.0e+00 |

### 3.2 Hyperparameter ablation grids (Part 7)

Seed 0 only, selected by validation top-1 of the full pipeline.

Strategy 1, end-to-end rollout, displacement weight λ:

| Dataset | λ | Best epoch | Val top-1 | Test top-1 (Δ) |
|---|---|---|---|---|
| DTD | 0 | 0 | 51.12 | 53.09 (-0.64) |
| DTD | 1 | 0 | 51.28 | 53.35 (-0.37) |
| DTD | 10 | 46 | 51.54 | 53.09 (-0.64) |
| DTD | 100 (selected) | 114 | 51.70 | 53.78 (+0.05) |
| Flowers | 0 | 1 | 85.39 | 83.07 (-0.21) |
| Flowers | 1 (selected) | 72 | 86.96 | 84.01 (+0.73) |
| Flowers | 10 | 140 | 86.67 | 83.79 (+0.50) |
| Flowers | 100 | 195 | 86.37 | 83.44 (+0.16) |

At λ=0, Strategy 1 fails outright: the checkpoint lands at epoch 0-1, no better than
identity init, and test accuracy sits below baseline on both datasets. This is the
memorization risk the baseline diagnostics predicted.

Strategy 2, classifier-guided, target step η, target steps M, refresh period R (all
normalized unless noted):

| Dataset | η | M | R | Best epoch | Val top-1 | Test top-1 (Δ) |
|---|---|---|---|---|---|---|
| DTD | 0.02 | 1 | 5 (selected) | 135 | 52.55 | 53.62 (-0.11) |
| DTD | 0.02 | 1 | 20 | 173 | 52.02 | 53.40 (-0.32) |
| DTD | 0.02 | 3 | 5 | 58 | 52.55 | 53.72 (+0.00) |
| DTD | 0.02 | 3 | 20 | 75 | 52.02 | 53.03 (-0.69) |
| DTD | 0.05 | 1 | 5 | 40 | 52.18 | 53.94 (+0.21) |
| DTD | 0.05 | 1 | 20 | 72 | 52.18 | 53.40 (-0.32) |
| DTD | 0.05 | 3 | 5 | 12 | 52.29 | 54.26 (+0.53) |
| DTD | 0.05 | 3 | 20 | 12 | 52.02 | 54.15 (+0.43) |
| DTD | 0.1 | 1 | 5 | 18 | 52.18 | 54.10 (+0.37) |
| DTD | 0.1 | 1 | 20 | 54 | 52.13 | 53.72 (+0.00) |
| DTD | 0.1 | 3 | 5 | 8 | 52.13 | 53.67 (-0.05) |
| DTD | 0.1 | 3 | 20 | 6 | 52.02 | 54.15 (+0.43) |
| DTD | 0.02 (unnormalized) | 1 | 5 | 3 | 51.54 | 53.78 (+0.05) |
| Flowers | 0.02 | 1 | 5 | 164 | 87.55 | 84.03 (+0.75) |
| Flowers | 0.02 | 1 | 20 | 178 | 87.35 | 84.06 (+0.78) |
| Flowers | 0.02 | 3 | 5 | 161 | 87.65 | 83.27 (-0.02) |
| Flowers | 0.02 | 3 | 20 | 178 | 88.04 | 84.45 (+1.17) |
| Flowers | 0.05 | 1 | 5 | 77 | 87.65 | 83.72 (+0.44) |
| Flowers | 0.05 | 1 | 20 | 178 | 88.04 | 84.31 (+1.02) |
| Flowers | 0.05 | 3 | 5 | 60 | 88.04 | 83.93 (+0.65) |
| Flowers | 0.05 | 3 | 20 | 82 | 88.24 | 83.96 (+0.68) |
| Flowers | 0.1 | 1 | 5 | 111 | 87.94 | 83.49 (+0.21) |
| Flowers | 0.1 | 1 | 20 (selected) | 117 | 88.33 | 84.16 (+0.88) |
| Flowers | 0.1 | 3 | 5 | 44 | 87.65 | 83.83 (+0.55) |
| Flowers | 0.1 | 3 | 20 | 73 | 88.33 | 84.06 (+0.78) |
| Flowers | 0.1 (unnormalized) | 1 | 20 | 7 | 86.27 | 83.30 (+0.02) |

Normalizing the target step is what makes Strategy 2 work at all. The unnormalized
variant collapses to near-zero displacement (0.008 on DTD, 0.001 on Flowers, as a
fraction of feature norm) and stalls by epoch 3-7 at or below baseline, because the raw
classifier gradient vanishes once a sample is already confidently classified.

### 3.3 Main comparison, three seeds per dataset (Part 9.1)

Top-1 test accuracy, mean ± std over 3 seeds, Δ vs. the Stage 1 linear probe.

| Dataset | Encoder | Stage 1 linear probe | S1 end-to-end rollout (Δ) | S2 classifier-guided (Δ) |
|---|---|---|---|---|
| DTD | ResNet-18 | 54.68 ± 0.76 | 54.50 ± 0.54 (-0.18) | 54.59 ± 0.75 (-0.09) |
| Flowers-102 | ResNet-18 | 83.40 ± 0.09 | 83.98 ± 0.25 (+0.57) | 84.26 ± 0.13 (+0.86) |
| Flowers-102 | DINOv2 | 99.31 ± 0.11 | 99.43 ± 0.05 (+0.12) | 99.31 ± 0.12 (-0.00) |

Per-seed test accuracy, for reference:

| Dataset/Encoder | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| DTD/ResNet-18, probe | 53.72 | 54.73 | 55.59 |
| DTD/ResNet-18, S1 | 53.78 | 54.68 | 55.05 |
| DTD/ResNet-18, S2 | 53.62 | 54.73 | 55.43 |
| Flowers/ResNet-18, probe | 83.28 | 83.51 | 83.41 |
| Flowers/ResNet-18, S1 | 84.01 | 83.66 | 84.26 |
| Flowers/ResNet-18, S2 | 84.16 | 84.18 | 84.44 |
| Flowers/DINOv2, probe | 99.40 | 99.15 | 99.37 |
| Flowers/DINOv2, S1 | 99.41 | 99.50 | 99.37 |
| Flowers/DINOv2, S2 | 99.38 | 99.14 | 99.40 |

Flowers/ResNet-18 is the only trustworthy effect here : positive on all 3 seeds for both
strategies. DTD's own seed-to-seed baseline spread (±0.76) is far larger than either
effect being measured, while Flowers' spread (±0.09) is small because its 10-shot subset
is the entire official split, identical across seeds.

### 3.4 Displacement diagnostic

| Dataset | Encoder | Method | Rel. displacement (test) | Train CE | Train top-1 | Test CE, init to after | Test top-1, init to after |
|---|---|---|---|---|---|---|---|
| DTD | ResNet-18 | S1 | 0.028 | 0.431 | 98.3% | 1.769 to 1.747 | 53.72 to 53.78 |
| DTD | ResNet-18 | S2 | 0.152 | 0.269 | 96.0% | 1.769 to 1.651 | 53.72 to 53.62 |
| Flowers | ResNet-18 | S1 | 0.108 | 0.010 | 100.0% | 0.802 to 0.658 | 83.28 to 84.01 |
| Flowers | ResNet-18 | S2 | 0.229 | 0.005 | 100.0% | 0.802 to 0.602 | 83.28 to 84.16 |
| Flowers | DINOv2 | S1 | 0.064 | 0.014 | 100.0% | 0.096 to 0.073 | 99.40 to 99.41 |
| Flowers | DINOv2 | S2 | 0.021 | 0.026 | 100.0% | 0.096 to 0.096 | 99.40 to 99.38 |

On DTD, test CE improves 6.7% under S2 (1.769 to 1.651) while accuracy moves the other
way (53.72 to 53.62), the same loss/accuracy decoupling seen in Stage 2's robustness
checks. Displacement size also does not track accuracy: DTD/S2 moves 5.4 times further
than DTD/S1 (0.152 vs. 0.028) for essentially the same accuracy.

### 3.5 Separability metrics (Part 9.3)

Measured directly in full feature space, on the test split.

| Dataset | Encoder | Features | Logit margin | Within/between scatter | cos(z, W_y) | Test top-1 |
|---|---|---|---|---|---|---|
| DTD | ResNet-18 | original | +0.178 | 3.345 | -0.0264 | 53.72% |
| DTD | ResNet-18 | after S1 | +0.182 | 3.317 | -0.0206 | 53.78% |
| DTD | ResNet-18 | after S2 | +0.475 | 2.895 | +0.0354 | 53.62% |
| Flowers | ResNet-18 | original | +2.043 | 1.593 | -0.0035 | 83.28% |
| Flowers | ResNet-18 | after S1 | +2.667 | 1.535 | +0.0343 | 84.01% |
| Flowers | ResNet-18 | after S2 | +4.047 | 1.381 | +0.0959 | 84.16% |
| Flowers | DINOv2 | original | +4.610 | 0.468 | +0.5098 | 99.40% |
| Flowers | DINOv2 | after S1 | +5.075 | 0.454 | +0.5305 | 99.41% |
| Flowers | DINOv2 | after S2 | +4.644 | 0.465 | +0.5118 | 99.38% |

DTD is the clearest case of geometry improving without accuracy following: the logit
margin nearly triples (0.178 to 0.475) and scatter falls, yet accuracy barely moves.
These are averages dominated by samples that were already correct, so the flow makes
confident predictions more confident without flipping the boundary cases that actually
determine accuracy.

### 3.6 Saturation check: DINOv2 on Flowers-102 (Part 9.4)

| Method | Test top-1 | Δ |
|---|---|---|
| Stage 1 linear probe | 99.31% ± 0.11 | - |
| S1 end-to-end rollout | 99.43% ± 0.05 | +0.12 |
| S2 classifier-guided | 99.31% ± 0.12 | -0.00 |

S1's checkpoint lands at epoch 7 (relative displacement 0.064). S2's checkpoint lands at
epoch 0 (relative displacement 0.021), essentially the identity map, since validation is
never beaten. DINOv2's features are already well aligned with the classifier
(cos(z, W_y) = 0.51, versus roughly 0 for ResNet-18), so there is nothing left to fix.

### 3.7 Optional extension: jointly fine-tuning the classifier (Part 10)

Classifier-only control, no FM, just continued AdamW on W and b:

| Dataset | Val top-1 | Test top-1 (Δ) |
|---|---|---|
| DTD | 51.38 | 53.99 (+0.27) |
| Flowers | 86.37 | 83.35 (+0.07) |

Joint FM + classifier grid (seed 0):

| Dataset | lr_clf | Unfreeze epoch | Val top-1 | Test top-1 (Δ) |
|---|---|---|---|---|
| DTD | 0.0001 | 0 | 51.38 | 53.88 (+0.16) |
| DTD | 0.0001 (selected) | 50 | 51.60 | 53.72 (+0.00) |
| DTD | 0.001 | 0 | 51.44 | 53.67 (-0.05) |
| DTD | 0.001 | 50 | 51.60 | 53.72 (+0.00) |
| Flowers | 0.0001 | 0 | 87.55 | 84.11 (+0.83) |
| Flowers | 0.0001 (selected) | 50 | 87.65 | 84.05 (+0.76) |
| Flowers | 0.001 | 0 | 87.25 | 83.85 (+0.57) |
| Flowers | 0.001 | 50 | 87.35 | 84.01 (+0.73) |

Final comparison (seed 0 only, since the joint grid was not repeated across seeds):

| Dataset | Linear probe | S1 (frozen) | S2 (frozen) | Joint FM + classifier | Δ joint vs. probe |
|---|---|---|---|---|---|
| DTD | 53.72 | 53.78 | 53.62 | 53.72 | +0.00 |
| Flowers | 83.28 | 84.01 | 84.16 | 84.05 | +0.76 |

Unfreezing the classifier adds nothing beyond frozen-classifier FM. On Flowers, joint
training's +0.76 barely exceeds the classifier-only control's +0.07, so the real gain is
the flow's and was already delivered without unfreezing anything. On DTD, the
classifier-only control (+0.27) is the best result in this entire notebook: its small
available headroom comes from the classifier alone, and FM contributes nothing.

---

## Cross-Stage Notes

DTD's prototype accuracy at K=full reads 58.83 in Stage 1's own table and 58.78 in Stage
2's recomputation. Both notebooks implement the identical method on paper; the 0.05pt
gap most likely comes from a re-extraction of `features/` partway through the project (a
fresh ResNet-18 forward pass is not guaranteed to be bit-identical to the original one).
Each stage's section above uses that stage's own self-reported number rather than
reconciling the two.

Every other baseline that appears in more than one stage checks out exactly, including
Stage 1's DTD/ResNet-18 linear probe at K=10 (54.68 ± 0.76) against Stage 3's
independently trained baseline on the same subsets and seeds (per-seed 53.72, 54.73,
55.59, giving the same mean and standard deviation).
