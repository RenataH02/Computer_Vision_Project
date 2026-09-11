# Computer Vision Project : From Frozen Features to Flow Matching

A three stage study of few shot image classification on frozen pretrained encoders, progressing from classical baselines in Stage 1, to a learned flow matching transformation applied at two different points in the pipeline in Stages 2 and 3.

Full results for all three stages are in [`RESULTS.md`](RESULTS.md).

*Team* : Renata Haiek, Saly Jammal

## Overview
All three stages share the same experimental backbone : features are extracted once from frozen pretrained encoders and cached, so every subsequent experiment is a cheap CPU side analysis on top of the same fixed representations. This cleanly separates the expensive, one time encoder stage from the cheap, reproducible classification/analysis stage that each notebook performs.

|Property|Description|
|---|---|
|Datasets|DTD (47 texture classes) and Oxford Flowers-102 (102 flower classes)|
|Frozen encoders|ImageNet-1K ResNet-18 (512-d, both datasets) and DINOv2 ViT-S/14 (384-d, Flowers-102 only)|
|Training set sizes|K ∈ {5, 10, full} labeled images per class, 3 seeds each|
|Metric|Top 1 accuracy on the official test split, mean ± std over 3 runs|

DTD and Flowers-102 were deliberately chosen to contrast two domains : DTD (textures) is out of domain for an object centric ImageNet encoder, while Flowers-102 (natural objects) suits it well, making every encoder/method comparison in the three stages more informative.

## Stage 1 : Classification Baselines
The goal is to build a reliable, reproducible classification pipeline on top of frozen encoders, using two methods : the required linear probe (softmax classifier trained with AdamW) and one prototype based branch, chosen to be image derived class prototypes, the normalized mean of each class's normalized training features, classified by cosine similarity. This choice was made because it reuses the exact same cached features as the linear probe and yields a directly comparable accuracy vs K curve.

Results : top 1 test accuracy, mean ± std over 3 seeds :
|Dataset|Encoder|Method|K=5|K=10|K=full|
|---|---|---|---|---|---|
|DTD|ResNet-18|linear probe|45.78 ± 1.21|54.68 ± 0.76|63.01 ± 0.07|
|DTD|ResNet-18|prototypes|46.51 ± 0.53|53.37 ± 0.85|58.83 ± 0.00|
|Flowers-102|ResNet-18|linear probe|75.68 ± 0.68|83.40 ± 0.09|83.40 ± 0.09|
|Flowers-102|ResNet-18|prototypes|70.19 ± 0.11|75.23 ± 0.00|75.23 ± 0.00|
|Flowers-102|DINOv2|linear probe|99.05 ± 0.22|99.31 ± 0.11|99.31 ± 0.11|
|Flowers-102|DINOv2|prototypes|99.04 ± 0.13|99.40 ± 0.00|99.40 ± 0.00|

*note : Flowers-102's official training split has exactly 10 image/class, so its 10 shot and full columns are identical by construction. Only DTD shows a genuine 5 to 10 to full progression.*

Key insights :
- The encoder matters far more than the classifier. Swapping ResNet-18 for DINOv2 on Flowers-102 is worth ~16 accuracy points, more than any classifier choice or amount of labeled data tested. DINOv2 with only 5 labeled images/class (99.05%) already beats ResNet-18 trained on the entire training split (83.40%).
- Prototypes are a strong low shot baseline but stop improving. At 5 shots they match or beat the linear probe (DTD: 46.51% vs 45.78%), but the linear probe pulls ahead with more data (DTD-full: 63.01% vs 58.83%) since it can learn discriminative directions rather than just re-estimating a class mean.
- Overfitting tracks feature quality. The DTD/ResNet-18 probe reaches 96.2% train accuracy but only 51.6% validation accuracy. Checkpoint selection by validation accuracy is essential there, while it's nearly irrelevant for the well separated DINOv2/Flowers features with a ~0.6-point gap.
- Residual errors are semantic, not random. On DTD the largest confusions are between near synonymous textures (dotted ↔ polka dotted, lined → banded). On Flowers-102 they're between visually similar flowers (canterbury bells ↔ monkshood), a human annotator could plausibly disagree on many of these.
- Carried forward : the prototype baseline feeds into Stage 2, and the linear probe setting feeds into Stage 3, so both later comparisons are like for like against this stage.

---
## Stage 2 : Flow Matching to Class Prototypes
The goal is to insert a learned flow matching transformation between the (L2 normalized) frozen feature and the Stage 1 prototype classifier, and test whether it can improve on the prototype baseline. Two training regimes are compared : standard FM (regress directly onto the ideal straight line velocity) and rolled out FM (train through the actual T step Euler inference procedure), each evaluated at T=4 and T=12 inference steps.
Results : top 1 test accuracy at K=full vs. the recomputed stage 1 prototype baseline :
|Dataset/Encoder|Baseline|Standard FM (T=12)|Rolled out FM (T=12)|
|---|---|---|---|
|DTD/ResNet-18|58.78|59.91(+1.13)|55.48(-3.30)|
|Flowers-102/ResNet-18|75.22|77.29(+2.07)|74.16(-1.05)|
|Flowers-102/DINOv2|99.40|99.48(+0.08)|99.47(+0.08)|

Key insights :
- FM's effect is small and depends on how much data is available. At K=5/10, standard FM often underperforms the prototype baseline (DTD K=10: 53.4%→49.6%, a 3.8pt drop). With too little data the network can't learn a transport that beats doing nothing. At K=full this flips to a small, consistent gain (DTD +1.1pt, Flowers/ResNet-18 +2.1pt).
- Standard FM beats rolled out FM in every configuration tested, often by several points, even though both train stably with smoothly decreasing loss. Two follow up checks : training rolled out for far longer, and matching its epoch budget to standard FM's, both ruled out undertraining as the cause : the gap is a genuinely harder optimization landscape (backpropagating through T sequential steps with supervision only on the final point rather than every point along the path).
- On DINOv2/Flowers, FM changes essentially nothing (±0.1pt), accuracy is already near ceiling (99%+), so there's nothing left for FM to fix.
- The learned transformation is a small, local nudge, not a re-embedding. Feature space visualizations look nearly identical before and after FM, and individual flow trajectories are short hops that rarely reach the target prototype. This is consistent with the modest, inconsistent accuracy deltas observed.

---
## Stage 3 : Flow Matching Before a Frozen Linear Classifier
The goal is to insert the same kind of learned FM transformation, but this time ahead of the frozen stage 1 linear probe rather than the prototype rule, using one encoder per dataset : ResNet-18, since it is the only option for DTD, with K=10 and T=4. FM is initialized at the identity, the pipeline exactly reproduces the stage 1 probe before any training, so every accuracy change is attributable to stage 3 training alone.
Two strategies are compared :
- Strategy 1, end to end rollout : backprop classification loss through the full T step rollout, with a displacement penalty to prevent the ~0.5M parameter network from simply memorizing the small training set.
- Strategy 2, classifier guided targets : the frozen classifier's gradient builds an explicit step length normalized target in feature space, and FM is trained with ordinary regression onto that target.

Results : top 1 test accuracy, mean over 3 seeds, K=10, T=4 :
|Dataset/Encoder|Stage 1 linear probe|Strategy 1 end to end|Strategy 2 classifier guided|
|---|---|---|---|
|DTD/ResNet-18|54.68|54.50(-0.18)|54.59(-0.09)|
|Flowers-102/ResNet-18|83.40|83.98(+0.58)|84.26(+0.86)|
|Flowers-102/DINOv2|99.31|99.43 (+0.12)|99.31 (−0.00)|

Key insights :
- FM produces a small, real gain only where the frozen representation is genuinely misaligned with the classifier and there's enough data to measure it. Flowers-102/ResNet-18 is the one trustworthy effect, positive across all 3 seeds for both strategies, in a paired design sensitive enough to resolve a sub point difference. DTD's own seed to seed spread (±0.76) is 4–8× larger than the effect being measured, and DINOv2 is already well aligned with its classifier, so neither shows a reliable change.
- Classifier guided targets (Strategy 2) beat end to end rollout on every count : more accurate where anything happens (+0.86 vs +0.58), more stable across seeds, and far less sensitive to hyperparameters (11 of 12 tested configurations beat baseline on Flowers, vs. Strategy 1 being actively harmful with no displacement penalty at all).
- The flow performs a small, class conditional rotation, not a re-embedding. Displacements are only 2–23% of feature norm and 2D visualizations look nearly unchanged, yet quantitative separability measures shift meaningfully : logit margins grow 2–2.7×, and features rotate measurably toward their own class's classifier weight direction, improving the geometry the classifier measures, without always flipping the decision on borderline examples. Can see this most clearly on DTD, where margin and confidence improve substantially but accuracy still doesn't move.
- Jointly fine tuning the frozen classifier alongside FM adds nothing beyond frozen classifier FM, confirming the gain measured on Flowers-102 is genuinely attributable to the learned flow, not to extra classifier capacity.

---
## Overall conclusions across the three stages
1. The choice of frozen encoder dominates every other design decision tested, far more than classifier choice, amount of labeled data, or the flow matching layer added in Stages 2 and 3.
2. Flow matching provides a real but small benefit, and only under specific conditions : enough training data to estimate a correction, and a base representation that is measurably misaligned with the downstream classifier it feeds into. Where either condition fails (DTD's overfit scarce data regime, DINOv2's already near ceiling representation), FM makes no reliable difference.
3. Simpler, more constrained training objectives outperform more direct but less constrained ones : in both Stage 2 with standard FM over rolled out FM, and Stage 3 with classifier guided targets over unconstrained end to end rollout, the version of FM training that couldn't "cheat" toward memorization generalized better.
4. The learned flow is consistently a small, local, class conditional adjustment : never a large geometric reorganization of the feature space, across every stage and visualization used to probe it.

## Reproducibility
All reported numbers are deterministic given the fixed seeds used throughout, balanced subset sampling, classifier/FM initialization, batch shuffling, and dimensionality reduction plots. So rerunning the entirety of any of the three notebooks reproduces every number in this README exactly.
