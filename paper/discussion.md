## Discussion

The LHFM result on LifeSnaps establishes a measurable effect for the
self-supervised representation on at least one task. On high-stress
prediction, LHFM exceeded both logistic regression and random forest
on identical engineered features by approximately 20 AUROC points
(0.567 versus 0.328 and 0.368). Because all three models received the
same features, the gap is attributable to the representation rather
than to feature engineering. The 95 % confidence interval on LHFM
itself remains wide and includes chance, which we treat as a
statement about the test fold (11 participants) rather than about the
method: the comparison to the baselines is statistically and
substantively the meaningful contrast at this cohort size. The
GLOBEM replication described in §Results, GLOBEM is the natural test
of whether this gap survives at a tenfold larger cohort with full
modality coverage.

We report one negative result alongside the positive one. On the
sleep-disruption task on the same cohort, random forest outperformed
LHFM (0.641 versus 0.518), and we surface this in the headline table
rather than tuning it away. The most parsimonious interpretation is
that sleep duration, sleep efficiency, and a small number of derived
sleep-stage ratios are nearly sufficient statistics for this task,
and a tree-based model can fit the relevant decision boundary faster
than a self-supervised transformer can learn it from a 49-participant
masked-feature-reconstruction pretraining set. The self-supervised
prior is not universally helpful: it requires either a task where
the relevant signal is distributed across modalities (as high-stress
appears to be in LifeSnaps), or a cohort large enough for the
pretraining objective to converge on a useful representation. The
LifeSnaps result documents one cohort where one of these two
conditions does not hold for one of the two evaluable tasks. We
expect this to remain visible in the GLOBEM analysis and treat it
as informative about where the method does and does not buy lift.

A second observation worth surfacing concerns the methodological
infrastructure rather than the numbers. The pipeline behaved
transparently when modalities or labels were absent. LifeSnaps has no
smartphone passive sensing and no climate enrichment; the adapter
emitted modality-missing flags, the encoder consumed those flags as
first-class input, and the downstream heads produced calibrated
predictions on the modalities that were present. On the
low-mood task, where the cohort's SEMA reports did not produce
sufficient positive examples for evaluation, the pipeline surfaced
this as zero positive rate and we marked the task unevaluable rather
than lowering the binarisation threshold to manufacture a positive
class. We mention this because the small methodological choices that
make a result trustworthy — participant-level splits, participant-
clustered bootstrap, identical features for baselines, the same
trainer for all tasks, public commit of the adapter before evaluation,
no row-level predictions committed to the public repository in
keeping with the LifeSnaps data-use agreement — are visible in the
git history of the public repository and reproducible by any reader
in approximately 15 minutes of CPU time. Reproducibility at this
level of granularity is the artefact we most want to make standard
for foundation-model work in this space; the result on LifeSnaps is a
demonstration that the artefact survives contact with a real cohort.

What LHFM enables for other groups is a scaffold for cohort-specific
fine-tuning that does not require rebuilding the evaluation,
fairness-audit, climate-regime-holdout, or multimodal adapter
pipelines from scratch. A research group with their own wearable +
smartphone cohort can implement a single subclass of the adapter
base, point the LHFM CLI at their data directory, and obtain
participant-clustered held-out test metrics, subgroup-stratified
AUROC, classical baselines on the same features, and an
integrated-gradients dashboard within an afternoon's setup. We
hope, in particular, that the next generation of multi-cohort
foundation-model papers in this space adopt participant-clustered
bootstrap as the default uncertainty estimator. A small change
relative to row-level resampling, but one that substantially affects
whether published confidence intervals say what their authors
typically intend them to say.

We close by reiterating what LHFM is not. It is not a medical device.
It is not a stress monitor, a mood tracker, or a sleep scorer for
clinical use. The four downstream tasks are coarse proxies derived
from EMA scales; they are not validated clinical instruments. The
test-set AUROCs reported here are within an order of magnitude of
the literature on this kind of task, which is to say that they are
research-prototype-credible rather than clinically-deployable. We
make no claim about deployment-readiness, and the model card,
ethics statement, and acceptable-use note distributed with this work
should be read by anyone considering an extension of LHFM to a
real-data setting beyond the published cohorts.
