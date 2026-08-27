# Ideas & Epoch Log

Running log of build epochs and PPT-worthy findings. Newest first.

## Epoch 1 — core pipeline night run (2026-08-27)

**Built:** datasets (WDBC/Cleveland/PIMA + universal CSV adapter), Qubit
Budgeter, classical twins, QSVM + VQC, Advantage Scout (KTA + geometric
difference), benchmark harness with reserved test split + conformal
calibration split, trust layer (split conformal + SHAP surrogate), FastAPI,
hardware runner (VQC on real IBM QPU).

**Findings:**
- Account has access to three 156-qubit Heron machines (ibm_fez,
  ibm_marrakesh, ibm_kingston) — PPT says "156-qubit", not 127.
- QSVM feature-map depth matters a lot: reps=2 over-entangles (moons 0.78),
  reps=1 wins (0.92). PPT point: "we tune entanglement depth per dataset."
- Cleveland heldout: svm_rbf 0.96 AUROC vs QSVM 0.88 / VQC 0.92 (untuned).
  The honesty story is real on day one — quantum doesn't auto-win.

**WDBC results:** QSVM 0.989 / VQC 0.995 vs best classical 0.997 AUROC —
honest parity at full data. Scout: quantum KTA 0.141 vs classical 0.063
(quantum kernel aligns 2.2x better with labels — good PPT stat). BUT the
sample-efficiency curve inverts the hoped claim: at n=30 logreg hits 0.993
while untuned QSVM sits at 0.899. WDBC is near-linearly-separable, so
classical saturates instantly — no small-n room. Lesson for the PPT: the
small-data claim must be *measured per dataset by the platform*, which is
exactly the product's point. Check PIMA (noisy, overlapping classes) for a
regime where the curve story lands.

**Next-epoch targets:**
- [ ] Per-dataset CV tuning of QSVM (C, reps) and qubit budget — close the
      Cleveland gap or report it honestly.
- [ ] Sample-efficiency curves: check whether QSVM's small-n claim shows up;
      if yes, that chart is the PPT centerpiece.
- [ ] Hardware run receipt (few QPU seconds, VQC, 3 patients).
- [ ] Frontend polish pass with GSAP once agent delivers.

**PPT ideas:**
- Show the Advantage Scout verdict per dataset in the demo — "the platform
  itself tells you when quantum is worth trying" is the differentiator.
- Conformal "flag for human review" badge = the clinical-safety wow.
