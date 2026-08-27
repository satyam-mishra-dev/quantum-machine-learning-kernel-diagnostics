# Ideas & Epoch Log

Running log of build epochs and PPT-worthy findings. Newest first.

## Epoch 6 — research round 2 pays off: PQK (2026-08-28, ~01:30)

**Projected quantum kernel (PQK) — the night's biggest model win.** Research
agent surfaced Huang et al.'s prescription for exactly our regime; an
implementation agent shipped it; verification: best single model on
Cleveland (0.965 AUROC, best spec@95sens) and ties best classical on WDBC
(0.996) — at ~1/100th-1/200th of QSVM's training cost (N circuit evals vs
N^2). Now also in the upload path, where it WON on live-uploaded PIMA
(best_model: pqk, +0.007 over best classical). PPT: "our second-generation
quantum model is both better and 100x cheaper — and it came from reading
the same paper our Scout is built on."

**Adapter validated on 3 unseen datasets via the live API** (ILPD Indian
liver, heart failure, Parkinson's voice): scout 3/3 consistent
("classical_will_match" every time, no false advantage calls), worst-case
upload-to-verdict 2 minutes. Scout lifetime: 7/7.

**Ensemble (q+c soft-vote): honest verdict "marginal, harmless"** — best
AUROC on WDBC by 0.0004, mid-pack on Cleveland. Kept as a served model,
never pitched as advantage.

**Gold citation found:** arXiv:2604.18837 — 970-experiment QSVM-vs-classical
benchmark, 0/29 significant quantum wins, KTA competitive only at ~2000x
compute. Independently validates our measured KTA failure AND our
honest-floor framing. Lead with it when a judge pushes back.

**Product polish from validation feedback:** upload jobs now report live
pipeline stage (ingest → budget → scout → training:<model> → verdict), UI
driven by real stages; NaN CSVs verified handled; row cap raised to 600.

**In flight:** PIMA robust retune (winner: budget 4, C 1, reps 2) +
re-benchmark including pqk/ensemble; select-vs-PCA evaluation on PIMA next.

## Epoch 5 — negative results are results (2026-08-28, ~00:00)

**Trainable-kernel experiment (delegated, verified):** kernel-target
alignment training worked as optimization (KTA +28% pima / +19% cleveland on
the train subsample) but produced NO held-out gain — pima -0.019, cleveland
-0.009, both within noise, both negative. Verdict: rejected from the
pipeline; base QSVM stands. PPT line: "we implemented the literature's
accuracy lever (Hubregtsen et al. kernel alignment), measured it honestly on
held-out data, and rejected it — that's what an honest platform does."

**Error-mitigated receipt (delegated, verified):** ibm_marrakesh, job
`da8724bsq5js73bko9eg` (publicly verifiable), DD+twirling+TREX, 4/4
sim-hardware agreement, and mitigation visibly tightened score fidelity
(worst sim-hw gap 0.12 vs 0.44 unmitigated). Demo comparison chart:
unmitigated vs mitigated receipts.

**Docs shipped:** DEMO.md (3-min click-path runbook + 6 hard-judge Q&As) and
README results table — all numbers traced to runs/*.json by an agent that
cross-checked and corrected two of my own epoch-log numbers.

**Serving-path verification:** borderline Cleveland patient (VQC 0.53) →
conformal set {0,1} → "flag for human review" + uncalibrated-probability
caution. The safety layer demonstrably fires — use this exact patient in the
live demo.

**BreastMNIST (done, delegated+verified):** the hybrid edge REVERSES —
classical 0.646±0.021 vs hybrid 0.556±0.089 over 3 seeds; the quantum head
is seed-unstable at 546 training images and its train-chosen threshold
collapses (sens 0.095). Honest negative, committed. Demo line: "quantum won
on pneumonia X-rays, lost on breast ultrasound — and the platform tells you
which is which. That's the product."

**In flight:** 9-slide problem-first pitch deck (sih2026/Q-Nidaan-pitch.pdf)
grounded in runs/*.json, quantum confined to one slide + the receipt.

## Epoch 4 — adversarial review + final leak-free numbers (2026-08-27, ~23:30)

**Review (subagent, 9 findings, all fixed):** (1) FATAL: our "geometric
difference" inverted the opposite kernel from Huang et al. — now we compute
both directions, cite Huang only for his (g_huang_naive_reg), and drive the
verdict from our own empirically-validated g_asym, labeled honestly. (2)
sample-efficiency curves leaked preprocessing (PCA fit on full train before
subsetting) — now the whole pipeline refits per subset; "n=30" really means
30 rows. (3) upload endpoint had OOM/DoS holes (unclamped qubit budget, no
file cap, O(n^2) kernel on unbounded rows) — capped 10 qubits/5MB/400 rows,
caps reported in results. Plus: true VQC epochs, conformal small-n guard,
deployment-floor field, uncalibrated-probability flag, ddof=1 over 5 seeds.

**Final verified numbers (post-fix, 5-seed leak-free curves):**
- Scout 4/4 holds. wdbc/cleveland/pima "classical_will_match"
  (g_ratio 0.30/0.45/0.31) → measured parity; quantum_synth
  "advantage_possible" (2.51) → QSVM 0.972 vs 0.743 best classical.
- Quantum-native curves now dominate at EVERY size: n=30 QSVM 0.826 vs
  svm_rbf 0.549. This is the headline chart, now methodologically clean.
- Cleveland: 4-qubit quantum (VQC 0.956, QSVM 0.955) beats the full-feature
  classical floor (0.955/0.950 on 13 features). PPT line: "4 numbers into a
  quantum circuit matched 13 into classical ML."
- PIMA: floor logreg_full 0.823 >> compressed models (~0.72-0.74) — the
  never-ship-worse guarantee earning its keep, honestly reported.
- PneumoniaMNIST: hybrid QNN 0.939 AUROC / 94.6% sensitivity beats matched
  CNN twin 0.925 / 90.8% — a real measured quantum win on images.

**PPT framing insight:** every number above SURVIVED an adversarial audit —
"our results are post-review, leak-free, and reproducible from runs/*.json"
is itself a differentiator no hackathon team will have.

**In flight:** trainable-kernel (KTA-optimized) accuracy experiment on
pima/cleveland; error-mitigated QPU receipt (DD+twirling+TREX) on the
retrained VQC.

## Epoch 3 — the loop closes: Scout goes 4/4 (2026-08-27, ~22:30)

**THE demo story, fully measured:** after tuning, the Advantage Scout is
4-for-4. Three real datasets: "classical_will_match" (g_ratio 0.30-0.46) →
measured parity (Cleveland QSVM 0.955 vs logreg 0.959; WDBC QSVM 0.996 vs
0.997). One quantum-native dataset: "advantage_possible" (g_ratio 2.11) →
measured QSVM 0.976 vs best classical 0.743 (+23 AUROC points). The platform
routes both regimes correctly. Auto-tuning also closed Cleveland's
quantum-classical gap from 7 points to parity, and found reps=2 on the
synthetic set on its own (CV 0.945 vs 0.655) — it rediscovered the labeling
kernel's structure.

**Design flaw caught & fixed:** tuner picked budget=4 for QSVM which also
compressed the classical twins → everyone got worse on PIMA. Split roles:
fair-comparison twins (compressed) + full-feature classical floor
("*_full" in heldout) — "the platform never ships worse than plain
classical ML" is now measured, not asserted.

**Competitive research (agent, cited):** geometric-difference scout is NOT
productized anywhere (only Huang et al. paper + a PennyLane tutorial) — our
novelty claim verified. Closest competitor: IBM QBioCode (QSVC/VQC vs
classical + profiling) but no scout, no conformal, no clinician report, no
hardware receipts — perfect differentiation slide. Literature anchors for
PPT: WDBC hybrid QSVM-QNN 99% (PeerJ CS 2026), heart QSVM 90.26%
(MDPI AI 2026), QML breast-cancer review avg AUC 0.91.

**Hardware upgrade:** rewrote runner as native Qiskit EstimatorV2 with
dynamical decoupling + measurement twirling + TREX readout mitigation
(resilience_level=1); PennyLane→Qiskit translation verified to 5e-16 on
simulator before any QPU spend. "Error-mitigated receipts" line for PPT.

## Epoch 2 — the Scout validation story (2026-08-27, late night)

**Headline finding (PPT centerpiece candidate):** across all three flagship
datasets, the geometric-difference test (Huang et al. 2021) computed
g/sqrt(n) of 0.30-0.46 — deep in "classical will match" territory — and the
measured benchmarks agreed 3/3 (best classical beat/matched QSVM everywhere).
The Advantage Scout is a *validated* predictor, not a hope. Pitch line:
"Our platform predicted, before training, that quantum would not beat
classical on these datasets — and it was right. When it says advantage is
possible, you can believe it."

**Hardware receipt:** ibm_fez (156 qubits), 4 patients (2 disease/2 healthy),
4/4 correct, 4/4 sim-hardware agreement. Noise shrinks scores toward zero
(-0.86 -> -0.44) but never flipped a prediction. Chart this in the PPT.

**In progress:** quantum-native synthetic dataset (labels generated by the
quantum feature map, Huang et al. relabeling) — the regime where quantum
provably wins. If QSVM crushes classical there AND the Scout flips to
"advantage_possible", the platform demonstrably routes both regimes
correctly. That closes the scientific loop live in the demo.

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
