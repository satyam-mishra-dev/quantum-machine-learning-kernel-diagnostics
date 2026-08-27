# Q-Nidaan — 3-Minute Demo Runbook

For a team member who did not build this. Every number below is copied from
`runs/*.json` (post-review, leak-free). Do not improvise numbers.

## 1. Setup and start

One-time setup (skip if `.venv/` and `web/node_modules/` exist):

```
cp .env.example .env               # IBM Quantum credentials (only needed for new QPU runs)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd web && npm install
```

Start (two terminals, from repo root):

```
.venv/bin/uvicorn qnidaan.app:app --port 8000
cd web && npm run dev
```

Open http://localhost:5173.

## 2. Click-path, in order

### Stop 1 — Landing page
Scroll to the **hardware receipt** section.

> "This ran on a real IBM quantum computer — ibm_marrakesh, 156 qubits.
> Job id `da8724bsq5js73bko9eg`, verifiable on IBM's dashboard. Four patients
> scored on hardware with error mitigation (dynamical decoupling, measurement
> twirling, TREX readout): 4/4 agreement between simulator and hardware.
> Noise shrinks scores toward zero — one patient went from -0.888 in
> simulation to -0.796 on hardware — but it never flipped a prediction."

### Stop 2 — Dashboard (5 dataset cards)

> "Five datasets, one honest benchmark each. Two quantum wins: on chest
> X-rays (PneumoniaMNIST) the hybrid quantum network gets 0.939 AUROC vs
> 0.925 for its matched classical CNN twin, and on the quantum-native set
> QSVM gets 0.972 vs 0.743 for the best classical model. On the three
> classic tabular datasets — breast cancer, heart disease, diabetes —
> it's honest parity, and we say so. The caption under each card is the
> full-feature classical floor: the platform never ships a model worse
> than plain classical ML without telling you. One more model on every
> card: PQK, our projected quantum kernel — best single model on
> Cleveland at 0.965 and tied with the best classical on WDBC at 0.996,
> at roughly 1/100th of QSVM's training cost."

### Stop 3 — quantum_synth detail (the headline chart)
Open the sample-efficiency chart.

> "This is the headline. QSVM beats both classical baselines at every
> training size: at n=30 it's 0.826 vs 0.549 for the RBF SVM and 0.379 for
> logistic regression; at n=200 it's 0.960 vs 0.678. And here's the part
> that matters: our Advantage Scout predicted this *before any training*.
> It computed a geometric-difference ratio of 2.51 — above threshold — and
> said 'advantage possible'. Then the measurements confirmed it."

### Stop 4 — cleveland detail (the honesty proof)

> "Same scout, opposite verdict. On Cleveland heart disease it said
> 'classical will match' — g-ratio 0.450, well below threshold — and it was
> right: QSVM 0.955, VQC 0.956, logistic regression 0.961. And the quiet
> star: PQK, the projected quantum kernel, is the best single model on this
> dataset at 0.965 — trained in 0.17 seconds versus QSVM's 15.6. A tool that
> only ever says 'quantum wins' is marketing; ours is a validated predictor.
> One more thing: the quantum models use just 4 qubits — 4 PCA components —
> and still match the full 13-feature classical floor at 0.955. Four numbers
> into a quantum circuit matched thirteen into classical ML."

### Stop 5 — Patient check

> "Per-patient inference is instant. Each result carries a conformal
> prediction badge — 'confident' or 'flag for human review' — with a
> statistical coverage guarantee from a held-out calibration split, plus a
> SHAP explanation of which features drove the call. That's the clinical
> safety layer: the model knows when it doesn't know."

### Stop 6 — Upload (use the Pima CSV)

> "It's not a fixed demo — drop any biomedical CSV. Watch the live stages:
> ingest, qubit budgeting, scout, quantum and classical twins trained side
> by side, honest verdict card. And watch the budgeting step: the platform
> picked feature-selection here, not PCA — it CV-tests both per dataset
> (0.814 vs 0.777 on Pima) and shows the runner-up score. That choice
> closed Pima's compression gap: the compressed models jumped to 0.79–0.81
> against a 0.823 full-feature floor. Now open the patient explanation —
> it names glucose, age, insulin, and BMI, actual clinical features a
> doctor can act on, instead of 'principal component 1'."

## 3. Hard judge questions

**Q: Does quantum actually beat classical here?**
A: On the three classic tabular datasets it's parity — and the platform
says so (WDBC: PQK 0.996 vs classical 0.996; Cleveland: PQK 0.965 vs
logreg 0.961; PIMA: VQC 0.802 vs SVM-RBF 0.806, full-feature floor 0.823
still ships). It wins where theory says it should: the quantum-native set
(QSVM 0.972 vs best classical 0.743) and chest X-rays (hybrid QNN 0.939
vs matched CNN 0.925, sensitivity 94.6% vs 90.8%). A 970-experiment
independent benchmark (arXiv:2604.18837) found 0/29 statistically
significant QSVM wins on classical data — which is exactly why measuring
that boundary per dataset, honestly, is the product.

**Q: Why quantum ML at all if classical matches on real data?**
A: Two measured reasons. First, sample efficiency in the right regime: on
the quantum-native set QSVM reaches 0.826 AUROC from just 30 samples while
classical sits at 0.379–0.549 — medical data is often small. Second,
compression: on Cleveland, 4 qubits matched the 13-feature classical floor
(0.955 vs 0.955). And on images we already have a real measured win.

**Q: Is the Advantage Scout real science or a made-up score?**
A: It's the geometric-difference test from Huang et al. (Nature Comms 2021),
with our own empirically validated asymmetric ratio, and its record is
deliberately stated asymmetrically. "Classical will match" is 6/6 against
measured outcomes — WDBC (g 0.30), Cleveland (0.45), plus ILPD, heart
failure, and Parkinson's through the live adapter — it has never wrongly
said "don't bother". "Advantage possible" is a necessary condition, not a
promise: on the quantum-native set (g 2.51) it preceded a +23-AUROC-point
quantum win, but on Pima under the deeper reps-2 kernel quantum merely
tied. When it says possible, the harness decides. We also implemented
kernel-target-alignment training, measured it on held-out data, and
rejected it (negative gain on both datasets) — no knob was tuned to make
the record look cleaner.

**Q: Why is your quantum model 100x faster than typical QSVMs?**
A: PQK — the projected quantum kernel, from the same Huang et al. paper the
Scout is built on. A fidelity QSVM needs a circuit per *pair* of patients
(N² evaluations); PQK measures each patient's circuit once (N evaluations)
and feeds the projected features to a classical kernel. Measured: 0.17 s vs
15.6 s to train on Cleveland, 0.47 s vs 133.7 s on Pima — and it's the best
single model on Cleveland (0.965) while tying best classical on WDBC
(0.996).

**Q: What about hardware noise? Simulators are easy.**
A: We ran it. VQC inference on ibm_marrakesh with error mitigation
(dynamical decoupling, measurement twirling, TREX readout), 1024 shots,
job id `da8724bsq5js73bko9eg`. Noise attenuates scores toward zero
(e.g. -0.888 sim → -0.796 hardware) but 4/4 predictions agreed with the
simulator. Receipts, not projections.

**Q: Do you retrain per patient? Is inference practical?**
A: No retraining. Models train once per dataset (QSVM in ~16–134 s on the
tabular sets, PQK in under half a second, hybrid QNN in 7.6 s); per-patient
scoring is a single forward pass and returns instantly, with the conformal
calibration already baked in from a held-out split. Retraining only happens
when a new dataset is uploaded — and the live adapter is validated
end-to-end on three unseen datasets (ILPD Indian liver, heart failure,
Parkinson's): worst case 2 minutes from upload to verdict.

**Q: What's novel vs IBM's QBioCode?**
A: QBioCode benchmarks QSVC/VQC against classical models and profiles
datasets — but it has no advantage scout (the geometric-difference test is
not productized anywhere we could find, only the paper and a tutorial), no
conformal trust layer, no clinician-facing report, and no error-mitigated
hardware receipts. We have all four, each backed by a number in `runs/`.
