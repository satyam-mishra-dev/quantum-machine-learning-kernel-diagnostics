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
> than plain classical ML without telling you."

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
> right: QSVM 0.955, VQC 0.956, logistic regression 0.961. A tool that only
> ever says 'quantum wins' is marketing; ours is a validated predictor.
> One more thing: the quantum models use just 4 qubits — 4 PCA components —
> and still match the full 13-feature classical floor at 0.955. Four numbers
> into a quantum circuit matched thirteen into classical ML."

### Stop 5 — Patient check

> "Per-patient inference is instant. Each result carries a conformal
> prediction badge — 'confident' or 'flag for human review' — with a
> statistical coverage guarantee from a held-out calibration split, plus a
> SHAP explanation of which features drove the call. That's the clinical
> safety layer: the model knows when it doesn't know."

### Stop 6 — Upload

> "It's not a fixed demo — drop any biomedical CSV. Watch the pipeline:
> qubit budgeting, quantum and classical twins trained side by side, the
> scout's verdict, and an honest verdict card at the end. The platform
> tells you whether quantum was worth it for *your* data."

## 3. Hard judge questions

**Q: Does quantum actually beat classical here?**
A: On the three classic tabular datasets, no — and the platform says so
(WDBC: QSVM 0.995 vs classical 0.996; Cleveland: 0.955/0.956 vs 0.961;
PIMA: VQC 0.743 vs 0.732). It wins where theory says it should: the
quantum-native set (QSVM 0.972 vs best classical 0.743) and chest X-rays
(hybrid QNN 0.939 vs matched CNN 0.925, sensitivity 94.6% vs 90.8%).
Measuring that boundary per dataset, honestly, is the product.

**Q: Why quantum ML at all if classical matches on real data?**
A: Two measured reasons. First, sample efficiency in the right regime: on
the quantum-native set QSVM reaches 0.826 AUROC from just 30 samples while
classical sits at 0.379–0.549 — medical data is often small. Second,
compression: on Cleveland, 4 qubits matched the 13-feature classical floor
(0.955 vs 0.955). And on images we already have a real measured win.

**Q: Is the Advantage Scout real science or a made-up score?**
A: It's the geometric-difference test from Huang et al. (Nature Comms 2021).
We compute both kernel directions, cite Huang for his form, and drive the
verdict from our own empirically validated asymmetric ratio. It survived an
adversarial audit, and it's 4-for-4: three 'classical will match' verdicts
(g-ratio 0.296, 0.450, 0.309) all came out parity; the one 'advantage
possible' verdict (2.51) came out +23 AUROC points for quantum.

**Q: What about hardware noise? Simulators are easy.**
A: We ran it. VQC inference on ibm_marrakesh with error mitigation
(dynamical decoupling, measurement twirling, TREX readout), 1024 shots,
job id `da8724bsq5js73bko9eg`. Noise attenuates scores toward zero
(e.g. -0.888 sim → -0.796 hardware) but 4/4 predictions agreed with the
simulator. Receipts, not projections.

**Q: Do you retrain per patient? Is inference practical?**
A: No retraining. Models train once per dataset (QSVM in ~15–98 s on the
tabular sets, hybrid QNN in 7.9 s); per-patient scoring is a single forward
pass and returns instantly, with the conformal calibration already baked in
from a held-out split. Retraining only happens when a new dataset is
uploaded.

**Q: What's novel vs IBM's QBioCode?**
A: QBioCode benchmarks QSVC/VQC against classical models and profiles
datasets — but it has no advantage scout (the geometric-difference test is
not productized anywhere we could find, only the paper and a tutorial), no
conformal trust layer, no clinician-facing report, and no error-mitigated
hardware receipts. We have all four, each backed by a number in `runs/`.
