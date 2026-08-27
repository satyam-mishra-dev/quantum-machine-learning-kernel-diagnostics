# Q-Nidaan

Hybrid quantum-classical ML platform for early disease detection (SIH 2026).

Upload a biomedical dataset → auto-compress to a qubit budget → train a hybrid
quantum model beside a matched classical twin → get an honest benchmark report
with explanations and per-patient uncertainty. Inference showcased on real IBM
quantum hardware.

## Results (measured, post-review)

Held-out AUROC per dataset. "Quantum" and "classical" are matched twins on
the same 4-qubit-budget features; "floor" is the best full-feature classical
model (the never-ship-worse baseline).

| Dataset | Best quantum | Best classical | Full-feature floor |
|---|---|---|---|
| WDBC breast cancer | 0.995 (QSVM) | 0.996 (SVM-RBF) | 0.996 (logreg) |
| Cleveland heart disease | 0.956 (VQC) | 0.961 (logreg) | 0.955 (SVM-RBF) |
| Pima diabetes | 0.802 (VQC) | 0.806 (SVM-RBF) | 0.823 (logreg) |
| Quantum-native synthetic | 0.972 (QSVM) | 0.743 (random forest) | 0.766 (random forest) |
| PneumoniaMNIST X-rays | 0.939 (hybrid QNN) | 0.925 (matched CNN) | — (image track) |

Advantage Scout: the "classical_will_match" verdict is 6/6 against
measured outcomes (WDBC/Cleveland g-ratio 0.30/0.45, plus ILPD, heart
failure, Parkinson's via the live adapter) — it has never wrongly said
"don't bother". "advantage_possible" is a necessary-condition signal,
not a promise: quantum_synth (g-ratio 2.51) → QSVM 0.972 vs 0.743;
Pima under the deeper reps-2 kernel (g-ratio 1.68) → quantum merely
tied (VQC 0.743 vs logreg 0.732). The harness always decides.

Hardware receipt: VQC inference on ibm_marrakesh (156 qubits, 4 used,
1024 shots) with dynamical decoupling, measurement twirling, and TREX
readout mitigation — job `da8724bsq5js73bko9eg`, 4/4 simulator-hardware
agreement.

All numbers regenerate from `python -m qnidaan.harness` and land in
`runs/*.json`.

## Setup

```
cp .env.example .env               # fill in IBM Quantum credentials
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd web && npm install
```

## Run

```
.venv/bin/uvicorn qnidaan.app:app --port 8000    # API
cd web && npm run dev                            # frontend on :5173
```

## Pipeline commands

```
.venv/bin/python -m qnidaan.harness              # benchmark all datasets
.venv/bin/python -m qnidaan.tune                 # CV-tune QSVM configs
.venv/bin/python -m qnidaan.hardware cleveland   # VQC inference on real QPU
```

Benchmark results land in `runs/`, trained bundles in `models/` (both
regenerable; `data/` and `.env` are gitignored and stay local).
