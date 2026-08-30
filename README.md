# QuantumFloor

Hybrid quantum-classical ML platform for early disease detection (SIH 2026).
Product name: **QuantumFloor**; the Python package and repo keep the working
name `qnidaan` (Q-Nidaan).

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
| WDBC breast cancer | 0.996 (PQK) | 0.996 (SVM-RBF) | 0.996 (logreg) |
| Cleveland heart disease | 0.965 (PQK) | 0.961 (logreg) | 0.955 (SVM-RBF) |
| Pima diabetes | 0.802 (VQC) | 0.806 (SVM-RBF) | 0.823 (logreg) |
| Quantum-native synthetic | 0.972 (QSVM) | 0.743 (random forest) | 0.766 (random forest) |
| PneumoniaMNIST X-rays | 0.939 (hybrid QNN) | 0.925 (matched CNN) | — (image track) |
| CuMiDa renal genomics (54,675 genes → 8 qubits) | 0.919 (PQK) | 0.924 (logreg, 8 genes) | 0.936 (random forest, all genes) |
| Diabetes progression (regression, R²) | 0.464 (VQR) | 0.472 (random forest) | 0.454 (ridge, full) |

Advantage Scout: the "classical_will_match" verdict is 6/6 (every dataset in current runs/) against
measured outcomes (WDBC/Cleveland g-ratio 0.30/0.45, plus ILPD, heart
failure, Parkinson's via the live adapter, and CuMiDa genomics) — it has never wrongly said
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

## Delivery Table (Expected Deliverables)

| # | PS deliverable | What we ship |
|---|---|---|
| D1 | Data handling pipelines | Tabular loaders + universal CSV adapter (`qnidaan/datasets.py`) and a medical-image track (`qnidaan/images.py`). Modalities: tabular CSV and 28x28 grayscale images today; ECG/PTB-XL signal adapter on the roadmap. Upload caps: 5MB / 600 rows / 10 qubits, reported back in results. |
| D2 | Hybrid quantum-classical models | QSVM, PQK, VQC, hybrid QNN (`qnidaan/quantum.py`, `images.py`) — each benchmarked against matched classical twins at the same qubit budget plus a full-feature classical floor (`classical.py`). Quantum regression (VQR, PQK-ridge) in `regression.py`, results landing tonight. |
| D3 | Feature engineering | Qubit Budgeter with auto-choice (`budgeter.py`): fits both PCA and mutual-info selection, CV-probes each, picks per dataset and records the runner-up score (Pima: select 0.814 vs pca 0.777 — selection lifted compressed models ~0.07 AUROC, `runs/compress_exp.json`). |
| D4 | Training & inference workflows | `python -m qnidaan.harness` (benchmark), `qnidaan.tune` (CV tuning), `POST /api/upload` AutoPilot with live stages (ingest → budget → scout → training:model → verdict) polled at `/api/jobs/{id}`, per-patient `POST /api/predict/{name}`. |
| D5 | Performance evaluation | Per-model AUROC, accuracy, spec@95%sensitivity, train_seconds, conformal qhat in every `runs/*.json`; 5-seed leak-free sample-efficiency curves; calibration caution flags. |
| D6 | Explainability | Per-patient SHAP top features in original clinical names, split conformal prediction sets with "flag for human review" on borderline cases, clinician-report fields (`qnidaan/report.py`). |
| D7 | Functional platform | React frontend (`web/`: Dashboard, DatasetDetail, PatientCheck, Upload), FastAPI backend (`qnidaan/app.py`), Docker + Render deployment (`render.yaml`). |
| D8 | Biomedical datasets (public/open) | WDBC, Cleveland, Pima, quantum-native synthetic, PneumoniaMNIST, BreastMNIST benchmarked in `runs/`; ILPD, heart failure, Parkinson's validated live via the adapter; CuMiDa renal genomics (54,675 genes) benchmarked in `runs/cumida.json`; diabetes-progression regression in `runs/regression_diabetes.json`. |
| D9 | Near-term hardware compatibility | Real-QPU inference receipt: ibm_marrakesh, job `da8724bsq5js73bko9eg`, 4 qubits / 1024 shots, DD + twirling + TREX mitigation, 4/4 sim-hardware agreement (`runs/hardware_cleveland.json`). |
| D10 | Comprehensive documentation | `README.md`, `COMPLIANCE.md` (line-by-line PS audit), `DEMO.md` (3-min runbook + judge Q&A), `IDEAS.md` (epoch log, negative results included). |

## What we deliberately do not build

- Arbitrary DICOM/EHR ingestion — CSV + supported image benchmarks only in MVP.
- Diagnostic claims — outputs are decision support with conformal uncertainty,
  never a diagnosis.
- Federated training — single-tenant MVP; federation is roadmap.
- Circuits beyond 12 qubits — near-term hardware honesty over paper qubit counts.
- Blanket quantum-advantage claims — the Scout and the harness decide per
  dataset, and negative results ship (see BreastMNIST).

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
