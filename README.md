# Q-Nidaan

Hybrid quantum-classical ML platform for early disease detection (SIH 2026).

Upload a biomedical dataset → auto-compress to a qubit budget → train a hybrid
quantum model beside a matched classical twin → get an honest benchmark report
with explanations and per-patient uncertainty. Inference showcased on real IBM
quantum hardware.

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
