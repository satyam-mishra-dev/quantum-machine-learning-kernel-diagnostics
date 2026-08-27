"""Q-Nidaan API.

Flagship models are pre-trained by the harness (models/*.joblib) and served
for instant per-patient inference. Uploaded CSVs run through the universal
adapter as background jobs.
"""
import io
import json
import threading
import uuid
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qnidaan.harness import MODELS_DIR, RUNS_DIR
from qnidaan.report import patient_report

app = FastAPI(title="Q-Nidaan")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_bundles = {}
_jobs = {}  # ponytail: in-memory jobs, fine for a demo box


def bundle(name):
    if name not in _bundles:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise HTTPException(404, f"no trained bundle for {name!r}; "
                                     "run the harness first")
        _bundles[name] = joblib.load(path)
    return _bundles[name]


@app.get("/api/datasets")
def datasets():
    out = []
    for f in sorted(RUNS_DIR.glob("*.json")):
        r = json.loads(f.read_text())
        if "heldout" not in r:  # tuned.json, hardware_*.json live here too
            continue
        servable = [m for m in r["heldout"] if not m.endswith("_full")]
        best = max(servable, key=lambda m: r["heldout"][m]["auroc"])
        floors = [m for m in r["heldout"] if m.endswith("_full")]
        floor = (max(floors, key=lambda m: r["heldout"][m]["auroc"])
                 if floors else None)
        out.append({
            "name": r["dataset"]["name"],
            "title": r["dataset"]["title"],
            "n_qubits": r["budget"]["n_qubits"],
            "best_model": best,
            "best_auroc": r["heldout"][best]["auroc"],
            "deployment_floor": floor,
            "deployment_floor_auroc": (r["heldout"][floor]["auroc"]
                                       if floor else None),
            "qsvm_auroc": r["heldout"].get("qsvm", {}).get("auroc"),
            "quantum_worth_trying": r.get("scout", {}).get(
                "quantum_worth_trying"),
            "scout_verdict": r.get("scout", {}).get("verdict"),
            "image_track": r.get("image_track", False),
        })
    return out


@app.get("/api/hardware")
def hardware():
    out = []
    for f in sorted(RUNS_DIR.glob("hardware_*.json")):
        out.append(json.loads(f.read_text()))
    return out


@app.get("/api/runs/{name}")
def run_detail(name: str):
    path = RUNS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, "no such run")
    return json.loads(path.read_text())


class PredictIn(BaseModel):
    features: dict[str, float]
    model: str = "qsvm"


@app.get("/api/schema/{name}")
def feature_schema(name: str):
    b = bundle(name)
    return {"feature_names": b["meta"]["feature_names"]}


def _component_labels(plan, names):
    """Human-readable labels: name PCA components by their top loadings."""
    if plan.method != "pca":
        return names
    pca = plan.pipeline.named_steps["pca"]
    labels = []
    for i, comp in enumerate(pca.components_):
        top = np.argsort(np.abs(comp))[::-1][:2]
        labels.append(f"PC{i+1} ({', '.join(names[j] for j in top)})")
    return labels


@app.post("/api/predict/{name}")
def predict(name: str, body: PredictIn):
    b = bundle(name)
    names = b["meta"]["feature_names"]
    missing = [f for f in names if f not in body.features]
    if missing:
        raise HTTPException(422, f"missing features: {missing}")
    x = np.array([[body.features[f] for f in names]])
    z = b["plan"].transform(x)[0]
    if body.model not in b["models"]:
        raise HTTPException(422, f"unknown model {body.model!r}")
    rep = patient_report(
        model=b["models"][body.model],
        surrogate=b["models"]["svm_rbf"],
        qhat=b["qhats"][body.model],
        z=z,
        Z_background=b["Z_background"],
        feature_names=_component_labels(b["plan"], names),
    )
    rep["model"] = body.model
    rep["dataset"] = name
    return rep


@app.post("/api/upload")
def upload(file: UploadFile = File(...), target: str = Form(...),
           budget: int = Form(8)):
    from qnidaan.datasets import load_csv

    budget = max(1, min(budget, 10))  # a 30-qubit statevector is an OOM
    raw = file.file.read(5_000_001)
    if len(raw) > 5_000_000:
        raise HTTPException(413, "CSV too large (5 MB cap for the demo)")
    try:
        X, y, meta = load_csv(io.BytesIO(raw), target)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception:
        raise HTTPException(422, "could not parse file as CSV")
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "running", "stage": "ingest", "dataset": meta}

    def work():
        try:
            _jobs[job_id]["result"] = _run_adapter(X, y, meta, budget,
                                                  _jobs[job_id])
            _jobs[job_id]["status"] = "done"
        except Exception as e:  # surface, don't crash the server
            _jobs[job_id].update(status="error", error=str(e))

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id, "n_samples": len(y),
            "n_features": meta["n_features"]}


def _run_adapter(X, y, meta, budget, job):
    """Universal adapter: the same pipeline the flagships went through."""
    from sklearn.model_selection import train_test_split

    from qnidaan.budgeter import fit_budget
    from qnidaan.classical import evaluate, make_twins
    from qnidaan.quantum import QSVM, make_kernel
    from qnidaan.scout import scout

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
    capped = len(Xtr) > 400  # quantum kernel is O(n^2) simulated circuits
    if capped:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(Xtr))[:400]
        Xtr, ytr = Xtr[idx], ytr[idx]
    job["stage"] = "qubit_budget"
    plan = fit_budget(Xtr, budget=budget)
    Ztr, Zte = plan.transform(Xtr), plan.transform(Xte)
    kernel = make_kernel(plan.n_qubits)
    job["stage"] = "advantage_scout"
    out = {
        "budget": {"n_qubits": plan.n_qubits, "method": plan.method,
                   "explained_variance": plan.explained_variance},
        "scout": scout(Ztr, ytr, kernel),
        "heldout": {},
        "train_rows_used": int(len(ytr)),
        "train_rows_capped": capped,
    }
    models = {**make_twins(), "qsvm": QSVM(n_qubits=plan.n_qubits, C=10)}
    for mname, m in models.items():
        job["stage"] = f"training:{mname}"
        m.fit(Ztr, ytr)
        out["heldout"][mname] = evaluate(m, Zte, yte)
    job["stage"] = "verdict"
    best = max(out["heldout"], key=lambda m: out["heldout"][m]["auroc"])
    out["verdict"] = {
        "best_model": best,
        "quantum_won": best == "qsvm",
        "qsvm_vs_best_classical": round(
            out["heldout"]["qsvm"]["auroc"]
            - max(v["auroc"] for k, v in out["heldout"].items()
                  if k != "qsvm"), 4),
    }
    return out


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "no such job")
    return _jobs[job_id]
