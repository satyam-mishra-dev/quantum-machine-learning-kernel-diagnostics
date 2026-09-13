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

import __main__ as _main

from qnidaan.harness import MODELS_DIR, RUNS_DIR, SoftVote

_main.SoftVote = SoftVote  # bundles pickled by `python -m qnidaan.harness`
                           # reference __main__.SoftVote; alias it so
                           # joblib.load resolves outside the harness
from qnidaan.readiness import readiness
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
        if r.get("dataset", {}).get("task") == "regression":
            continue  # r2/mae metrics; not a classification dashboard row
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


@app.get("/api/readiness/{name}")
def readiness_for(name: str):
    path = RUNS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, "no such run")
    r = json.loads(path.read_text())
    if not r.get("scout"):  # image-track runs have no scout
        raise HTTPException(404, "no readiness score for image-track runs")
    d = r["dataset"]
    n = (d.get("n_train") or 0) + (d.get("n_test") or 0)
    return readiness(n, d.get("n_features", 0),
                     float(d.get("prevalence", 0.5)),
                     r["scout"], r.get("budget", {}))


@app.get("/api/circuit/{name}")
def circuit(name: str, model: str = "qsvm"):
    """The real circuit behind a served model: ascii drawing, device-level
    depth/gate counts, and the basis-state distribution from one simulator
    execution on a real background sample."""
    import pennylane as qml

    from qnidaan.quantum import _device, _feature_map

    b = bundle(name)
    if model not in ("qsvm", "pqk", "vqc") or model not in b["models"]:
        raise HTTPException(422, f"no quantum circuit for model {model!r}")
    m = b["models"][model]
    n_qubits = m.n_qubits
    x = np.asarray(b["Z_background"][0], dtype=float)
    dev = _device(n_qubits)
    wires = list(range(n_qubits))

    if model == "vqc":
        weights = np.asarray(m.weights_, dtype=float)

        @qml.qnode(dev)
        def qc(x):
            qml.AngleEmbedding(x, wires=wires, rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=wires)
            return qml.probs(wires=wires)
    else:  # qsvm / pqk share the feature map; show it as-is
        reps = m.reps

        @qml.qnode(dev)
        def qc(x):
            _feature_map(x, wires, reps)
            return qml.probs(wires=wires)

    probs = np.asarray(qc(x))
    top = np.argsort(probs)[::-1][:8]
    res = qml.specs(qc, level="device")(x)["resources"]
    return {
        "model": model,
        "n_qubits": n_qubits,
        "ascii": qml.draw(qc, level="device")(x),
        "depth": int(res.depth),
        "gates": {k: int(v) for k, v in res.gate_types.items()},
        "measurement": {"basis_state_probs": {
            format(int(i), f"0{n_qubits}b"): round(float(probs[i]), 4)
            for i in top
        }},
    }


class PredictIn(BaseModel):
    features: dict[str, float]
    model: str = "qsvm"


@app.get("/api/schema/{name}")
def feature_schema(name: str):
    b = bundle(name)
    return {"feature_names": b["meta"]["feature_names"]}


def _component_labels(plan, names):
    """Human-readable labels: name PCA components by their top loadings."""
    if plan.method == "select":
        return [names[i] for i in plan.selected_features]
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


from concurrent.futures import ProcessPoolExecutor
import asyncio
from fastapi.responses import StreamingResponse
from qnidaan.db import create_job, get_job, get_job_logs

pool = ProcessPoolExecutor(max_workers=1)

from typing import Optional

@app.post("/api/upload")
def upload(target: str = Form(""),
           budget: int = Form(8),
           benchmark: Optional[str] = Form(None),
           file: Optional[UploadFile] = File(None)):
    from qnidaan.datasets import load_csv, load_image_benchmark
    from qnidaan.worker import run_adapter_worker

    budget = max(1, min(budget, 10))  # a 30-qubit statevector is an OOM
    
    if benchmark in ["pneumoniamnist", "breastmnist"]:
        try:
            X, y, meta = load_image_benchmark(benchmark)
        except Exception as e:
            raise HTTPException(422, f"could not load benchmark: {e}")
    else:
        if not file:
            raise HTTPException(422, "must provide file or valid benchmark")
            
        raw = file.file.read(5_000_001)
        if len(raw) > 5_000_000:
            raise HTTPException(413, "File too large (5 MB cap for the demo)")
            
        filename = file.filename or ""
        try:
            if filename.endswith('.zip'):
                from qnidaan.datasets import load_zip
                X, y, meta = load_zip(io.BytesIO(raw), budget)
            else:
                X, y, meta = load_csv(io.BytesIO(raw), target)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception:
            raise HTTPException(422, "could not parse file")
    
    job_id = uuid.uuid4().hex[:8]
    create_job(job_id, meta)

    # Dispatch to ProcessPoolExecutor
    future = pool.submit(run_adapter_worker, job_id, X, y, meta, budget)
    
    def on_done(f):
        from qnidaan.db import update_job
        try:
            f.result()
        except Exception as e:
            update_job(job_id, status="FAILED", stage="FAILED", error=str(e), log_message=f"Worker crashed: {str(e)}")
            
    future.add_done_callback(on_done)
    
    return {"job_id": job_id, "n_samples": len(y),
            "n_features": meta["n_features"], "profile": meta.get("profile")}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = get_job(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    return j


@app.get("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    """Server-Sent Events for job status updates."""
    j = get_job(job_id)
    if not j:
        raise HTTPException(404, "no such job")

    async def event_generator():
        last_log_id = 0
        while True:
            current_job = get_job(job_id)
            if not current_job:
                break
                
            logs = get_job_logs(job_id)
            for log in logs[last_log_id:]:
                yield f"data: {json.dumps(log)}\n\n"
            last_log_id = len(logs)
            
            if current_job["status"] in ["COMPLETED", "FAILED", "error", "done"]:
                # Yield final state just in case
                yield f"data: {json.dumps({'stage': current_job['stage'], 'status': current_job['status'], 'message': 'Job terminated'})}\n\n"
                break
                
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

from fastapi.responses import FileResponse

@app.get("/api/jobs/{job_id}/report")
def get_job_report(job_id: str):
    j = get_job(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    if j["status"] != "COMPLETED":
        raise HTTPException(409, "job not completed")
    run_dir = Path(__file__).resolve().parent.parent / "runs" / f"QF-{job_id}"
    report_path = run_dir / "report.pdf"
    if not report_path.exists():
        raise HTTPException(404, "Report not found")
    return FileResponse(report_path, filename=f"report_{job_id}.pdf", media_type="application/pdf")

@app.get("/api/jobs/{job_id}/artifacts")
def get_job_artifacts(job_id: str):
    j = get_job(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    if j["status"] != "COMPLETED":
        raise HTTPException(409, "job not completed")
    zip_path = Path(__file__).resolve().parent.parent / "runs" / f"QuantumFloor_{job_id}.zip"
    if not zip_path.exists():
        raise HTTPException(404, "Artifacts zip not found")
    return FileResponse(zip_path, filename=f"QuantumFloor_{job_id}.zip", media_type="application/zip")
