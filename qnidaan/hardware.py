"""Run flagship VQC inference on real IBM Quantum hardware.

Train on simulator, infer on the QPU — the receipt for "this runs on a real
quantum computer". VQC inference is one circuit per patient, so a 3-patient
demo costs a few QPU seconds against the free Open Plan's 10 min/28 days.

Usage: python -m qnidaan.hardware cleveland --patients 3
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main(dataset: str, n_patients: int, shots: int):
    import os

    import pennylane as qml
    from dotenv import load_dotenv
    from qiskit_ibm_runtime import QiskitRuntimeService

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    from qnidaan.datasets import load_split

    bundle = joblib.load(MODELS_DIR / f"{dataset}.joblib")
    vqc = bundle["models"]["vqc"]
    plan = bundle["plan"]

    _, Xte, _, yte, _ = load_split(dataset)
    # stratified pick so the receipt shows both classes
    pos = np.flatnonzero(yte == 1)[: max(1, n_patients // 2)]
    neg = np.flatnonzero(yte == 0)[: n_patients - len(pos)]
    idx = np.concatenate([pos, neg])
    Z = plan.transform(Xte)[idx]
    truth = yte[idx]

    sim_scores = vqc._scores(vqc.weights_, vqc.bias_, Z)

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=os.environ["QISKIT_IBM_TOKEN"],
        instance=os.environ["QISKIT_IBM_INSTANCE"],
    )
    backend = service.least_busy(operational=True, simulator=False)
    print(f"backend: {backend.name} ({backend.num_qubits} qubits)",
          flush=True)

    dev = qml.device("qiskit.remote", wires=vqc.n_qubits, backend=backend,
                     shots=shots)

    @qml.qnode(dev)
    def hw_circuit(weights, x):
        qml.AngleEmbedding(x, wires=range(vqc.n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(vqc.n_qubits))
        return qml.expval(qml.PauliZ(0))

    t0 = time.time()
    hw_scores = np.array(
        [float(hw_circuit(vqc.weights_, z)) for z in Z]
    ) + float(vqc.bias_)
    wall = time.time() - t0

    result = {
        "dataset": dataset,
        "backend": backend.name,
        "backend_qubits": backend.num_qubits,
        "n_qubits_used": vqc.n_qubits,
        "shots": shots,
        "wall_seconds": round(wall, 1),
        "patients": [
            {
                "true_label": int(truth[i]),
                "sim_score": round(float(sim_scores[i]), 4),
                "hw_score": round(float(hw_scores[i]), 4),
                "sim_prediction": int(sim_scores[i] >= 0),
                "hw_prediction": int(hw_scores[i] >= 0),
                "agree": bool((sim_scores[i] >= 0) == (hw_scores[i] >= 0)),
            }
            for i in range(len(Z))
        ],
    }
    result["sim_hw_agreement"] = float(
        np.mean([p["agree"] for p in result["patients"]])
    )
    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"hardware_{dataset}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="cleveland")
    ap.add_argument("--patients", type=int, default=3)
    ap.add_argument("--shots", type=int, default=1024)
    a = ap.parse_args()
    main(a.dataset, a.patients, a.shots)
