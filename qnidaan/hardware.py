"""Run flagship VQC inference on real IBM Quantum hardware, error-mitigated.

Train on simulator, infer on the QPU. The trained PennyLane VQC is rebuilt
as a native Qiskit circuit (verified equivalent on simulator before any QPU
time is spent) and executed via EstimatorV2 with dynamical decoupling,
measurement twirling and TREX readout mitigation (resilience_level=1).

VQC inference is one circuit per patient: a 4-patient receipt costs a few
QPU seconds against the free Open Plan's 10 min / 28 days.

Usage: python -m qnidaan.hardware cleveland --patients 4
"""
import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def vqc_to_qiskit(vqc, x):
    """Rebuild AngleEmbedding(Y) + StronglyEntanglingLayers as Qiskit.

    PennyLane Rot(phi, theta, omega) applies RZ(phi), RY(theta), RZ(omega)
    in circuit order; SEL layer l entangles CNOT[i, (i + r_l) % n] with
    r_l = (l % (n-1)) + 1.
    """
    from qiskit import QuantumCircuit

    n = vqc.n_qubits
    w = np.asarray(vqc.weights_)  # (layers, n, 3)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(float(x[i]), i)
    for layer in range(w.shape[0]):
        for i in range(n):
            phi, theta, omega = (float(v) for v in w[layer, i])
            qc.rz(phi, i)
            qc.ry(theta, i)
            qc.rz(omega, i)
        r = (layer % (n - 1)) + 1 if n > 1 else 0
        for i in range(n):
            qc.cx(i, (i + r) % n)
    return qc


def _verify_translation(vqc, Z):
    """Statevector cross-check: Qiskit rebuild must match PennyLane."""
    from qiskit.quantum_info import SparsePauliOp, Statevector

    obs = SparsePauliOp("I" * (vqc.n_qubits - 1) + "Z")  # Z on qubit 0
    pl = vqc._scores(vqc.weights_, vqc.bias_, Z) - float(vqc.bias_)
    qk = np.array([
        Statevector(vqc_to_qiskit(vqc, z)).expectation_value(obs).real
        for z in Z
    ])
    err = float(np.max(np.abs(pl - qk)))
    assert err < 1e-6, f"translation mismatch: {err}"
    return err


def main(dataset: str, n_patients: int, shots: int):
    from dotenv import load_dotenv
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import (
        generate_preset_pass_manager,
    )
    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

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

    err = _verify_translation(vqc, Z)
    print(f"qiskit translation verified (max err {err:.2e})", flush=True)
    sim_scores = vqc._scores(vqc.weights_, vqc.bias_, Z)

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=os.environ["QISKIT_IBM_TOKEN"],
        instance=os.environ["QISKIT_IBM_INSTANCE"],
    )
    backend = service.least_busy(operational=True, simulator=False)
    print(f"backend: {backend.name} ({backend.num_qubits} qubits)",
          flush=True)

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    obs = SparsePauliOp("I" * (vqc.n_qubits - 1) + "Z")
    pubs = []
    for z in Z:
        isa_qc = pm.run(vqc_to_qiskit(vqc, z))
        pubs.append((isa_qc, obs.apply_layout(isa_qc.layout)))

    estimator = EstimatorV2(mode=backend)
    estimator.options.default_shots = shots
    estimator.options.resilience_level = 1  # TREX readout mitigation
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XpXm"
    estimator.options.twirling.enable_measure = True

    t0 = time.time()
    job = estimator.run(pubs)
    res = job.result()
    wall = time.time() - t0
    hw_scores = np.array([float(r.data.evs) for r in res]) + float(vqc.bias_)

    result = {
        "dataset": dataset,
        "backend": backend.name,
        "backend_qubits": backend.num_qubits,
        "n_qubits_used": vqc.n_qubits,
        "shots": shots,
        "error_mitigation": ["dynamical_decoupling(XpXm)",
                             "measurement_twirling", "TREX_readout"],
        "job_id": job.job_id(),
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
    ap.add_argument("--patients", type=int, default=4)
    ap.add_argument("--shots", type=int, default=1024)
    a = ap.parse_args()
    main(a.dataset, a.patients, a.shots)
