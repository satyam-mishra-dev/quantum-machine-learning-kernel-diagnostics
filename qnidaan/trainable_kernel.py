"""Trainable quantum kernel: per-feature scaling optimized for KTA.

Kernel alignment training (Hubregtsen et al. 2021): learn weights w so the
feature map phi(w * x) maximizes kernel-target alignment on the TRAINING
split. Test data never enters the optimization. QSVM then uses the aligned
kernel — this is the honest accuracy lever for hard tabular sets like PIMA.

Usage: python -m qnidaan.trainable_kernel pima
"""
import json
import sys
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

SEED = 42
KTA_SUBSAMPLE = 80  # ponytail: KTA cost is O(n^2) circuits per step
STEPS = 30


def make_trainable_kernel(n_qubits, reps=1):
    dev = qml.device("lightning.qubit", wires=n_qubits)
    wires = list(range(n_qubits))

    def feature_map(x, w):
        z = w * x
        for _ in range(reps):
            qml.AngleEmbedding(z, wires=wires, rotation="Y")
            for i in range(n_qubits - 1):
                qml.IsingZZ(z[i] * z[i + 1], wires=[i, i + 1])

    @qml.qnode(dev, interface="autograd")
    def kernel_circuit(x1, x2, w):
        feature_map(x1, w)
        qml.adjoint(feature_map)(x2, w)
        return qml.probs(wires=wires)

    def kernel(x1, x2, w):
        return kernel_circuit(x1, x2, w)[0]

    return kernel


def train_alignment(Z_train, y_train, n_qubits, reps=1, steps=STEPS,
                    seed=SEED):
    """Return optimized per-feature weights (starts at all-ones)."""
    rng = np.random.default_rng(seed)
    if len(Z_train) > KTA_SUBSAMPLE:
        idx = rng.permutation(len(Z_train))[:KTA_SUBSAMPLE]
        Z_train, y_train = Z_train[idx], np.asarray(y_train)[idx]
    y_pm = pnp.array(2 * np.asarray(y_train) - 1, requires_grad=False)
    Z = pnp.array(Z_train, requires_grad=False)
    kernel = make_trainable_kernel(n_qubits, reps)
    w = pnp.array(np.ones(n_qubits), requires_grad=True)
    opt = qml.AdamOptimizer(0.2)

    def neg_kta(w):
        K = qml.kernels.square_kernel_matrix(
            Z, lambda a, b: kernel(a, b, w), assume_normalized_kernel=True)
        Y = pnp.outer(y_pm, y_pm)
        return -pnp.sum(K * Y) / (
            pnp.sqrt(pnp.sum(K * K)) * pnp.sqrt(pnp.sum(Y * Y)))

    for step in range(steps):
        w, cost = opt.step_and_cost(neg_kta, w)
        if step % 5 == 0:
            print(f"  step {step}: KTA={-cost:.4f}", flush=True)
    return np.asarray(w)


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score

    from qnidaan.budgeter import fit_budget
    from qnidaan.datasets import load_split
    from qnidaan.harness import _tuned_config
    from qnidaan.quantum import QSVM

    name = sys.argv[1] if len(sys.argv) > 1 else "pima"
    tuned = _tuned_config(name) or {"budget": 8, "C": 10, "reps": 1}
    Xtr, Xte, ytr, yte, _ = load_split(name)
    plan = fit_budget(Xtr, budget=tuned["budget"])
    Ztr, Zte = plan.transform(Xtr), plan.transform(Xte)

    t0 = time.time()
    w = train_alignment(Ztr, ytr, plan.n_qubits, reps=tuned["reps"])
    print(f"aligned weights: {np.round(w, 3)} ({time.time()-t0:.0f}s)")

    base = QSVM(plan.n_qubits, reps=tuned["reps"], C=tuned["C"]).fit(Ztr, ytr)
    auc_base = roc_auc_score(yte, base.predict_proba(Zte)[:, 1])

    aligned = QSVM(plan.n_qubits, reps=tuned["reps"], C=tuned["C"])
    aligned.fit(Ztr * w, ytr)  # scaling folds into the inputs
    auc_aligned = roc_auc_score(yte, aligned.predict_proba(Zte * w)[:, 1])

    out = {"dataset": name, "weights": [round(float(x), 4) for x in w],
           "auroc_base": round(float(auc_base), 4),
           "auroc_aligned": round(float(auc_aligned), 4),
           "gain": round(float(auc_aligned - auc_base), 4)}
    print(json.dumps(out, indent=2))
    from qnidaan.harness import RUNS_DIR
    p = RUNS_DIR / "kernel_alignment.json"
    hist = json.loads(p.read_text()) if p.exists() else {}
    hist[name] = out
    p.write_text(json.dumps(hist, indent=2))
