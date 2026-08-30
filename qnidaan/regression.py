"""Quantum regression: VQR (variational) and PQK-ridge vs classical twins.

Same pipeline discipline as harness.py: qubit-budgeted features, one honest
held-out evaluation, results JSON in runs/. Dataset: sklearn diabetes
(442 samples, 10 features, continuous disease-progression score).
"""
import json
import time
from pathlib import Path

import numpy as np
import pennylane as qml
from sklearn.metrics import mean_absolute_error, r2_score

from qnidaan.budgeter import fit_budget
from qnidaan.quantum import SEED, _device, _feature_map

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


class VQR:
    """Variational quantum regressor: AngleEmbedding + StronglyEntanglingLayers,
    scale * <Z0> + bias on standardized targets, MSE + Adam."""

    def __init__(self, n_qubits, n_layers=3, epochs=25, lr=0.1,
                 batch_size=32, seed=SEED):
        self.n_qubits, self.n_layers = n_qubits, n_layers
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.rng = np.random.default_rng(seed)
        self._build_circuit()

    def _build_circuit(self):
        dev = _device(self.n_qubits)
        n_qubits = self.n_qubits

        @qml.qnode(dev, interface="autograd")
        def circuit(weights, x):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def __getstate__(self):  # qnode closures don't pickle; rebuild on load
        state = self.__dict__.copy()
        state.pop("circuit")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_circuit()

    def fit(self, X, y):
        from pennylane import numpy as pnp

        X, y = np.asarray(X), np.asarray(y, dtype=float)
        self.y_mean_, self.y_std_ = y.mean(), y.std()
        yn = (y - self.y_mean_) / self.y_std_
        shape = qml.StronglyEntanglingLayers.shape(self.n_layers,
                                                   self.n_qubits)
        weights = pnp.array(0.1 * self.rng.normal(size=shape),
                            requires_grad=True)
        scale = pnp.array(1.0, requires_grad=True)
        bias = pnp.array(0.0, requires_grad=True)
        opt = qml.AdamOptimizer(self.lr)

        def cost(w, s, b, Xb, yb):
            preds = s * pnp.stack([self.circuit(w, x) for x in Xb]) + b
            return pnp.mean((preds - yb) ** 2)

        n = len(X)
        for _ in range(self.epochs):  # true epochs: every batch, every pass
            order = self.rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                weights, scale, bias = opt.step(
                    lambda w, s, b: cost(w, s, b, X[idx], yn[idx]),
                    weights, scale, bias)
        self.weights_, self.scale_, self.bias_ = weights, scale, bias
        return self

    def predict(self, X):
        z = np.array([self.circuit(self.weights_, x) for x in np.asarray(X)])
        return (self.scale_ * z + self.bias_) * self.y_std_ + self.y_mean_


class PQKRidge:
    """Projected quantum kernel regression: 1-local <X>,<Y>,<Z> per wire
    (3n features), then KernelRidge (RBF) with quick CV over alpha/gamma."""

    def __init__(self, n_qubits, reps=1):
        self.n_qubits, self.reps = n_qubits, reps
        self._build_circuit()

    def _build_circuit(self):
        dev = _device(self.n_qubits)
        wires = list(range(self.n_qubits))
        reps = self.reps

        @qml.qnode(dev)
        def circuit(x):
            _feature_map(x, wires, reps)
            return ([qml.expval(qml.PauliX(w)) for w in wires]
                    + [qml.expval(qml.PauliY(w)) for w in wires]
                    + [qml.expval(qml.PauliZ(w)) for w in wires])

        self.circuit = circuit

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("circuit")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_circuit()

    def _features(self, X):
        return np.array([self.circuit(x) for x in np.asarray(X)], dtype=float)

    def fit(self, X, y):
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import GridSearchCV

        F = self._features(X)
        gs = GridSearchCV(KernelRidge(kernel="rbf"),
                          {"alpha": [0.1, 1, 10],
                           "gamma": [0.1, 0.5, 1, 2]}, cv=3)
        gs.fit(F, y)
        self.model_, self.params_ = gs.best_estimator_, gs.best_params_
        return self

    def predict(self, X):
        return self.model_.predict(self._features(X))


def _fit_eval(model, Ztr, ytr, Zte, yte):
    t0 = time.time()
    model.fit(Ztr, ytr)
    pred = model.predict(Zte)
    return {"r2": round(float(r2_score(yte, pred)), 4),
            "mae": round(float(mean_absolute_error(yte, pred)), 2),
            "train_seconds": round(time.time() - t0, 2)}


def run(budget=8):
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED)
    plan = fit_budget(Xtr, budget)  # 10 features > 8 -> PCA to [0, pi]
    Ztr, Zte = plan.transform(Xtr), plan.transform(Xte)

    result = {
        "dataset": {"name": "diabetes", "n_samples": len(X),
                    "n_features": X.shape[1], "task": "regression",
                    "target": "disease progression score"},
        "budget": {"n_qubits": plan.n_qubits, "method": plan.method,
                   "explained_variance": plan.explained_variance},
        "heldout": {},
    }
    models = {
        "vqr": VQR(n_qubits=plan.n_qubits),
        "pqk_ridge": PQKRidge(n_qubits=plan.n_qubits),
        "ridge": Ridge(random_state=SEED),
        "rf": RandomForestRegressor(random_state=SEED),
    }
    for name, model in models.items():
        result["heldout"][name] = _fit_eval(model, Ztr, ytr, Zte, yte)
        print(f"  diabetes/{name}: {result['heldout'][name]}", flush=True)

    # deployment floor: Ridge on full 10 features, no qubit budget
    scaler = StandardScaler().fit(Xtr)
    result["heldout"]["ridge_full"] = _fit_eval(
        Ridge(random_state=SEED), scaler.transform(Xtr), ytr,
        scaler.transform(Xte), yte)
    result["heldout"]["ridge_full"]["full_features"] = True
    print(f"  diabetes/ridge_full: {result['heldout']['ridge_full']}",
          flush=True)

    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / "regression_diabetes.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  wrote {out}", flush=True)
    return result


if __name__ == "__main__":
    result = run()
    print(f"\n{'model':<12}{'r2':>8}{'mae':>8}{'train_s':>10}")
    for name, m in result["heldout"].items():
        print(f"{name:<12}{m['r2']:>8.3f}{m['mae']:>8.1f}"
              f"{m['train_seconds']:>10.1f}")
    assert max(m["r2"] for m in result["heldout"].values()) > 0.3
    print("regression track OK")
