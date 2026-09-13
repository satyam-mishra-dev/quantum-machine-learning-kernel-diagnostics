"""Quantum models: QSVM (quantum kernel) and VQC (variational classifier).

Simulator-first (default.qubit / lightning.qubit). Hardware execution lives in
hardware.py — training never touches the QPU.
"""
import numpy as np
import pennylane as qml
from sklearn.svm import SVC

SEED = 42

import os
import pennylane as qml

class QuantumBackendManager:
    _instance = None
    
    @classmethod
    def get_info(cls, n_qubits=2):
        if not cls._instance:
            cls._instance = cls(n_qubits)
        cls._instance.get_device()
        return cls._instance.execution_mode, cls._instance.error

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.token = os.environ.get("IBM_QUANTUM_TOKEN") or os.environ.get("QISKIT_IBM_TOKEN")
        self.instance = os.environ.get("QISKIT_IBM_INSTANCE", "ibm_brisbane")
        self.execution_mode = "SIMULATED EXECUTION"
        self.error = None
        self.dev = None

    def get_device(self):
        if self.dev: return self.dev
        
        # Hardware execution via QPU has been removed to prevent PennyLane 0.45.1 incompatibilities
        # and limit execution to fast local simulation.
                
        try:
            self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
        except Exception:
            self.dev = qml.device("default.qubit", wires=self.n_qubits)
        return self.dev

def _device(n_qubits):
    manager = QuantumBackendManager(n_qubits)
    QuantumBackendManager._instance = manager
    return manager.get_device()


# ---------------------------------------------------------------- QSVM ------

def _feature_map(x, wires, reps):
    """Shared AngleEmbedding + IsingZZ encoding (QSVM and PQK)."""
    for _ in range(reps):
        qml.AngleEmbedding(x, wires=wires, rotation="Y")
        for i in range(len(wires) - 1):
            qml.IsingZZ(x[i] * x[i + 1], wires=[wires[i], wires[i + 1]])


def make_kernel(n_qubits, reps=1):
    """Fidelity kernel with a ZZ-style entangling feature map."""
    dev = _device(n_qubits)
    wires = list(range(n_qubits))

    def feature_map(x):
        _feature_map(x, wires, reps)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        feature_map(x1)
        qml.adjoint(feature_map)(x2)
        return qml.probs(wires=wires)

    def kernel(x1, x2):
        return float(kernel_circuit(x1, x2)[0])

    return kernel


import time

def kernel_matrix(kernel, A, B=None, symmetric=False, callback=None, callback_interval=10):
    """Compute kernel matrix with progress reporting.
    
    callback_interval: report every N evaluations (default 10 for responsive UI)
    For training (symmetric=True): evaluates n*(n+1)//2 pairs
    For test (symmetric=False): evaluates len(A)*len(B) pairs
    """
    B = A if B is None else B
    n_a, n_b = len(A), len(B)
    K = np.zeros((n_a, n_b))
    
    if symmetric:
        total_evals = (n_a * (n_a + 1)) // 2
    else:
        total_evals = n_a * n_b
    
    completed = 0
    start_time = time.time()
    for i, a in enumerate(A):
        start_j = i if symmetric else 0
        for j in range(start_j, n_b):
            val = kernel(a, B[j])
            K[i, j] = val
            if symmetric and i != j:
                K[j, i] = val  # fill symmetric immediately
            completed += 1
            
            if callback and (completed % callback_interval == 0 or completed == total_evals):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_evals - completed) / rate if rate > 0 else None
                callback(completed, total_evals, elapsed, rate, eta)
                
    if symmetric:
        # Ensure diagonal is correct (self-similarity = 1.0 for fidelity kernels)
        # Already filled symmetric pairs above; just fix diagonal if needed
        pass
        
    return K


class QSVM:
    """SVC on a precomputed quantum fidelity kernel."""

    def __init__(self, n_qubits, reps=1, C=1.0):
        self.n_qubits, self.reps = n_qubits, reps
        self.kernel = make_kernel(n_qubits, reps)
        self.svc = SVC(kernel="precomputed", C=C, probability=True,
                       random_state=SEED)

    def __getstate__(self):  # qnode closures don't pickle; rebuild on load
        state = self.__dict__.copy()
        state.pop("kernel", None)
        state.pop("kernel_callback", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.kernel = make_kernel(self.n_qubits, self.reps)

    def fit(self, X, y):
        self.X_train_ = np.asarray(X)
        cb = getattr(self, 'kernel_callback', None)
        n = len(self.X_train_)
        total_train = (n * (n + 1)) // 2
        # Wrap callback to prefix training phase
        train_cb = None
        if cb:
            def train_cb(completed, total, elapsed, rate, eta):
                cb(completed, total_train, elapsed, rate, eta, phase="TRAIN_KERNEL")
        K = kernel_matrix(self.kernel, self.X_train_, symmetric=True, callback=train_cb, callback_interval=5)
        self.K_train_ = K
        self.svc.fit(K, y)
        return self

    def predict_proba(self, X):
        cb = getattr(self, 'kernel_callback', None)
        X = np.asarray(X)
        n_test = len(X)
        n_train = len(self.X_train_)
        total_test = n_test * n_train
        test_cb = None
        if cb:
            def test_cb(completed, total, elapsed, rate, eta):
                cb(completed, total_test, elapsed, rate, eta, phase="TEST_KERNEL")
        K = kernel_matrix(self.kernel, X, self.X_train_, callback=test_cb, callback_interval=5)
        return self.svc.predict_proba(K)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ----------------------------------------------------------------- PQK ------

class PQK:
    """Projected quantum kernel: same feature map as QSVM, but measure 1-local
    <X>, <Y>, <Z> on every wire (3*n_qubits features, N circuit evals — linear,
    not N^2) and classify with an RBF SVC on top."""

    GAMMAS = (0.1, 0.5, 1, 2)

    def __init__(self, n_qubits, reps=1, C=1.0):
        self.n_qubits, self.reps, self.C = n_qubits, reps, C
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

    def __getstate__(self):  # qnode closures don't pickle; rebuild on load
        state = self.__dict__.copy()
        state.pop("circuit", None)
        state.pop("kernel_callback", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_circuit()

    def _features(self, X, callback=None):
        rows = np.asarray(X)
        result = []
        start_time = time.time()
        for idx, x in enumerate(rows):
            result.append(self.circuit(x))
            if callback and ((idx + 1) % 5 == 0 or idx + 1 == len(rows)):
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(rows) - idx - 1) / rate if rate > 0 else None
                callback(idx + 1, len(rows), elapsed, rate, eta)
        return np.array(result, dtype=float)

    def fit(self, X, y):
        from sklearn.model_selection import GridSearchCV

        cb = getattr(self, 'kernel_callback', None)
        F = self._features(X, callback=cb)
        gs = GridSearchCV(
            SVC(kernel="rbf", C=self.C, probability=True, random_state=SEED),
            {"gamma": list(self.GAMMAS)}, cv=3,
        )
        gs.fit(F, y)
        self.svc = gs.best_estimator_
        self.gamma_ = gs.best_params_["gamma"]
        self.X_train_features_ = F
        return self

    def predict_proba(self, X):
        return self.svc.predict_proba(self._features(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ----------------------------------------------------------------- VQC ------

class VQC:
    """Variational classifier: angle embedding + StronglyEntanglingLayers."""

    def __init__(self, n_qubits, n_layers=3, epochs=30, lr=0.1,
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

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("circuit", None)
        state.pop("epoch_callback", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_circuit()

    def _scores(self, weights, bias, X):
        return np.array([self.circuit(weights, x) for x in X]) + bias

    def fit(self, X, y, epoch_callback=None):
        from pennylane import numpy as pnp

        X, y = np.asarray(X), np.asarray(y)
        y_pm = 2 * y - 1  # {-1, +1}
        shape = qml.StronglyEntanglingLayers.shape(self.n_layers,
                                                   self.n_qubits)
        weights = pnp.array(0.1 * self.rng.normal(size=shape),
                            requires_grad=True)
        bias = pnp.array(0.0, requires_grad=True)
        opt = qml.AdamOptimizer(self.lr)

        def cost(w, b, Xb, yb):
            scores = pnp.stack([self.circuit(w, x) for x in Xb]) + b
            return pnp.mean((scores - yb) ** 2)

        n = len(X)
        for epoch in range(self.epochs):  # true epochs: every batch, every pass
            order = self.rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                weights, bias = opt.step(
                    lambda w, b: cost(w, b, X[idx], y_pm[idx]), weights, bias
                )
            if epoch_callback:
                # evaluate full cost for logging
                c = cost(weights, bias, X, y_pm)
                epoch_callback(epoch + 1, self.epochs, float(c))
                
        self.weights_, self.bias_ = weights, bias
        return self

    def predict_proba(self, X):
        s = self._scores(self.weights_, self.bias_, np.asarray(X))
        p1 = np.clip((s + 1) / 2, 0, 1)  # affine score->prob
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


if __name__ == "__main__":
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    X, y = make_moons(120, noise=0.15, random_state=0)
    X = MinMaxScaler((0, np.pi)).fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=0, stratify=y)
    q = QSVM(n_qubits=2).fit(Xtr, ytr)
    acc_q = (q.predict(Xte) == yte).mean()
    v = VQC(n_qubits=2, epochs=40).fit(Xtr, ytr)
    acc_v = (v.predict(Xte) == yte).mean()
    p = PQK(n_qubits=2).fit(Xtr, ytr)
    acc_p = (p.predict(Xte) == yte).mean()
    import pickle
    p2 = pickle.loads(pickle.dumps(p))
    assert (p2.predict(Xte) == p.predict(Xte)).all()
    print(f"moons: qsvm acc={acc_q:.2f}, vqc acc={acc_v:.2f}, "
          f"pqk acc={acc_p:.2f} (gamma={p.gamma_})")
    assert acc_q > 0.8 and acc_v > 0.7 and acc_p > 0.7
    print("quantum models OK")
