"""Quantum models: QSVM (quantum kernel) and VQC (variational classifier).

Simulator-first (default.qubit / lightning.qubit). Hardware execution lives in
hardware.py — training never touches the QPU.
"""
import numpy as np
import pennylane as qml
from sklearn.svm import SVC

SEED = 42


def _device(n_qubits):
    try:
        return qml.device("lightning.qubit", wires=n_qubits)
    except Exception:  # lightning missing on some platforms
        return qml.device("default.qubit", wires=n_qubits)


# ---------------------------------------------------------------- QSVM ------

def make_kernel(n_qubits, reps=1):
    """Fidelity kernel with a ZZ-style entangling feature map."""
    dev = _device(n_qubits)
    wires = list(range(n_qubits))

    def feature_map(x):
        for _ in range(reps):
            qml.AngleEmbedding(x, wires=wires, rotation="Y")
            for i in range(n_qubits - 1):
                qml.IsingZZ(x[i] * x[i + 1], wires=[i, i + 1])

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        feature_map(x1)
        qml.adjoint(feature_map)(x2)
        return qml.probs(wires=wires)

    def kernel(x1, x2):
        return float(kernel_circuit(x1, x2)[0])

    return kernel


def kernel_matrix(kernel, A, B=None, symmetric=False):
    B = A if B is None else B
    K = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        start = i if symmetric else 0
        for j in range(start, len(B)):
            K[i, j] = kernel(a, B[j])
    if symmetric:
        K = K + K.T - np.diag(np.diag(K))
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
        state.pop("kernel")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.kernel = make_kernel(self.n_qubits, self.reps)

    def fit(self, X, y):
        self.X_train_ = np.asarray(X)
        K = kernel_matrix(self.kernel, self.X_train_, symmetric=True)
        self.K_train_ = K
        self.svc.fit(K, y)
        return self

    def predict_proba(self, X):
        K = kernel_matrix(self.kernel, np.asarray(X), self.X_train_)
        return self.svc.predict_proba(K)

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
        state.pop("circuit")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_circuit()

    def _scores(self, weights, bias, X):
        return np.array([self.circuit(weights, x) for x in X]) + bias

    def fit(self, X, y):
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
        for _ in range(self.epochs):  # true epochs: every batch, every pass
            order = self.rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                weights, bias = opt.step(
                    lambda w, b: cost(w, b, X[idx], y_pm[idx]), weights, bias
                )
        self.weights_, self.bias_ = weights, bias
        return self

    def predict_proba(self, X):
        s = self._scores(self.weights_, self.bias_, np.asarray(X))
        p1 = np.clip((s + 1) / 2, 0, 1)  # ponytail: affine score->prob, Platt scaling if calibration matters
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
    print(f"moons: qsvm acc={acc_q:.2f}, vqc acc={acc_v:.2f}")
    assert acc_q > 0.8 and acc_v > 0.7
    print("quantum models OK")
