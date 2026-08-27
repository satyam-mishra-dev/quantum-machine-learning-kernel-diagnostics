"""Advantage Scout: pre-training diagnostics for "can quantum help here?"

- Kernel-target alignment (KTA) of the quantum kernel vs the labels.
- Geometric difference g(K_classical, K_quantum) from Huang et al. 2021
  ("Power of data in quantum machine learning", Nat. Commun. 12, 2631):
  g ~ sqrt(||sqrt(Kc) Kq^-1 sqrt(Kc)||_inf). g >> 1 on a dataset means the
  quantum kernel's geometry differs enough that advantage is *possible*;
  g ~ 1 means classical will match it, skip quantum.
"""
import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import rbf_kernel


def kernel_target_alignment(K, y):
    y_pm = 2 * np.asarray(y) - 1
    Y = np.outer(y_pm, y_pm)
    return float(np.sum(K * Y) / (np.linalg.norm(K) * np.linalg.norm(Y)))


def geometric_difference(K_classical, K_quantum, reg=1e-7):
    n = len(K_quantum)
    sq = sqrtm(K_classical).real
    Kq_inv = np.linalg.inv(K_quantum + reg * n * np.eye(n))
    M = sq @ Kq_inv @ sq
    return float(np.sqrt(np.linalg.norm(M, ord=2)))


def scout(Z_train, y_train, quantum_kernel, max_samples=100, seed=42):
    """Cheap pre-training verdict on a subsample of the training set."""
    from qnidaan.quantum import kernel_matrix

    rng = np.random.default_rng(seed)
    if len(Z_train) > max_samples:
        idx = rng.permutation(len(Z_train))[:max_samples]
        Z_train, y_train = Z_train[idx], np.asarray(y_train)[idx]

    Kq = kernel_matrix(quantum_kernel, Z_train, symmetric=True)
    Kc = rbf_kernel(Z_train)
    kta_q = kernel_target_alignment(Kq, y_train)
    kta_c = kernel_target_alignment(Kc, y_train)
    g = geometric_difference(Kc, Kq)
    sqrt_n = np.sqrt(len(Z_train))
    return {
        "n_scouted": len(Z_train),
        "kta_quantum": kta_q,
        "kta_classical": kta_c,
        "geometric_difference": g,
        "g_threshold_sqrt_n": float(sqrt_n),
        # advantage possible if geometry differs materially AND the quantum
        # kernel actually aligns with labels
        "quantum_worth_trying": bool(g > 0.25 * sqrt_n and kta_q > 0.05),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 40)
    y_pm = 2 * y - 1
    K_perfect = np.outer(y_pm, y_pm).astype(float)
    assert kernel_target_alignment(K_perfect, y) > 0.99
    K = rbf_kernel(rng.normal(size=(40, 4)))
    assert abs(geometric_difference(K, K) - 1.0) < 0.2  # same kernel -> g~1
    print("scout OK")
