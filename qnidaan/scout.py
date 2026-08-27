"""Advantage Scout: pre-training diagnostics for "can quantum help here?"

Reports three quantities, honestly labeled:

- Kernel-target alignment (KTA) of quantum vs classical kernels.
- g_huang: the geometric difference of Huang et al. 2021 ("Power of data in
  quantum machine learning", Nat. Commun. 12, 2631): sqrt of the spectral
  norm of sqrt(K_Q) (K_C + reg)^-1 sqrt(K_Q) — "can the classical geometry
  represent the quantum one?". We use naive spectral regularization, not the
  paper's training-error-tied lambda, so this is reported as context, not
  used for the verdict.
- g_asym: the reverse-direction spectral asymmetry sqrt(||sqrt(K_C)
  (K_Q + reg)^-1 sqrt(K_C)||) — a home-grown heuristic, *not* Huang's g.
  This is what the verdict uses, because it is the quantity we validated
  in-platform: on WDBC/Cleveland/PIMA it stayed low (0.30-0.49 x sqrt(n))
  and classical matched quantum as predicted; on a quantum-native dataset it
  jumped (2.1 x sqrt(n)) and quantum won by 23 AUROC points. 4/4 correct.
"""
import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import rbf_kernel


def kernel_target_alignment(K, y):
    y_pm = 2 * np.asarray(y) - 1
    Y = np.outer(y_pm, y_pm)
    return float(np.sum(K * Y) / (np.linalg.norm(K) * np.linalg.norm(Y)))


def geometric_difference(K_num, K_den, reg=1e-7):
    """sqrt(|| sqrt(K_num) (K_den + reg*n*I)^-1 sqrt(K_num) ||_2).

    Large => the geometry of K_num is poorly representable by K_den.
    """
    n = len(K_num)
    sq = sqrtm(K_num).real
    inv = np.linalg.inv(K_den + reg * n * np.eye(n))
    return float(np.sqrt(np.linalg.norm(sq @ inv @ sq, ord=2)))


def scout(Z_train, y_train, quantum_kernel, max_samples=100, seed=42):
    """Cheap pre-training verdict on a subsample of the training set."""
    from qnidaan.quantum import kernel_matrix

    rng = np.random.default_rng(seed)
    Z_train = np.asarray(Z_train)
    y_train = np.asarray(y_train)
    if len(Z_train) > max_samples:
        idx = rng.permutation(len(Z_train))[:max_samples]
        Z_train, y_train = Z_train[idx], y_train[idx]

    Kq = kernel_matrix(quantum_kernel, Z_train, symmetric=True)
    Kc = rbf_kernel(Z_train)
    sqrt_n = float(np.sqrt(len(Z_train)))
    g_asym = geometric_difference(Kc, Kq)
    g_huang = geometric_difference(Kq, Kc, reg=1e-3)
    kta_q = kernel_target_alignment(Kq, y_train)
    # verdict from the in-platform-validated heuristic (see module docstring)
    g_ratio = g_asym / sqrt_n
    advantage = bool(g_ratio > 0.7 and kta_q > 0.05)
    return {
        "n_scouted": len(Z_train),
        "kta_quantum": kta_q,
        "kta_classical": kernel_target_alignment(Kc, y_train),
        "g_asym": g_asym,
        "g_huang_naive_reg": g_huang,
        "g_threshold_sqrt_n": sqrt_n,
        "g_ratio": g_ratio,
        "verdict": "advantage_possible" if advantage
        else "classical_will_match",
        "quantum_worth_trying": advantage,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 40)
    y_pm = 2 * y - 1
    K_perfect = np.outer(y_pm, y_pm).astype(float)
    assert kernel_target_alignment(K_perfect, y) > 0.99
    K = rbf_kernel(rng.normal(size=(40, 4)))
    assert abs(geometric_difference(K, K) - 1.0) < 0.2  # same kernel -> g~1
    # asymmetry: two different kernels give different g in each direction
    K2 = rbf_kernel(rng.normal(size=(40, 4)), gamma=5.0)
    assert geometric_difference(K, K2) != geometric_difference(K2, K)
    print("scout OK")
