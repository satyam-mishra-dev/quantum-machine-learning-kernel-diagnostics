"""Quantum Readiness Score: a deterministic 0-100 rating of how well a
dataset suits quantum-vs-classical benchmarking on this platform.

Four subscores, each 0-100:

small_data_fit (weight 0.30)
    Quantum kernels only plausibly help in the small-data regime.
    100 for 100 <= n <= 1000; linear taper to 40 by n = 5000 (and 40
    beyond); 60 below n = 50 (too small to trust any estimate); linear
    ramp 60 -> 100 between n = 50 and n = 100.

compressibility (weight 0.25)
    How much signal survives the qubit-budget compression, from the
    auto-budgeter's cross-validated AUROC: 100 * min(1, cv_auroc / 0.85).
    Falls back to 70 when the budget record has no cv_auroc.

quantum_geometry (weight 0.30)
    From the advantage scout. 100 if verdict == "advantage_possible";
    otherwise g_ratio / 0.7 * 70 capped at 70 with a floor of 30.
    +10 bonus if kta_quantum > 2 * kta_classical (capped at 100).

class_balance (weight 0.15)
    100 at 50/50, tapering linearly to 30 at 95/5 (clipped below).

overall = 0.30*small_data_fit + 0.25*compressibility
        + 0.30*quantum_geometry + 0.15*class_balance
"""


def _small_data_fit(n):
    if n < 50:
        return 60.0
    if n < 100:
        return 60.0 + (n - 50) / 50 * 40.0
    if n <= 1000:
        return 100.0
    if n < 5000:
        return 100.0 - (n - 1000) / 4000 * 60.0
    return 40.0


def readiness(n_samples, n_features, class_balance, scout_dict, budget_dict):
    """Score a dataset's fit for quantum benchmarking. See module docstring."""
    small = _small_data_fit(n_samples)

    cv = (budget_dict or {}).get("cv_auroc")
    compress = 100.0 * min(1.0, cv / 0.85) if cv else 70.0

    s = scout_dict or {}
    if s.get("verdict") == "advantage_possible":
        geom = 100.0
    else:
        geom = max(30.0, min(70.0, (s.get("g_ratio") or 0.0) / 0.7 * 70.0))
    if s.get("kta_quantum", 0.0) > 2 * s.get("kta_classical", 0.0):
        geom = min(100.0, geom + 10.0)

    minority = min(class_balance, 1.0 - class_balance)
    balance = max(30.0, min(100.0, 30.0 + (minority - 0.05) / 0.45 * 70.0))

    overall = (0.30 * small + 0.25 * compress
               + 0.30 * geom + 0.15 * balance)

    if geom >= 90:
        verdict = ("Good candidate: the scout sees kernel structure a "
                   "classical kernel may miss — worth benchmarking on real "
                   "hardware.")
    elif overall >= 70:
        verdict = ("Good candidate for quantum benchmarking; advantage "
                   "unlikely, but the classical floor is guaranteed either "
                   "way.")
    elif overall >= 50:
        verdict = ("Reasonable benchmark candidate, but expect classical "
                   "models to match the quantum ones on this data.")
    else:
        verdict = ("Poor fit for quantum benchmarking: the data regime "
                   "gives quantum kernels little room to help.")

    return {
        "overall": round(overall, 1),
        "subscores": {
            "small_data_fit": round(small, 1),
            "compressibility": round(compress, 1),
            "quantum_geometry": round(geom, 1),
            "class_balance": round(balance, 1),
        },
        "verdict_text": verdict,
    }


if __name__ == "__main__":
    r = readiness(300, 13, 0.46,
                  {"verdict": "classical_will_match", "g_ratio": 0.45,
                   "kta_quantum": 0.18, "kta_classical": 0.14},
                  {"cv_auroc": 0.82})
    assert 0 <= r["overall"] <= 100
    assert r["subscores"]["small_data_fit"] == 100.0
    assert r["subscores"]["compressibility"] == round(100 * 0.82 / 0.85, 1)
    assert r["subscores"]["quantum_geometry"] == 45.0  # 0.45/0.7*70
    assert readiness(300, 13, 0.5, {"verdict": "advantage_possible"},
                     {})["subscores"]["quantum_geometry"] == 100.0
    assert readiness(20, 5, 0.05, {}, {})["subscores"]["class_balance"] == 30.0
    assert readiness(7000, 5, 0.5, {}, {})["subscores"]["small_data_fit"] == 40.0
    print("readiness OK", r)
