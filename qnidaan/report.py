"""Trust layer: split conformal prediction + per-patient explanation.

Conformal is hand-rolled (adaptive prediction sets for binary labels) so the
math is fully visible in the demo — no library API in the way. Guarantees
P(true label in prediction set) >= 1 - alpha on exchangeable data.

Explanations come from SHAP on the calibrated classical twin as a surrogate
(exact & fast); the report labels them as surrogate explanations.
"""
import numpy as np

ALPHA = 0.1  # 90% coverage target


def fit_conformal(model, Z_cal, y_cal, alpha=ALPHA):
    """Return the conformal quantile qhat from a held-out calibration set."""
    probs = model.predict_proba(Z_cal)
    scores = 1 - probs[np.arange(len(y_cal)), y_cal]  # 1 - p(true class)
    n = len(scores)
    if np.ceil((n + 1) * (1 - alpha)) > n:
        return 1.0  # calibration set too small: sets are always {0,1}
    q = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n,
                    method="higher")
    return float(q)


def prediction_set(model, Z, qhat):
    """Classes whose score makes the cut; never empty (argmax fallback)."""
    probs = model.predict_proba(Z)
    sets = []
    for p in probs:
        s = [c for c in (0, 1) if 1 - p[c] <= qhat] or [int(np.argmax(p))]
        sets.append(s)
    return sets


def explain_patient(surrogate, Z_background, z, feature_names, top_k=5):
    """SHAP top features for one compressed-feature patient row."""
    import shap

    explainer = shap.Explainer(
        lambda X: surrogate.predict_proba(X)[:, 1], Z_background[:50]
    )
    sv = explainer(z.reshape(1, -1), silent=True).values[0]
    order = np.argsort(np.abs(sv))[::-1][:top_k]
    return [
        {"feature": feature_names[i] if i < len(feature_names)
         else f"component_{i}", "shap": float(sv[i])}
        for i in order
    ]


def patient_report(model, surrogate, qhat, z, Z_background, feature_names):
    prob = float(model.predict_proba(z.reshape(1, -1))[0, 1])
    pset = prediction_set(model, z.reshape(1, -1), qhat)[0]
    from qnidaan.quantum import VQC

    return {
        "risk_probability": round(prob, 4),
        # VQC maps its expval affinely to [0,1]; rank-valid, not calibrated
        "probability_calibrated": not isinstance(model, VQC),
        "prediction": int(prob >= 0.5),
        "conformal_set": pset,
        "confident": len(pset) == 1,
        "flag_for_review": len(pset) > 1,
        "coverage_guarantee": f"{int((1 - ALPHA) * 100)}%",
        "top_features": explain_patient(
            surrogate, Z_background, z, feature_names
        ),
        "explanation_source": "SHAP on calibrated classical surrogate",
    }


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + 0.3 * rng.normal(size=300) > 0).astype(int)
    m = LogisticRegression().fit(X[:200], y[:200])
    qhat = fit_conformal(m, X[200:250], y[200:250])
    sets = prediction_set(m, X[250:], qhat)
    covered = np.mean([y[250 + i] in s for i, s in enumerate(sets)])
    assert covered >= 0.85, f"coverage {covered} below target"
    assert all(len(s) >= 1 for s in sets)
    print(f"conformal OK (empirical coverage {covered:.2f})")
