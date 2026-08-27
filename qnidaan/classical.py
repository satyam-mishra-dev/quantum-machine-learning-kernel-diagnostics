"""Classical twins: the honest baselines every quantum model competes against.

Same compressed features, same splits, same metric policy as the quantum side.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.svm import SVC

SEED = 42


def make_twins():
    return {
        "logreg": LogisticRegression(max_iter=2000, random_state=SEED),
        "svm_rbf": CalibratedClassifierCV(  # calibrated so we get real probs
            SVC(kernel="rbf", random_state=SEED), cv=3
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=SEED
        ),
    }


def specificity_at_sensitivity(y_true, y_score, target_sens=0.95):
    """Specificity at the lowest threshold achieving >= target sensitivity."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = tpr >= target_sens
    if not ok.any():
        return 0.0
    return float(1 - fpr[ok][0])


def evaluate(model, Xte, yte):
    prob = model.predict_proba(Xte)[:, 1]
    return {
        "auroc": float(roc_auc_score(yte, prob)),
        "accuracy": float(((prob >= 0.5).astype(int) == yte).mean()),
        "spec_at_95sens": specificity_at_sensitivity(yte, prob),
    }


if __name__ == "__main__":
    from qnidaan.budgeter import fit_budget
    from qnidaan.datasets import load_split

    Xtr, Xte, ytr, yte, meta = load_split("wdbc")
    plan = fit_budget(Xtr, budget=8)
    Ztr, Zte = plan.transform(Xtr), plan.transform(Xte)
    for name, m in make_twins().items():
        m.fit(Ztr, ytr)
        r = evaluate(m, Zte, yte)
        assert r["auroc"] > 0.9, f"{name} suspiciously weak: {r}"
        print(f"wdbc/{name}: auroc={r['auroc']:.3f} acc={r['accuracy']:.3f} "
              f"spec@95sens={r['spec_at_95sens']:.3f}")
    print("classical twins OK")
