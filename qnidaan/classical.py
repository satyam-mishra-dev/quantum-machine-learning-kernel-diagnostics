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


from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix,
    average_precision_score, accuracy_score
)

def evaluate(model, Xte, yte):
    try:
        prob = model.predict_proba(Xte)[:, 1]
    except Exception:
        prob = model.decision_function(Xte)
        
    y_pred = (prob >= 0.5).astype(int)
    
    fpr, tpr, _ = roc_curve(yte, prob)
    prec, rec, _ = precision_recall_curve(yte, prob)
    
    try:
        auroc = float(roc_auc_score(yte, prob))
    except ValueError:
        auroc = float('nan')
        
    try:
        auprc = float(average_precision_score(yte, prob))
    except ValueError:
        auprc = float('nan')
        
    cm = confusion_matrix(yte, y_pred, labels=[0,1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn+fp) > 0 else 0.0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        sensitivity = 0.0
        specificity = 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": float(accuracy_score(yte, y_pred)),
        "precision": float(precision_score(yte, y_pred, zero_division=0)),
        "recall": float(recall_score(yte, y_pred, zero_division=0)),
        "f1": float(f1_score(yte, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "spec_at_95sens": float(specificity_at_sensitivity(yte, prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": {"fpr": fpr[::max(1, len(fpr)//50)].tolist(), "tpr": tpr[::max(1, len(tpr)//50)].tolist()},
        "pr_curve": {"precision": prec[::max(1, len(prec)//50)].tolist(), "recall": rec[::max(1, len(rec)//50)].tolist()},
        "sample_count": len(yte)
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
