"""Benchmark harness: one dataset in, one honest results JSON out.

Per dataset: budget -> scout -> train quantum models + classical twins on the
same compressed features -> evaluate once on the untouched test split ->
sample-efficiency curves (the small-data claim, measured not promised).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from qnidaan.budgeter import fit_budget
from qnidaan.classical import evaluate, make_twins
from qnidaan.datasets import load_split
from qnidaan.quantum import QSVM, VQC, make_kernel
from qnidaan.scout import scout

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
SEED = 42
CURVE_SIZES = [30, 60, 100, 200]
CURVE_SEEDS = [0, 1, 2]


def _fit_eval(model, Ztr, ytr, Zte, yte):
    t0 = time.time()
    model.fit(Ztr, ytr)
    out = evaluate(model, Zte, yte)
    out["train_seconds"] = round(time.time() - t0, 2)
    return out


def sample_efficiency(Ztr, ytr, Zte, yte, n_qubits, qsvm_kw=None):
    """Accuracy/AUROC vs training-set size: QSVM vs classical twins."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    qsvm_kw = qsvm_kw or {"C": 10, "reps": 1}
    rng = np.random.default_rng(SEED)
    curve = []
    for size in [s for s in CURVE_SIZES if s <= len(ytr)]:
        point = {"n_train": size}
        for name, make in {
            "qsvm": lambda: QSVM(n_qubits=n_qubits, **qsvm_kw),
            "logreg": lambda: LogisticRegression(max_iter=2000),
            "svm_rbf": lambda: SVC(kernel="rbf"),
        }.items():
            aurocs = []
            for seed in CURVE_SEEDS:
                idx = _stratified_subsample(ytr, size, seed)
                m = make()
                m.fit(Ztr[idx], ytr[idx])
                if hasattr(m, "predict_proba"):
                    score = m.predict_proba(Zte)[:, 1]
                else:
                    score = m.decision_function(Zte)
                from sklearn.metrics import roc_auc_score
                aurocs.append(roc_auc_score(yte, score))
            point[name] = {
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_std": float(np.std(aurocs)),
            }
        curve.append(point)
    return curve


def _stratified_subsample(y, size, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        take = max(2, int(round(size * len(cls_idx) / len(y))))
        idx.extend(rng.choice(cls_idx, min(take, len(cls_idx)),
                              replace=False))
    return np.array(idx)


def _tuned_config(name):
    path = RUNS_DIR / "tuned.json"
    if path.exists():
        return json.loads(path.read_text()).get(name)
    return None


def run_dataset(name, budget=8, curves=True):
    Xtr, Xte, ytr, yte, meta = load_split(name)
    tuned = _tuned_config(name)
    qsvm_kw = {"C": 10, "reps": 1}
    if tuned:
        budget = tuned["budget"]
        qsvm_kw = {"C": tuned["C"], "reps": tuned["reps"]}
    plan = fit_budget(Xtr, budget=budget)
    Ztr, Zte = plan.transform(Xtr), plan.transform(Xte)

    result = {
        "dataset": meta,
        "budget": {
            "n_qubits": plan.n_qubits,
            "method": plan.method,
            "explained_variance": plan.explained_variance,
        },
    }

    kernel = make_kernel(plan.n_qubits, qsvm_kw["reps"])
    result["scout"] = scout(Ztr, ytr, kernel)

    # split conformal needs calibration data the model never trained on
    from sklearn.model_selection import train_test_split

    from qnidaan.report import fit_conformal

    Zfit, Zcal, yfit, ycal = train_test_split(
        Ztr, ytr, test_size=0.2, random_state=SEED, stratify=ytr
    )

    models = {
        **make_twins(),
        "qsvm": QSVM(n_qubits=plan.n_qubits, **qsvm_kw),
        "vqc": VQC(n_qubits=plan.n_qubits, epochs=60),
    }
    result["qsvm_config"] = {"tuned": bool(tuned), **qsvm_kw}
    result["heldout"] = {}
    qhats = {}
    for mname, model in models.items():
        result["heldout"][mname] = _fit_eval(model, Zfit, yfit, Zte, yte)
        qhats[mname] = fit_conformal(model, Zcal, ycal)
        result["heldout"][mname]["conformal_qhat"] = qhats[mname]
        print(f"  {name}/{mname}: {result['heldout'][mname]}", flush=True)

    # deployment floor: best classical on FULL features (no qubit budget).
    # The compressed twins are the fair comparison; this is the guarantee
    # that the platform never ships worse than plain classical ML.
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    Ftr, Fte = scaler.transform(Xtr), scaler.transform(Xte)
    for mname, model in make_twins().items():
        full_name = f"{mname}_full"
        result["heldout"][full_name] = _fit_eval(model, Ftr, ytr, Fte, yte)
        result["heldout"][full_name]["full_features"] = True
        print(f"  {name}/{full_name}: {result['heldout'][full_name]}",
              flush=True)

    if curves:
        result["sample_efficiency"] = sample_efficiency(
            Ztr, ytr, Zte, yte, plan.n_qubits, qsvm_kw
        )

    import joblib
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"plan": plan, "models": models, "meta": meta,
                 "qhats": qhats, "Z_background": Zfit[:50]},
                MODELS_DIR / f"{name}.joblib")

    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"{name}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  wrote {out}", flush=True)
    return result


if __name__ == "__main__":
    for name in sys.argv[1:] or ["wdbc", "cleveland", "pima"]:
        print(f"=== {name} ===", flush=True)
        run_dataset(name)
