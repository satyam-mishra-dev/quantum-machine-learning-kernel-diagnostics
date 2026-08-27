"""Compression-method experiment: PCA vs mutual-info selection ("select").

Same protocol as the harness (fixed seed-42 splits, tuned budget/C/reps),
AUROC on the untouched test split. Writes runs/compress_exp.json.
"""
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from qnidaan.budgeter import fit_budget
from qnidaan.classical import evaluate, make_twins
from qnidaan.datasets import load_split
from qnidaan.quantum import PQK, QSVM

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
SEED = 42
DATASETS = ["pima", "cleveland", "wdbc"]
METHODS = ["pca", "select"]


def main():
    tuned = json.loads((RUNS_DIR / "tuned.json").read_text())
    results = {}
    for name in DATASETS:
        Xtr, Xte, ytr, yte, meta = load_split(name)
        cfg = tuned[name]
        Xfit, _, yfit, _ = train_test_split(
            Xtr, ytr, test_size=0.2, random_state=SEED, stratify=ytr
        )
        results[name] = {}
        for method in METHODS:
            plan = fit_budget(Xfit, budget=cfg["budget"],
                              y=yfit if method == "select" else None,
                              method=method)
            Zfit, Zte = plan.transform(Xfit), plan.transform(Xte)
            twins = make_twins()
            models = {
                "logreg": twins["logreg"],
                "svm_rbf": twins["svm_rbf"],
                "qsvm": QSVM(n_qubits=plan.n_qubits, C=cfg["C"],
                             reps=cfg["reps"]),
                "pqk": PQK(n_qubits=plan.n_qubits, reps=cfg["reps"]),
            }
            scores = {}
            for mname, model in models.items():
                model.fit(Zfit, yfit)
                scores[mname] = round(evaluate(model, Zte, yte)["auroc"], 4)
                print(f"{name}/{method}/{mname}: {scores[mname]}", flush=True)
            results[name][method] = scores
            if plan.selected_features is not None:
                feats = [meta["feature_names"][i]
                         for i in plan.selected_features]
                results[name]["selected_features"] = feats
                print(f"{name} selected: {feats}", flush=True)
    out = RUNS_DIR / "compress_exp.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
