"""Epoch-2 tuner: small CV grid for QSVM per dataset.

Grid: qubit budget x C x feature-map reps, scored by 3-fold CV AUROC on a
subsample of the training split (test split never touched). Winners land in
runs/tuned.json, which the harness picks up automatically.
"""
import json
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from qnidaan.budgeter import fit_budget
from qnidaan.datasets import load_split
from qnidaan.harness import RUNS_DIR
from qnidaan.quantum import QSVM

SEED = 42
MAX_CV_SAMPLES = 150  # ponytail: subsample cap keeps kernel CV minutes, not hours
GRID = {"budget": [4, 8], "C": [1, 10, 100], "reps": [1, 2]}


def tune_dataset(name):
    Xtr, _, ytr, _, _ = load_split(name)
    rng = np.random.default_rng(SEED)
    if len(ytr) > MAX_CV_SAMPLES:
        idx = rng.permutation(len(ytr))[:MAX_CV_SAMPLES]
        Xtr, ytr = Xtr[idx], ytr[idx]

    best = None
    for budget in GRID["budget"]:
        plan = fit_budget(Xtr, budget=budget)
        Z = plan.transform(Xtr)
        for C in GRID["C"]:
            for reps in GRID["reps"]:
                t0 = time.time()
                aurocs = []
                cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
                for tr, va in cv.split(Z, ytr):
                    m = QSVM(n_qubits=plan.n_qubits, reps=reps, C=C)
                    m.fit(Z[tr], ytr[tr])
                    aurocs.append(roc_auc_score(
                        ytr[va], m.predict_proba(Z[va])[:, 1]))
                score = float(np.mean(aurocs))
                cfg = {"budget": budget, "C": C, "reps": reps,
                       "cv_auroc": round(score, 4)}
                print(f"  {name} {cfg} ({time.time()-t0:.0f}s)", flush=True)
                if best is None or score > best["cv_auroc"]:
                    best = cfg
    return best


if __name__ == "__main__":
    path = RUNS_DIR / "tuned.json"
    tuned = json.loads(path.read_text()) if path.exists() else {}
    for name in sys.argv[1:] or ["cleveland", "wdbc", "pima"]:
        print(f"=== tuning {name} ===", flush=True)
        tuned[name] = tune_dataset(name)
        path.write_text(json.dumps(tuned, indent=2))
        print(f"best for {name}: {tuned[name]}", flush=True)
