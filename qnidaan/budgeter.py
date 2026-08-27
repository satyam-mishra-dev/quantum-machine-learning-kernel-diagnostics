"""Qubit Budgeter: compress any feature matrix to fit a qubit budget.

Angle encoding uses one qubit per feature, so d features -> n_qubits = d.
If d <= budget we just scale; otherwise PCA down to the budget. Output is
scaled to [0, pi] so it can feed rotation gates directly.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class BudgetPlan:
    n_features_in: int
    n_qubits: int
    method: str            # "scale" | "pca" | "select"
    explained_variance: float | None = None
    pipeline: Pipeline = field(default=None, repr=False)
    selected_features: list | None = None  # "select" only
    cv_auroc: float | None = None            # auto mode: winner's CV score
    runner_up_cv_auroc: float | None = None  # auto mode: loser's CV score

    def transform(self, X):
        return self.pipeline.transform(X)


def fit_budget(X_train, budget: int = 8, y=None,
               method: str = "pca") -> BudgetPlan:
    d = X_train.shape[1]
    if d <= budget:
        pipe = Pipeline([("scale", MinMaxScaler((0, np.pi)))])
        pipe.fit(X_train)
        return BudgetPlan(d, d, "scale", None, pipe)
    if method == "select":  # mutual-information top-k, keeps raw features
        if y is None:
            raise ValueError('method="select" requires y')
        pipe = Pipeline([
            ("select", SelectKBest(mutual_info_classif, k=budget)),
            ("scale", MinMaxScaler((0, np.pi))),
        ])
        pipe.fit(X_train, y)
        idx = pipe.named_steps["select"].get_support(indices=True)
        return BudgetPlan(d, budget, "select", None, pipe,
                          [int(i) for i in idx])
    pipe = Pipeline([
        ("std", StandardScaler()),
        ("pca", PCA(n_components=budget, random_state=0)),
        ("scale", MinMaxScaler((0, np.pi))),
    ])
    pipe.fit(X_train)
    ev = float(pipe.named_steps["pca"].explained_variance_ratio_.sum())
    return BudgetPlan(d, budget, "pca", ev, pipe)


def fit_budget_auto(X_train, y, budget: int = 8) -> BudgetPlan:
    """Fit both pca and select plans, keep whichever CV-scores higher.

    Scored by 5-fold StratifiedKFold AUROC of LogisticRegression on the
    transformed training data (ties go to pca). d <= budget is just "scale".
    """
    if X_train.shape[1] <= budget:
        return fit_budget(X_train, budget)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    plans, scores = {}, {}
    for method in ("pca", "select"):
        plans[method] = fit_budget(X_train, budget, y=y, method=method)
        scores[method] = float(np.mean(cross_val_score(
            LogisticRegression(max_iter=2000),
            plans[method].transform(X_train), y,
            cv=cv, scoring="roc_auc")))
    winner = "select" if scores["select"] > scores["pca"] else "pca"
    plan = plans[winner]
    plan.cv_auroc = round(scores[winner], 4)
    plan.runner_up_cv_auroc = round(
        scores["pca" if winner == "select" else "select"], 4)
    return plan


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 30))
    plan = fit_budget(X, budget=8)
    Z = plan.transform(X)
    assert Z.shape == (100, 8) and plan.method == "pca"
    assert Z.min() >= -1e-9 and Z.max() <= np.pi + 1e-9
    small = fit_budget(X[:, :5], budget=8)
    assert small.n_qubits == 5 and small.method == "scale"
    y = (X[:, 3] + 0.1 * rng.normal(size=100) > 0).astype(int)
    sel = fit_budget(X, budget=8, y=y, method="select")
    Zs = sel.transform(X)
    assert Zs.shape == (100, 8) and sel.method == "select"
    assert len(sel.selected_features) == 8 and 3 in sel.selected_features
    assert Zs.min() >= -1e-9 and Zs.max() <= np.pi + 1e-9
    try:
        fit_budget(X, budget=8, method="select")
        raise AssertionError("select without y should raise")
    except ValueError:
        pass
    auto = fit_budget_auto(X, y, budget=8)
    assert auto.method in ("pca", "select")
    assert auto.cv_auroc is not None and auto.runner_up_cv_auroc is not None
    assert auto.cv_auroc >= auto.runner_up_cv_auroc
    assert auto.transform(X).shape == (100, 8)
    auto_small = fit_budget_auto(X[:, :5], y, budget=8)
    assert auto_small.method == "scale" and auto_small.cv_auroc is None
    print(f"budgeter OK (pca kept {plan.explained_variance:.0%} variance, "
          f"select kept {sel.selected_features}, auto chose {auto.method} "
          f"cv={auto.cv_auroc} vs {auto.runner_up_cv_auroc})")
