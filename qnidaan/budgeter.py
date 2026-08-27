"""Qubit Budgeter: compress any feature matrix to fit a qubit budget.

Angle encoding uses one qubit per feature, so d features -> n_qubits = d.
If d <= budget we just scale; otherwise PCA down to the budget. Output is
scaled to [0, pi] so it can feed rotation gates directly.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class BudgetPlan:
    n_features_in: int
    n_qubits: int
    method: str            # "scale" | "pca"
    explained_variance: float | None = None
    pipeline: Pipeline = field(default=None, repr=False)

    def transform(self, X):
        return self.pipeline.transform(X)


def fit_budget(X_train, budget: int = 8) -> BudgetPlan:
    d = X_train.shape[1]
    if d <= budget:
        pipe = Pipeline([("scale", MinMaxScaler((0, np.pi)))])
        pipe.fit(X_train)
        return BudgetPlan(d, d, "scale", None, pipe)
    pipe = Pipeline([
        ("std", StandardScaler()),
        ("pca", PCA(n_components=budget, random_state=0)),
        ("scale", MinMaxScaler((0, np.pi))),
    ])
    pipe.fit(X_train)
    ev = float(pipe.named_steps["pca"].explained_variance_ratio_.sum())
    return BudgetPlan(d, budget, "pca", ev, pipe)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 30))
    plan = fit_budget(X, budget=8)
    Z = plan.transform(X)
    assert Z.shape == (100, 8) and plan.method == "pca"
    assert Z.min() >= -1e-9 and Z.max() <= np.pi + 1e-9
    small = fit_budget(X[:, :5], budget=8)
    assert small.n_qubits == 5 and small.method == "scale"
    print(f"budgeter OK (pca kept {plan.explained_variance:.0%} variance)")
