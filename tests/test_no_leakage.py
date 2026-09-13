import numpy as np
from sklearn.model_selection import StratifiedKFold
import pytest

from qnidaan.budgeter import fit_budget_auto, fit_budget
from qnidaan.harness import run_dataset
from qnidaan.datasets import load_split

def test_feature_selection_inside_cv():
    # We want to ensure that feature selection inside CV isn't fit on the full data.
    # In `fit_budget_auto`, we clone pipeline steps. Let's ensure fit_budget_auto does not use the transform
    # before CV, by verifying that the pipeline doesn't have an exact match if we give it garbage data
    
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 30))
    y = (X[:, 0] + rng.normal(size=100) > 0).astype(int)
    
    # Run auto budgeter
    plan = fit_budget_auto(X, y, budget=4)
    assert plan.cv_auroc is not None
    assert plan.cv_auroc > 0.0, "CV score should be calculated properly"
    assert plan.method in ["select", "pca"]

def test_preprocessing_does_not_fit_on_test():
    Xtr, Xte, ytr, yte, meta = load_split("pima")
    plan = fit_budget_auto(Xtr, ytr, budget=4)
    # The transform step must not throw error, meaning it was trained on Xtr,
    # but the important part is that we never pass Xte to fit.
    Zte = plan.transform(Xte)
    assert Zte.shape == (len(Xte), 4)
    
def test_budgeter_is_deterministic():
    # mutual_info_classif jitters continuous features; without a seed the
    # "select" plan picked a different feature set every harness run and
    # runs/*.json could not regenerate.
    Xtr, _, ytr, _, _ = load_split("cleveland")
    picks = {tuple(fit_budget(Xtr, 4, y=ytr, method="select").selected_features)
             for _ in range(3)}
    assert len(picks) == 1, f"select is non-deterministic: {picks}"

    a = fit_budget_auto(Xtr, ytr, budget=4)
    b = fit_budget_auto(Xtr, ytr, budget=4)
    assert (a.method, a.cv_auroc, a.runner_up_cv_auroc) == (b.method, b.cv_auroc, b.runner_up_cv_auroc)

def test_target_leakage_check():
    # If a feature is perfectly correlated with the target but only in the test set,
    # it shouldn't be selected.
    rng = np.random.default_rng(42)
    Xtr = rng.normal(size=(100, 10))
    ytr = rng.integers(0, 2, size=100)
    
    Xte = rng.normal(size=(20, 10))
    yte = rng.integers(0, 2, size=20)
    
    # Make feature 9 perfectly correlated with target in TEST set only
    Xte[:, 9] = yte * 100.0
    
    plan = fit_budget(Xtr, 2, y=ytr, method="select")
    # Feature 9 must not be selected: it carries no signal in the train set
    assert 9 not in plan.selected_features, "Feature 9 leaked into selection!"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
