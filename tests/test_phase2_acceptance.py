import sys
import os
import pytest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qnidaan.datasets import load_split
from qnidaan.budgeter import fit_budget
from qnidaan.classical import evaluate, make_twins
from qnidaan.worker import run_adapter_worker
from qnidaan.db import create_job, get_job

import uuid
def test_phase2_schema():
    # Load dataset
    Xtr, Xte, ytr, yte, meta = load_split("wdbc")
    job_id = f"test_phase2_{uuid.uuid4().hex[:8]}"
    create_job(job_id, meta)
    
    # Run pipeline synchronously
    success = run_adapter_worker(job_id, Xtr, ytr, meta, budget=4)
    assert success, "Worker failed to run"
    
    # Verify job status
    job = get_job(job_id)
    assert job["status"] == "COMPLETED"
    
    result = job["result"]
    
    # Step 10: Evaluation Engine
    for model_name, metrics in result["heldout"].items():
        assert "auroc" in metrics
        assert "auprc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics
        assert "roc_curve" in metrics
        assert "pr_curve" in metrics
        
    # Step 11: Quantum vs Classical Analytics
    assert "full_feature_classical" in result["heldout"]
    assert "full_feature_floor_auroc" in result["verdict"]
    
    # Step 12: Uncertainty + Calibration
    assert "uncertainty" in result
    assert "qsvm" in result["uncertainty"]
    assert "prediction_sets" in result["uncertainty"]["qsvm"]
    
    # Step 13: Explainability
    assert "explainability" in result
    if result["explainability"].get("status") != "not available":
        assert result["explainability"]["type"] == "Surrogate Classical Explainability"
        assert "top_components" in result["explainability"]
    
    # Step 14: Circuit Lab
    assert "circuit" in result
    assert "depth" in result["circuit"]
    assert "num_operations" in result["circuit"]
    assert "drawing" in result["circuit"]
    
    # Artifact generation
    run_dir = Path(__file__).resolve().parent.parent / "runs" / f"QF-{job_id}"
    assert (run_dir / "report.html").exists(), "HTML report not generated"
    assert (run_dir / "metrics.json").exists()
    assert (run_dir.parent / f"QuantumFloor_{job_id}.zip").exists()
    
if __name__ == "__main__":
    pytest.main([__file__])
