import os
import json
import pytest
import io
import pandas as pd
from fastapi.testclient import TestClient

from qnidaan.app import app
from qnidaan.db import init_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    init_db()

def test_phase3_universal_ingestion():
    # 1. Test ambiguous target detection
    csv_content = b"feature1,outcome,label\n1,0,1\n2,1,0\n3,0,1\n4,1,0"
    file = io.BytesIO(csv_content)
    file.name = "test.csv"
    
    response = client.post(
        "/api/upload",
        data={"budget": 4},
        files={"file": ("test.csv", file, "text/csv")}
    )
    
    assert response.status_code == 422
    assert "target_ambiguous" in response.json()["detail"]
    assert "outcome" in response.json()["detail"]
    assert "label" in response.json()["detail"]

def test_phase3_auto_target_detection():
    # 2. Test successful auto-detection when only one binary column is named target/outcome
    csv_content = b"feature1,feature2,target\n1,10.5,0\n2,20.1,1\n3,30.2,0\n4,40.5,1\n5,50.1,0\n6,60.2,1"
    file = io.BytesIO(csv_content)
    file.name = "test2.csv"
    
    response = client.post(
        "/api/upload",
        data={"budget": 4},
        files={"file": ("test2.csv", file, "text/csv")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["profile"]["target_column"] == "target"

def test_phase3_qpu_execution_type():
    # 3. Test execution_type routing
    os.environ["IBM_QUANTUM_TOKEN"] = "mock_token"
    
    from qnidaan.quantum import _device
    import pennylane as qml
    
    dev = _device(2)
    # The device should fallback to lightning or default if pennylane-qiskit is not installed,
    # but the token check in worker.py should correctly flag it as REAL QPU EXECUTION
    
    from qnidaan.worker import run_adapter_worker
    
    # Clean up
    del os.environ["IBM_QUANTUM_TOKEN"]
