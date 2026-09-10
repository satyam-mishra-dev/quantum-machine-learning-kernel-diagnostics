import subprocess
import time
import requests
import json
import os
import sys

def run_tests():
    print("Starting FastAPI server...")
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "qnidaan.app:app", "--host", "127.0.0.1", "--port", "8001"], stdout=sys.stdout, stderr=sys.stderr)
    
    # Wait for startup
    started = False
    for _ in range(20):
        try:
            requests.get("http://127.0.0.1:8001/", timeout=1)
            started = True
            break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
            
    if not started:
        print("Failed to start FastAPI server")
        server.terminate()
        return
        
    try:
        print("\n--- TEST 01: Application Health ---")
        try:
            # We don't have a /api/health explicitly in app.py, let's try getting datasets
            res = requests.get("http://127.0.0.1:8001/api/datasets")
            assert res.status_code == 200, "Health check failed"
            print("PASS")
        except Exception as e:
            print("FAIL:", e)
            
        print("\n--- TEST 02: Basic CSV Ingestion ---")
        sample_path = os.path.join(os.path.dirname(__file__), "..", "web", "public", "samples", "heart_failure.csv")
        with open(sample_path, "rb") as f:
            csv_content = f.read()
            
        files = {'file': ('heart_failure.csv', csv_content, 'text/csv')}
        data = {'target': 'DEATH_EVENT', 'budget': '4'}
        start_t = time.time()
        res = requests.post("http://127.0.0.1:8001/api/upload", files=files, data=data)
        elapsed = time.time() - start_t
        assert res.status_code == 200, f"Upload failed: {res.text}"
        job_id = res.json()["job_id"]
        assert elapsed < 1.0, f"Upload took too long: {elapsed}s"
        print(f"PASS (Took {elapsed:.2f}s, Job ID: {job_id})")
        
        print("\n--- TEST 03 & 04: Real-time Autopilot & Non-blocking API ---")
        done = False
        
        # Test non-blocking while job is queued/running
        res_health = requests.get("http://127.0.0.1:8001/api/datasets")
        assert res_health.status_code == 200, "API is blocking!"
        
        print(f"Connecting to SSE for job {job_id}...")
        response = requests.get(f"http://127.0.0.1:8001/api/jobs/{job_id}/stream", stream=True)
        
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith('data: '):
                    data = json.loads(decoded[6:])
                    print("SSE Event:", data)
                    if "status" in data and data["status"] in ["COMPLETED", "FAILED", "error", "done"]:
                        print("Job finished with status:", data["status"])
                        done = True
                        break
                    
                    # Check non-blocking again during execution
                    health_check = requests.get("http://127.0.0.1:8001/api/datasets")
                    assert health_check.status_code == 200, "API blocked during SSE event"

        print("PASS")

        print("\n--- TEST 19: Job Persistence ---")
        res = requests.get(f"http://127.0.0.1:8001/api/jobs/{job_id}")
        assert res.status_code == 200
        job_data = res.json()
        assert job_data["status"] == "COMPLETED"
        print("PASS")

    finally:
        server.terminate()
        server.wait()

if __name__ == "__main__":
    run_tests()
