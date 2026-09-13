import pytest
pytestmark = pytest.mark.skip(reason="Manual integration test requiring local server and heart.csv")

import time
import requests

def test_upload():
    # Submit heart.csv
    print("Uploading heart.csv...")
    start_time = time.time()
    try:
        with open("heart.csv", "rb") as f:
            res = requests.post("http://127.0.0.1:8000/api/upload", files={"file": ("heart.csv", f)}, data={"budget": 2, "target": "target"})
    except FileNotFoundError:
        print("heart.csv not found")
        return
    
    upload_time = time.time() - start_time
    print(f"Upload response: {res.json()}")
    print(f"Upload time: {upload_time:.2f}s")
    
    job_id = res.json()["job_id"]
    
    print(f"Polling job {job_id}...")
    # Poll job endpoint
    for _ in range(60): # 2 minute timeout
        try:
            res = requests.get(f"http://127.0.0.1:8000/api/jobs/{job_id}")
            job = res.json()
            print(f"Status: {job.get('status')}, Stage: {job.get('stage')}")
            if job.get('status') in ['COMPLETED', 'FAILED']:
                print(f"Final error: {job.get('error')}")
                break
        except Exception as e:
            print(f"Error polling: {e}")
        time.sleep(2)
        
    print(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    test_upload()
