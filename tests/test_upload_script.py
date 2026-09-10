import time
import requests

def test_upload():
    # Submit breastmnist benchmark
    res = requests.post("http://127.0.0.1:8000/api/upload", data={"budget": 2, "benchmark": "breastmnist", "target": "target"})
    print("Upload response:", res.json())
    job_id = res.json()["job_id"]
    
    # Poll job endpoint
    while True:
        res = requests.get(f"http://127.0.0.1:8000/api/jobs/{job_id}")
        job = res.json()
        print(f"Status: {job['status']}, Stage: {job['stage']}")
        if job['status'] in ['COMPLETED', 'FAILED']:
            print("Error:", job.get('error'))
            break
        time.sleep(2)

if __name__ == "__main__":
    test_upload()
