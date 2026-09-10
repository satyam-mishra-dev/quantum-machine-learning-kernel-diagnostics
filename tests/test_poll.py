import pytest
pytestmark = pytest.mark.skip(reason="Manual script")

import requests, time

def test_poll():
    job_id = 'cd5781b7'
    print(f"Polling {job_id}...")
    for _ in range(60):
        try:
            res = requests.get(f'http://127.0.0.1:8000/api/jobs/{job_id}')
            job = res.json()
            print(f"Status: {job.get('status')}, Stage: {job.get('stage')}")
            if job.get('status') in ['COMPLETED', 'FAILED']:
                print(f"Final error (if any): {job.get('error')}")
                break
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(2)

if __name__ == "__main__":
    test_poll()
