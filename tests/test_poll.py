import requests, time
job_id = 'cd5781b7'
print(f"Polling {job_id}...")
for _ in range(60):
    res = requests.get(f'http://127.0.0.1:8000/api/jobs/{job_id}')
    job = res.json()
    print(f"Status: {job.get('status')}, Stage: {job.get('stage')}")
    if job.get('status') in ['COMPLETED', 'FAILED']:
        print(f"Final error (if any): {job.get('error')}")
        break
    time.sleep(2)
