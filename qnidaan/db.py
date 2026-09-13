import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent.parent / "runs" / "jobs.db"

def get_conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            stage TEXT NOT NULL,
            dataset_meta TEXT,
            result TEXT,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS job_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT,
            stage TEXT,
            message TEXT,
            telemetry TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        )
    ''')
    try:
        conn.execute("ALTER TABLE job_logs ADD COLUMN telemetry TEXT")
    except sqlite3.OperationalError:
        pass # Column already exists
    conn.commit()
    conn.close()

def create_job(job_id: str, dataset_meta: dict):
    conn = get_conn()
    now = datetime.now().isoformat()
    conn.execute(
        "INSERT INTO jobs (id, status, stage, dataset_meta, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (job_id, "QUEUED", "QUEUED", json.dumps(dataset_meta), now, now)
    )
    conn.commit()
    conn.close()

def update_job(job_id: str, status: str = None, stage: str = None, result: dict = None, error: str = None, log_message: str = None, telemetry: dict = None):
    conn = get_conn()
    now = datetime.now().isoformat()
    
    fields = []
    values = []
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if stage is not None:
        fields.append("stage = ?")
        values.append(stage)
    if result is not None:
        fields.append("result = ?")
        values.append(json.dumps(result))
    if error is not None:
        fields.append("error = ?")
        values.append(error)
        
    if fields:
        fields.append("updated_at = ?")
        values.append(now)
        query = f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?"
        values.append(job_id)
        conn.execute(query, values)
        
    if log_message or stage or telemetry:
        if stage is None:
            cur_stage = conn.execute("SELECT stage FROM jobs WHERE id = ?", (job_id,)).fetchone()
            stage = cur_stage['stage'] if cur_stage else 'UNKNOWN'
        msg = log_message or f"Entered stage: {stage}"
        tel_str = json.dumps(telemetry) if telemetry else None
        conn.execute(
            "INSERT INTO job_logs (job_id, stage, message, telemetry, created_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, stage, msg, tel_str, now)
        )
        
    conn.commit()
    conn.close()

def get_job(job_id: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    if not row:
        return None
    
    return {
        "job_id": row["id"],
        "status": row["status"],
        "stage": row["stage"],
        "dataset": json.loads(row["dataset_meta"]) if row["dataset_meta"] else {},
        "result": json.loads(row["result"]) if row["result"] else None,
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"]
    }

def get_job_logs(job_id: str):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM job_logs WHERE job_id = ? ORDER BY id ASC", (job_id,)).fetchall()
    conn.close()
    return [{"stage": r["stage"], "message": r["message"], "created_at": r["created_at"], "telemetry": json.loads(r["telemetry"]) if ("telemetry" in r.keys() and r["telemetry"]) else None} for r in rows]

init_db()
