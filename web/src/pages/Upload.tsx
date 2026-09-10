import { useEffect, useRef, useState, type ReactNode } from 'react';
import clsx from 'clsx';
import gsap from 'gsap';
import { api, MODEL_LABELS, isQuantum, type JobStatus } from '../lib/api';
import { Badge, Button, Card, ColdStartNote, ErrorState, Eyebrow, KV, PageTitle, ReadinessBreakdown } from '../components/ui';
import { CircuitLab } from '../components/CircuitLab';

/** Shipped in web/public/samples/. Every one is verified to parse through
    qnidaan.datasets.load_csv with the target column named here. Cleveland and
    Pima are deliberately absent: those CSVs are headerless, so the adapter
    has no column name to target. */
const SAMPLES = [
  { file: 'parkinsons.csv', title: "Parkinson's (UCI voice)", target: 'status', rows: 195, feats: 22 },
  { file: 'ilpd.csv', title: 'Indian Liver Patient (ILPD)', target: 'target', rows: 579, feats: 10 },
  { file: 'heart_failure.csv', title: 'Heart failure clinical records', target: 'DEATH_EVENT', rows: 299, feats: 12 },
];

const STAGES = [
  { id: 'QUEUED', label: 'Job queued' },
  { id: 'INGESTING', label: 'Dataset loaded' },
  { id: 'SPLITTING', label: 'Train/test split' },
  { id: 'PREPROCESSING', label: 'Preprocessing' },
  { id: 'BUDGETING', label: 'Qubit budget selection' },
  { id: 'SCOUTING', label: 'Quantum geometry scout' },
  { id: 'PLANNING', label: 'Execution planning' },
  { id: 'TRAINING_CLASSICAL', label: 'Classical twins' },
  { id: 'TRAINING_QUANTUM', label: 'Quantum kernel' },
  { id: 'EXPLAINING', label: 'Explainability & SHAP' },
  { id: 'EVALUATING', label: 'Evaluation & verdict' },
  { id: 'COMPLETED', label: 'Completed' }
];

function DatasetIntelligence({ profile }: { profile: any }): ReactNode {
  if (!profile) return null;
  return (
    <Card className="mt-6 p-5 bg-paper">
      <Eyebrow>Dataset Intelligence</Eyebrow>
      
      <div className="mt-4 p-4 border border-quantum/30 bg-quantum-wash/30 rounded-sm">
        <div className="flex justify-between items-start">
            <div>
              <div className="eyebrow text-quantum">Modality Detector</div>
              <div className="mt-1 font-serif text-xl text-ink font-medium">{profile.modality}</div>
            </div>
            <Badge tone="quantum">Confidence: {profile.modality_confidence}</Badge>
        </div>
        <p className="mt-2 text-[13px] text-ink-60">{profile.modality_reasoning}</p>
      </div>

      <div className="mt-5 grid grid-cols-3 gap-6">
        <div>
          <div className="eyebrow">Rows</div>
          <div className="mt-1 font-mono text-xl tnum text-ink">{profile.rows?.toLocaleString()}</div>
        </div>
        <div>
          <div className="eyebrow">Columns</div>
          <div className="mt-1 font-mono text-xl tnum text-ink">{profile.columns}</div>
        </div>
        <div>
          <div className="eyebrow">Target</div>
          <div className="mt-1 font-mono text-xl text-ink">{profile.target_column}</div>
        </div>
        <div>
          <div className="eyebrow">Missing Values</div>
          <div className="mt-1 font-mono text-xl text-ink">{profile.missing_pct?.toFixed(1)}%</div>
        </div>
        <div>
          <div className="eyebrow">Classes</div>
          <div className="mt-1 font-mono text-xl text-ink">{profile.classes}</div>
        </div>
        <div>
          <div className="eyebrow">Computational Cost</div>
          <div className="mt-1 font-mono text-xl text-ink">{profile.computational_cost_estimate}</div>
        </div>
      </div>
      <div className="mt-6 border-t border-rule pt-4 grid grid-cols-3 gap-6">
         <div>
          <div className="eyebrow">Features</div>
          <div className="text-[13px] text-ink-60 mt-1">
            {profile.numerical_features} numerical / {profile.categorical_features} categorical
          </div>
         </div>
         <div>
          <div className="eyebrow">Class Imbalance</div>
          <div className="text-[13px] text-ink-60 mt-1">
            Ratio {profile.imbalance_ratio?.toFixed(2)}:1
          </div>
         </div>
         <div>
          <div className="eyebrow">Leakage Audit</div>
          <div className="text-[13px] text-ink-60 mt-1">
            {profile.identifier_columns > 0 ? (
                <span className="text-debit">Dropped {profile.identifier_columns} identifiers: {profile.leakage_risk_columns.join(', ')}</span>
            ) : (
                <span className="text-credit">Pass (No obvious ID leaks)</span>
            )}
          </div>
         </div>
      </div>
    </Card>
  );
}

function Pipeline({ done, stage, logs = [] }: { done: boolean; stage?: string; logs?: any[] }): ReactNode {
  const currentIdx = STAGES.findIndex(s => s.id === stage);
  
  return (
    <div className="mt-4 flex gap-8">
      <ol className="space-y-0 w-1/3 border-r border-rule pr-4">
        {STAGES.map((s, i) => {
          const state = done || (currentIdx > -1 && i < currentIdx) ? 'done' : i === currentIdx ? 'active' : 'pending';
          return (
            <li key={s.id} className="flex items-center gap-3 py-2">
              <span className="flex w-5 justify-center">
                {state === 'done' ? (
                  <span className="font-mono text-[13px] text-credit">✓</span>
                ) : state === 'active' ? (
                  <span className="inline-block h-2.5 w-2.5 rounded-full bg-quantum animate-pulse" />
                ) : (
                  <span className="inline-block h-2 w-2 rounded-full border border-rule" />
                )}
              </span>
              <span
                className={clsx(
                  'text-[13px]',
                  state === 'done' && 'text-ink-60',
                  state === 'active' && 'font-medium text-ink',
                  state === 'pending' && 'text-ink-45',
                )}
              >
                {s.label}
              </span>
            </li>
          );
        })}
      </ol>
      
      <div className="w-2/3 h-64 flex flex-col justify-between bg-wash/40 p-4 rounded-sm border border-rule">
        <div className="font-mono text-[11px] text-ink-60 flex flex-col-reverse overflow-y-auto h-full">
          {logs.map((l, i) => (
            <div key={i} className="mb-1">
              <span className="text-ink-45">[{new Date(l.created_at).toLocaleTimeString()}]</span> {l.message}
            </div>
          ))}
        </div>
        {logs[0]?.telemetry && (
          <div className="mt-3 pt-3 border-t border-rule shrink-0">
             <div className="flex justify-between text-[11px] font-mono text-quantum mb-1.5">
                <span>{logs[0].telemetry.substage || logs[0].telemetry.model} Execution</span>
                <span>{logs[0].telemetry.percentage?.toFixed(1) || 0}%</span>
             </div>
             <div className="w-full bg-rule/50 h-1.5 rounded-full overflow-hidden">
                 <div className="bg-quantum h-full transition-all duration-300" style={{ width: `${Math.min(100, logs[0].telemetry.percentage || 0)}%` }} />
             </div>
             <div className="flex justify-between text-[10px] text-ink-60 mt-1.5 font-mono">
                <span>{logs[0].telemetry.completed || 0} / {logs[0].telemetry.total || 0} samples</span>
                {logs[0].telemetry.eta_seconds != null && <span>ETA {Math.ceil(logs[0].telemetry.eta_seconds)}s ({logs[0].telemetry.rate?.toFixed(1)}/s)</span>}
             </div>
          </div>
        )}
      </div>
    </div>
  );
}

import { ResearchConsole } from '../components/ResearchConsole';

export function Upload(): ReactNode {
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState('');
  const [budget, setBudget] = useState(8);
  const [dragOver, setDragOver] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [profile, setProfile] = useState<any>(null);
  const [logs, setLogs] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Server-Sent Events subscription
  useEffect(() => {
    if (!jobId) return;
    
    const eventSource = new EventSource(`/api/jobs/${jobId}/stream`);
    
    eventSource.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.status) {
        setJob(data);
        if (data.status === 'COMPLETED' || data.status === 'FAILED') {
          eventSource.close();
          // Fetch final job payload
          api<JobStatus>(`/api/jobs/${jobId}`).then(setJob);
        }
      } else if (data.message) {
        setLogs(prev => {
          // Keep last 500 logs to prevent memory leak on long runs
          const newLogs = [data, ...prev];
          return newLogs.length > 500 ? newLogs.slice(0, 500) : newLogs;
        });
        setJob(prev => prev ? ({ ...prev, stage: data.stage } as JobStatus) : { status: 'RUNNING', stage: data.stage } as JobStatus);
      }
    };
    
    eventSource.onerror = () => {
      eventSource.close();
      setJob(prev => {
        // If it was already completed/failed, just ignore the error
        if (prev && (prev.status === 'COMPLETED' || prev.status === 'FAILED')) return prev;
        return prev ? ({ ...prev, status: 'FAILED', error: 'Connection lost' } as JobStatus) : { status: 'FAILED', error: 'Connection lost' } as JobStatus;
      });
      setError('Connection lost to backend job tracker. Status may be stale.');
    };

    return () => { eventSource.close(); };
  }, [jobId]);

  const useSample = async (s: (typeof SAMPLES)[number]): Promise<void> => {
    const res = await fetch(`/samples/${s.file}`);
    if (!res.ok) return;
    const blob = await res.blob();
    setFile(new File([blob], s.file, { type: 'text/csv' }));
    setTarget(s.target);
  };

  const submit = async (): Promise<void> => {
    if (!file) {
      setError('Choose a CSV.');
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setError(`CSV is ${(file.size / 1048576).toFixed(1)} MB — uploads are capped at 5 MB.`);
      return;
    }
    setError(null);
    setSubmitting(true);
    setJob(null);
    setLogs([]);
    setProfile(null);
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('target', target);
      form.append('budget', String(budget));
      const res = await api<any>('/api/upload', { method: 'POST', body: form });
      setJobId(res.job_id);
      setProfile(res.profile);
      setJob({ status: 'RUNNING', stage: 'QUEUED' } as JobStatus);
    } catch (e) {
      const msg = (e as Error).message;
      try {
        if (msg.includes('target_ambiguous')) {
          const parsed = JSON.parse(msg.replace(/'/g, '"'));
          setError(`Target ambiguous. Please specify one: ${parsed.candidates.join(', ')}`);
          return;
        }
      } catch (err) {}
      setError(/413|too large/i.test(msg) ? 'File too large — uploads are capped at 5 MB.' : msg);
    } finally {
      setSubmitting(false);
    }
  };

  const submitBenchmark = async (benchmarkName: string): Promise<void> => {
    setError(null);
    setSubmitting(true);
    setJob(null);
    setLogs([]);
    setProfile(null);
    try {
      const form = new FormData();
      form.append('benchmark', benchmarkName);
      form.append('target', 'target');
      form.append('budget', String(budget));
      const res = await api<any>('/api/upload', { method: 'POST', body: form });
      setJobId(res.job_id);
      setProfile(res.profile);
      setJob({ status: 'RUNNING', stage: 'QUEUED' } as JobStatus);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const running = job?.status === 'RUNNING' || job?.status === 'QUEUED' || (jobId != null && job == null && !error);

  return (
    <div>
      <Eyebrow>Universal adapter</Eyebrow>
      <PageTitle>Computational Research Console</PageTitle>
      <p className="mt-2 max-w-2xl text-[14px] text-ink-60">
        Upload a dataset. QuantumFloor automatically profiles, validates, and evaluates it across a rigorous quantum vs matched-classical benchmark.
      </p>

      {!running && job?.status !== 'COMPLETED' && (
      <Card className="mt-6 p-5">
        <div className="mb-4">
          <span className="eyebrow">No dataset handy? Start from a tabular sample</span>
          <div className="mt-2 grid gap-2 sm:grid-cols-3">
            {SAMPLES.map((s) => (
              <div key={s.file} className="rounded-sm border border-rule bg-wash/40 p-3">
                <div className="text-[13px] font-medium leading-snug text-ink">{s.title}</div>
                <div className="mt-1 text-[11px] text-ink-60 tnum">
                  {s.rows} rows &middot; {s.feats} features
                </div>
                <div className="mt-1 font-mono text-[11px] text-quantum">target: {s.target}</div>
                <div className="mt-2 flex items-center gap-3">
                  <button type="button" className="link text-[12px]" onClick={() => void useSample(s)}>Use this</button>
                  <a className="link text-[12px]" href={`/samples/${s.file}`} download>Download</a>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="mb-6 pt-4 border-t border-rule">
          <span className="eyebrow text-quantum">Image Benchmark Track (MedMNIST)</span>
          <div className="mt-2 grid gap-2 sm:grid-cols-2">
              <div className="rounded-sm border border-quantum/30 bg-quantum-wash/30 p-3">
                <div className="text-[13px] font-medium leading-snug text-ink">PneumoniaMNIST</div>
                <div className="mt-1 text-[11px] text-ink-60">Chest X-Rays · 5,856 images</div>
                <div className="mt-2 flex items-center gap-3">
                  <button type="button" className="link text-[12px] text-quantum" onClick={() => submitBenchmark('pneumoniamnist')}>Run Pipeline</button>
                </div>
              </div>
              <div className="rounded-sm border border-quantum/30 bg-quantum-wash/30 p-3">
                <div className="text-[13px] font-medium leading-snug text-ink">BreastMNIST</div>
                <div className="mt-1 text-[11px] text-ink-60">Breast Ultrasound · 780 images</div>
                <div className="mt-2 flex items-center gap-3">
                  <button type="button" className="link text-[12px] text-quantum" onClick={() => submitBenchmark('breastmnist')}>Run Pipeline</button>
                </div>
              </div>
          </div>
        </div>

        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const f = e.dataTransfer.files[0];
            if (f) setFile(f);
          }}
          className={clsx(
            'flex flex-col items-center justify-center gap-2 rounded-sm border border-dashed px-6 py-10 text-center transition-colors',
            dragOver ? 'border-quantum bg-quantum-wash' : 'border-rule bg-wash/40',
          )}
        >
          <div className="text-sm font-medium text-ink">
            {file ? file.name : 'Drop Dataset Here'}
          </div>
          <div className="text-[12px] text-ink-60">
            {file ? `${(file.size / 1024).toFixed(1)} KB` : 'CSV / Supported Dataset'}
          </div>
          <label className="cursor-pointer">
            <span className="link text-[13px]">browse files</span>
            <input type="file" accept=".csv,text/csv,.zip,application/zip" className="sr-only" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>
        </div>

        <div className="mt-4 flex flex-wrap items-end gap-4">
          <label className="block">
            <span className="eyebrow">Target column</span>
            <input
              className="mt-1.5 block h-9 w-48 rounded-sm border border-rule bg-paper px-2 font-mono text-[13px]"
              placeholder="(auto-detect)"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
            />
          </label>
          <label className="block">
            <span className="eyebrow">Qubit budget</span>
            <input
              type="number" min={1} max={10}
              className="mt-1.5 block h-9 w-24 rounded-sm border border-rule bg-paper px-2 font-mono text-[13px] tnum"
              value={budget}
              onChange={(e) => setBudget(Math.min(10, Math.max(1, parseInt(e.target.value || '8', 10))))}
            />
          </label>
          <Button variant="quantum" onClick={() => void submit()} disabled={submitting || running}>
            {submitting ? 'Validating…' : 'Run QuantumFloor Analysis'}
          </Button>
        </div>

        {error && (
          <div className="mt-4">
            <ErrorState>{error}</ErrorState>
          </div>
        )}
      </Card>
      )}

      {profile && <DatasetIntelligence profile={profile} />}

      {(running || job) && !error && (
        <Card className="mt-6 p-5">
          <div className="flex items-center justify-between">
            <Eyebrow>Autopilot Engine{jobId ? ` · job ${jobId}` : ''}</Eyebrow>
            {job?.status === 'COMPLETED' && <Badge tone="credit">done</Badge>}
            {job?.status === 'FAILED' && <Badge tone="debit">error</Badge>}
            {(job?.status === 'RUNNING' || job?.status === 'QUEUED') && <Badge tone="hold">running</Badge>}
          </div>
          <Pipeline done={job?.status === 'COMPLETED'} stage={job?.stage} logs={logs} />
          {job?.status === 'FAILED' && (
            <div className="mt-3">
              <ErrorState>Pipeline failed: {job.error}</ErrorState>
            </div>
          )}
          
          {job?.result?.circuit && job?.status !== 'FAILED' && (
             <div className="mt-4">
                 <CircuitLab circuit={job.result.circuit} stage={job.stage} telemetry={logs[0]?.telemetry} />
             </div>
          )}
        </Card>
      )}

      {job?.status === 'COMPLETED' && job.result && <ResearchConsole result={job.result} jobId={jobId} profile={job.profile} />}
    </div>
  );
}
