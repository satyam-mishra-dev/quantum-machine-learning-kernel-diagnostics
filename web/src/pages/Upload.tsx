import { useEffect, useRef, useState, type ReactNode } from 'react';
import clsx from 'clsx';
import gsap from 'gsap';
import { api, MODEL_LABELS, isQuantum, type JobStatus } from '../lib/api';
import { Badge, Button, Card, ColdStartNote, ErrorState, Eyebrow, KV, PageTitle, ReadinessBreakdown } from '../components/ui';

/** Shipped in web/public/samples/. Every one is verified to parse through
    qnidaan.datasets.load_csv with the target column named here. Cleveland and
    Pima are deliberately absent: those CSVs are headerless, so the adapter
    has no column name to target. */
const SAMPLES = [
  { file: 'parkinsons.csv', title: "Parkinson's (UCI voice)", target: 'status', rows: 195, feats: 22 },
  { file: 'ilpd.csv', title: 'Indian Liver Patient (ILPD)', target: 'target', rows: 579, feats: 10 },
  { file: 'heart_failure.csv', title: 'Heart failure clinical records', target: 'DEATH_EVENT', rows: 299, feats: 12 },
];

const STAGES = ['Ingest', 'Qubit budget', 'Advantage scout', 'Train quantum + classical twins', 'Verdict'];

const STAGE_INDEX: Record<string, number> = { ingest: 0, qubit_budget: 1, advantage_scout: 2, verdict: 4 };
const stageIndex = (s?: string): number | null =>
  s == null ? null : s.startsWith('training:') ? 3 : STAGE_INDEX[s] ?? null;

/** The polled `stage` field drives the highlight; if the backend doesn't
    report one we fall back to marching on a timer. */
function Pipeline({ done, stage: serverStage }: { done: boolean; stage?: string }): ReactNode {
  const [stage, setStage] = useState(0);
  const fromServer = stageIndex(serverStage);
  const dotRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (done || fromServer != null) return;
    const t = setInterval(() => setStage((s) => Math.min(s + 1, STAGES.length - 2)), 3500);
    return () => clearInterval(t);
  }, [done, fromServer]);
  useEffect(() => {
    const el = dotRef.current;
    if (!el || done) return;
    const tween = gsap.to(el, { opacity: 0.25, duration: 0.6, yoyo: true, repeat: -1, ease: 'sine.inOut' });
    return () => { tween.kill(); };
  }, [done, stage]);
  const current = done ? STAGES.length : fromServer ?? stage;
  return (
    <ol className="mt-4 space-y-0">
      {STAGES.map((s, i) => {
        const state = i < current ? 'done' : i === current ? 'active' : 'pending';
        return (
          <li key={s} className="flex items-center gap-3 py-2">
            <span className="flex w-5 justify-center">
              {state === 'done' ? (
                <span className="font-mono text-[13px] text-credit">✓</span>
              ) : state === 'active' ? (
                <span ref={dotRef} className="inline-block h-2.5 w-2.5 rounded-full bg-quantum" />
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
              {s}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function VerdictCard({ result }: { result: NonNullable<JobStatus['result']> }): ReactNode {
  const { verdict, heldout, budget, scout } = result;
  const delta = verdict.quantum_vs_best_classical ?? verdict.qsvm_vs_best_classical;
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const tween = gsap.fromTo(el, { autoAlpha: 0, y: 20, scale: 0.98 }, { autoAlpha: 1, y: 0, scale: 1, duration: 0.7, ease: 'power3.out' });
    return () => { tween.kill(); };
  }, []);
  return (
    <div ref={ref} className="q-gradient-border mt-6 p-6 sm:p-8">
      <Eyebrow className="text-quantum">The verdict — published either way</Eyebrow>
      <div className="mt-3 font-serif text-2xl font-semibold text-ink sm:text-3xl">
        {verdict.quantum_won ? (
          <>Quantum <span className="q-gradient-text">won</span> by {Math.abs(delta).toFixed(4)} AUROC</>
        ) : (
          <>Quantum lost by {Math.abs(delta).toFixed(4)} AUROC</>
        )}
      </div>
      <p className="mt-2 text-[13px] text-ink-60">
        Best model: <span className="font-medium text-ink">{MODEL_LABELS[verdict.best_model] ?? verdict.best_model}</span>
        {' '}· {budget.n_qubits} qubits ({budget.method})
        {' '}· scout said {scout.quantum_worth_trying ? 'quantum worth trying' : 'no quantum edge expected'}
      </p>
      {/* The upload adapter only accepts numeric tabular CSVs, so a finished
          run is by construction tabular — confirm it, don't pretend to infer more. */}
      <p className="mt-1 font-mono text-[11px] text-ink-60">
        Detected modality: Tabular (numeric CSV)
      </p>
      {result.train_rows_capped && (
        <p className="mt-1 font-mono text-[11px] text-hold">
          trained on a {result.train_rows_used ?? 400}-row subsample to keep the quantum kernel tractable
        </p>
      )}
      {result.readiness && (
        <div className="mt-5 border-t border-rule pt-5">
          <Eyebrow>Quantum readiness</Eyebrow>
          <div className="mt-3 max-w-sm">
            <ReadinessBreakdown r={result.readiness} />
          </div>
        </div>
      )}
      <div className="mt-5 grid gap-x-10 sm:grid-cols-2">
        {Object.entries(heldout).map(([m, e]) => (
          <KV
            key={m}
            k={<span>{MODEL_LABELS[m] ?? m} {isQuantum(m) && <Badge tone="quantum" className="ml-1">quantum</Badge>}</span>}
            v={`AUROC ${e.auroc.toFixed(3)}`}
          />
        ))}
      </div>
    </div>
  );
}

export function Upload(): ReactNode {
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState('');
  const [budget, setBudget] = useState(8);
  const [dragOver, setDragOver] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Poll the job every 2s until done/error.
  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    const tick = async (): Promise<void> => {
      try {
        const j = await api<JobStatus>(`/api/jobs/${jobId}`);
        if (!alive) return;
        setJob(j);
        if (j.status === 'running') setTimeout(() => void tick(), 2000);
      } catch (e) {
        if (alive) setError((e as Error).message);
      }
    };
    void tick();
    return () => { alive = false; };
  }, [jobId]);

  /** Pull a shipped sample into the same File the drop zone would produce,
      and fill in its target column so the run is one click away. */
  const useSample = async (s: (typeof SAMPLES)[number]): Promise<void> => {
    const res = await fetch(`/samples/${s.file}`);
    if (!res.ok) return;
    const blob = await res.blob();
    setFile(new File([blob], s.file, { type: 'text/csv' }));
    setTarget(s.target);
  };

  const submit = async (): Promise<void> => {
    if (!file || !target) {
      setError('Choose a CSV and name its target column.');
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setError(`CSV is ${(file.size / 1048576).toFixed(1)} MB — uploads are capped at 5 MB.`);
      return;
    }
    setError(null);
    setSubmitting(true);
    setJob(null);
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('target', target);
      form.append('budget', String(budget));
      const res = await api<{ job_id: string }>('/api/upload', { method: 'POST', body: form });
      setJobId(res.job_id);
    } catch (e) {
      const msg = (e as Error).message;
      setError(/413|too large/i.test(msg) ? 'File too large — uploads are capped at 5 MB.' : msg);
    } finally {
      setSubmitting(false);
    }
  };

  const running = job?.status === 'running' || (jobId != null && job == null && !error);

  return (
    <div>
      <Eyebrow>Universal adapter</Eyebrow>
      <PageTitle>Upload your own dataset</PageTitle>
      <p className="mt-2 max-w-2xl text-[14px] text-ink-60">
        Any CSV with a binary target column runs through the exact pipeline the flagships went
        through — qubit budgeting, the advantage scout, quantum models and their classical twins —
        and gets the same honest verdict.
      </p>

      <Card className="mt-6 p-5">
        <div className="mb-4">
          <span className="eyebrow">No dataset handy? Start from a sample</span>
          <div className="mt-2 grid gap-2 sm:grid-cols-3">
            {SAMPLES.map((s) => (
              <div key={s.file} className="rounded-sm border border-rule bg-wash/40 p-3">
                <div className="text-[13px] font-medium leading-snug text-ink">{s.title}</div>
                <div className="mt-1 text-[11px] text-ink-60 tnum">
                  {s.rows} rows &middot; {s.feats} features
                </div>
                <div className="mt-1 font-mono text-[11px] text-quantum">target: {s.target}</div>
                <div className="mt-2 flex items-center gap-3">
                  <button
                    type="button"
                    className="link text-[12px]"
                    onClick={() => void useSample(s)}
                  >
                    Use this
                  </button>
                  <a className="link text-[12px]" href={`/samples/${s.file}`} download>
                    Download
                  </a>
                </div>
              </div>
            ))}
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
            {file ? file.name : 'Drag a CSV here'}
          </div>
          <div className="text-[12px] text-ink-60">
            {file ? `${(file.size / 1024).toFixed(1)} KB` : 'or'}
          </div>
          <label className="cursor-pointer">
            <span className="link text-[13px]">browse files</span>
            <input
              type="file"
              accept=".csv,text/csv"
              className="sr-only"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>
        </div>

        <div className="mt-4 flex flex-wrap items-end gap-4">
          <label className="block">
            <span className="eyebrow">Target column</span>
            <input
              className="mt-1.5 block h-9 w-48 rounded-sm border border-rule bg-paper px-2 font-mono text-[13px]"
              placeholder="e.g. outcome"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
            />
          </label>
          <label className="block">
            <span className="eyebrow">Qubit budget</span>
            <input
              type="number"
              min={1}
              max={10}
              className="mt-1.5 block h-9 w-24 rounded-sm border border-rule bg-paper px-2 font-mono text-[13px] tnum"
              value={budget}
              onChange={(e) => setBudget(Math.min(10, Math.max(1, parseInt(e.target.value || '8', 10))))}
            />
          </label>
          <Button variant="quantum" onClick={() => void submit()} disabled={submitting || running}>
            {submitting ? 'Uploading…' : 'Run the pipeline'}
          </Button>
        </div>

        <ColdStartNote>
          This trains a quantum kernel live on Render's free tier (0.1 CPU), so
          a run takes 10-15 minutes — and if the backend has been idle, add
          about a minute for it to wake up first.
        </ColdStartNote>

        {error && (
          <div className="mt-4">
            <ErrorState>{error}</ErrorState>
          </div>
        )}
      </Card>

      {(running || job) && !error && (
        <Card className="mt-6 p-5">
          <div className="flex items-center justify-between">
            <Eyebrow>Pipeline{jobId ? ` · job ${jobId}` : ''}</Eyebrow>
            {job?.status === 'done' && <Badge tone="credit">done</Badge>}
            {job?.status === 'error' && <Badge tone="debit">error</Badge>}
            {running && <Badge tone="hold">running</Badge>}
          </div>
          <Pipeline done={job?.status === 'done'} stage={job?.stage} />
          {job?.status === 'error' && (
            <div className="mt-3">
              <ErrorState>Pipeline failed: {job.error}</ErrorState>
            </div>
          )}
        </Card>
      )}

      {job?.status === 'done' && job.result && <VerdictCard result={job.result} />}
    </div>
  );
}
