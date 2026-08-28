import { useEffect, useRef, useState, type ReactNode } from 'react';
import gsap from 'gsap';
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import {
  api, MODEL_LABELS, useApi, type DatasetCard, type Prediction,
} from '../lib/api';
import { AXIS_TICK, SHAP_DOWN, SHAP_UP, TOOLTIP_STYLE } from '../lib/chart';
import { SAMPLE_PATIENTS } from '../lib/samples';
import { Badge, Button, Card, ErrorState, Eyebrow, PageTitle, Skeleton , ColdStartNote } from '../components/ui';

const MODELS = ['qsvm', 'vqc', 'logreg', 'svm_rbf', 'random_forest'];

// ---------------------------------------------------------------- Risk gauge
function RiskGauge({ risk }: { risk: number }): ReactNode {
  const arcRef = useRef<SVGPathElement>(null);
  const numRef = useRef<HTMLDivElement>(null);
  // Semicircle r=80, length = PI * r
  const LEN = Math.PI * 80;
  const color = risk < 0.33 ? 'var(--color-credit)' : risk < 0.66 ? 'var(--color-hold)' : 'var(--color-debit)';
  useEffect(() => {
    const arc = arcRef.current;
    const num = numRef.current;
    if (!arc || !num) return;
    const obj = { v: 0 };
    gsap.set(arc, { strokeDasharray: LEN, strokeDashoffset: LEN });
    const tl = gsap.timeline();
    tl.to(arc, { strokeDashoffset: LEN * (1 - risk), duration: 1.3, ease: 'power2.out' }, 0);
    tl.to(obj, {
      v: risk, duration: 1.3, ease: 'power2.out',
      onUpdate: () => { num.textContent = `${(obj.v * 100).toFixed(1)}%`; },
    }, 0);
    return () => { tl.kill(); };
  }, [risk, LEN]);
  return (
    <div className="relative mx-auto w-[220px]">
      <svg viewBox="0 0 200 110" className="w-full">
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="var(--color-rule)" strokeWidth="12" strokeLinecap="round" />
        <path ref={arcRef} d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke={color} strokeWidth="12" strokeLinecap="round" />
      </svg>
      <div className="absolute inset-x-0 bottom-0 text-center">
        <div ref={numRef} className="font-mono text-[30px] leading-none text-ink tnum">0.0%</div>
        <div className="eyebrow mt-1">risk probability</div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- SHAP chart
function ShapBars({ features }: { features: Prediction['top_features'] }): ReactNode {
  const rows = [...features].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
  return (
    <div style={{ height: Math.max(140, rows.length * 34) }}>
      <ResponsiveContainer>
        <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 24, left: 8, bottom: 0 }}>
          <CartesianGrid horizontal={false} stroke="var(--color-rule)" />
          <XAxis type="number" tick={AXIS_TICK} tickLine={false} axisLine={{ stroke: 'var(--color-rule)' }} />
          <YAxis type="category" dataKey="feature" width={130} tick={{ ...AXIS_TICK, fill: 'var(--color-ink)' }} tickLine={false} axisLine={false} />
          <Tooltip
            cursor={{ fill: 'var(--color-wash)' }}
            contentStyle={TOOLTIP_STYLE}
            formatter={(v) => [Number(v).toFixed(4), 'SHAP']}
          />
          <Bar dataKey="shap" radius={[0, 4, 4, 0]} isAnimationActive animationDuration={800}>
            {rows.map((r) => (
              <Cell key={r.feature} fill={r.shap >= 0 ? SHAP_UP : SHAP_DOWN} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-1 flex items-center gap-4 font-mono text-[11px] text-ink-60">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-[2px]" style={{ background: SHAP_UP }} /> pushes risk up
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-[2px]" style={{ background: SHAP_DOWN }} /> pushes risk down
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Report card
function ReportCard({ report }: { report: Prediction }): ReactNode {
  return (
    <Card className="p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <Eyebrow>Clinical risk report</Eyebrow>
          <div className="mt-1 font-serif text-lg font-semibold text-ink">
            {report.dataset} · {MODEL_LABELS[report.model] ?? report.model}
          </div>
        </div>
        {report.confident ? (
          <Badge tone="credit">confident · {report.coverage_guarantee} coverage</Badge>
        ) : (
          <Badge tone="hold">flag for human review · conformal set {'{'}{report.conformal_set.join(', ')}{'}'}</Badge>
        )}
      </div>

      <div className="mt-6 grid gap-8 md:grid-cols-2">
        <div className="flex flex-col items-center justify-center gap-4">
          <RiskGauge risk={report.risk_probability} />
          <Badge tone={report.prediction === 1 ? 'debit' : 'credit'}>
            prediction: {report.prediction === 1 ? 'disease likely' : 'disease unlikely'}
          </Badge>
          {report.probability_calibrated === false && (
            <div className="font-mono text-[11px] text-hold">
              score-based estimate, not a calibrated probability
            </div>
          )}
        </div>
        <div>
          <Eyebrow>Top contributing features</Eyebrow>
          <div className="mt-3">
            <ShapBars features={report.top_features} />
          </div>
        </div>
      </div>

      <p className="mt-6 border-t border-rule pt-3 font-mono text-[11px] text-ink-45">
        Explanation: {report.explanation_source} · conformal guarantee:{' '}
        {report.coverage_guarantee} coverage on exchangeable data · not a medical device.
      </p>
    </Card>
  );
}

// ---------------------------------------------------------------- Page
export function PatientCheck(): ReactNode {
  const { data: datasets, error: dsError, loading: dsLoading } = useApi<DatasetCard[]>('/api/datasets');
  const [dataset, setDataset] = useState<string>('');
  const [model, setModel] = useState('qsvm');
  const [values, setValues] = useState<Record<string, string>>({});
  const [report, setReport] = useState<Prediction | null>(null);
  const [predictErr, setPredictErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const active = dataset || datasets?.[0]?.name || '';
  const { data: schema, error: schemaError } = useApi<{ feature_names: string[] }>(
    active ? `/api/schema/${active}` : null,
  );

  // Reset the form when the dataset changes.
  useEffect(() => {
    setValues({});
    setReport(null);
    setPredictErr(null);
  }, [active]);

  const loadSample = (): void => {
    const sample = SAMPLE_PATIENTS[active];
    if (!sample) return;
    setValues(Object.fromEntries(Object.entries(sample).map(([k, v]) => [k, String(v)])));
  };

  const submit = async (): Promise<void> => {
    if (!schema) return;
    setBusy(true);
    setPredictErr(null);
    setReport(null);
    try {
      const features = Object.fromEntries(
        schema.feature_names.map((f) => [f, parseFloat(values[f] ?? '')]),
      );
      const bad = schema.feature_names.filter((f) => Number.isNaN(features[f]));
      if (bad.length) throw new Error(`fill in: ${bad.slice(0, 5).join(', ')}${bad.length > 5 ? '…' : ''}`);
      const rep = await api<Prediction>(`/api/predict/${active}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features, model }),
      });
      setReport(rep);
    } catch (e) {
      setPredictErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <Eyebrow>Per-patient inference</Eyebrow>
      <PageTitle>Patient check</PageTitle>
      <p className="mt-2 max-w-2xl text-[14px] text-ink-60">
        Enter patient measurements and get a calibrated risk estimate with a conformal
        confidence guarantee and a per-patient explanation.
      </p>

      {dsError && <ErrorState>Could not reach the API ({dsError}).</ErrorState>}
      {dsLoading && (
        <>
          <Skeleton className="mt-6 h-40" />
          <ColdStartNote />
        </>
      )}

      {datasets && (
        <Card className="mt-6 p-5">
          <div className="flex flex-wrap items-end gap-4">
            <label className="block">
              <span className="eyebrow">Dataset</span>
              <select
                className="mt-1.5 block h-9 rounded-sm border border-rule bg-paper px-2 text-[13px]"
                value={active}
                onChange={(e) => setDataset(e.target.value)}
              >
                {datasets.map((d) => (
                  <option key={d.name} value={d.name}>{d.title}</option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="eyebrow">Model</span>
              <select
                className="mt-1.5 block h-9 rounded-sm border border-rule bg-paper px-2 text-[13px]"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              >
                {MODELS.map((m) => (
                  <option key={m} value={m}>{MODEL_LABELS[m] ?? m}</option>
                ))}
              </select>
            </label>
            <Button onClick={loadSample} disabled={!SAMPLE_PATIENTS[active]}>
              Load sample patient
            </Button>
            <Button variant="quantum" onClick={() => void submit()} disabled={busy || !schema}>
              {busy ? 'Scoring…' : 'Assess risk'}
            </Button>
          </div>

          {schemaError && (
            <div className="mt-4">
              <ErrorState>No trained bundle for “{active}” ({schemaError}).</ErrorState>
            </div>
          )}

          {schema && (
            <div className="mt-5 grid gap-3 border-t border-rule pt-4 sm:grid-cols-3 lg:grid-cols-5">
              {schema.feature_names.map((f) => (
                <label key={f} className="block">
                  <span className="block truncate font-mono text-[11px] text-ink-60" title={f}>{f}</span>
                  <input
                    type="number"
                    step="any"
                    className="mt-1 block h-8 w-full rounded-sm border border-rule bg-paper px-2 font-mono text-[12px] tnum"
                    value={values[f] ?? ''}
                    onChange={(e) => setValues((v) => ({ ...v, [f]: e.target.value }))}
                  />
                </label>
              ))}
            </div>
          )}

          {predictErr && (
            <div className="mt-4">
              <ErrorState>{predictErr}</ErrorState>
            </div>
          )}
        </Card>
      )}

      {report && (
        <div className="mt-6">
          <ReportCard report={report} />
        </div>
      )}
    </div>
  );
}
