import type { ReactNode } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { isFullFloor, isQuantum, MODEL_LABELS, useApi, type RunDetail } from '../lib/api';
import { AXIS_TICK, CLASSICAL_BAR, QUANTUM_BAR, SERIES_COLORS, TOOLTIP_STYLE } from '../lib/chart';
import { Badge, buttonClass, Card, ErrorState, Eyebrow, KV, PageTitle, Skeleton } from '../components/ui';
import { Reveal } from '../components/motion';

const fmt = (v: number | null | undefined, d = 3): string => (v == null ? '—' : v.toFixed(d));

function ModelBars({ heldout }: { heldout: RunDetail['heldout'] }): ReactNode {
  // Compressed models first (the fair comparison), full-feature floors last as context.
  const rows = Object.entries(heldout)
    .map(([model, m]) => ({
      model,
      label: MODEL_LABELS[model] ?? model,
      auroc: m.auroc,
      quantum: isQuantum(model),
      full: isFullFloor(model, m),
    }))
    .sort((a, b) => Number(a.full) - Number(b.full));
  const hasFloors = rows.some((r) => r.full);
  return (
    <div className="h-72">
      <ResponsiveContainer>
        <BarChart data={rows} margin={{ top: 20, right: 8, left: -16, bottom: 0 }} barCategoryGap="28%">
          <CartesianGrid vertical={false} stroke="var(--color-rule)" />
          <XAxis dataKey="label" tick={AXIS_TICK} tickLine={false} axisLine={{ stroke: 'var(--color-rule)' }} interval={0} />
          <YAxis domain={[0, 1]} tick={AXIS_TICK} tickLine={false} axisLine={false} />
          <Tooltip
            cursor={{ fill: 'var(--color-wash)' }}
            contentStyle={TOOLTIP_STYLE}
            formatter={(v) => [Number(v).toFixed(3), 'AUROC']}
          />
          <Bar dataKey="auroc" radius={[4, 4, 0, 0]} isAnimationActive animationDuration={900}>
            <LabelList dataKey="auroc" position="top" formatter={(v: unknown) => Number(v).toFixed(3)} style={{ fill: 'var(--color-ink)', fontSize: 11, fontFamily: 'var(--font-mono)' }} />
            {rows.map((r) => (
              <Cell
                key={r.model}
                fill={r.full ? 'var(--color-wash)' : r.quantum ? QUANTUM_BAR : CLASSICAL_BAR}
                stroke={r.full ? CLASSICAL_BAR : undefined}
                strokeWidth={r.full ? 1.5 : 0}
                strokeDasharray={r.full ? '4 3' : undefined}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-1 flex items-center gap-4 font-mono text-[11px] text-ink-60">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-[2px]" style={{ background: QUANTUM_BAR }} /> quantum
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-[2px]" style={{ background: CLASSICAL_BAR }} /> classical
        </span>
        {hasFloors && (
          <span className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-2.5 rounded-[2px]"
              style={{ border: `1.5px dashed ${CLASSICAL_BAR}`, background: 'var(--color-wash)' }}
            />{' '}
            full-feature floor (context)
          </span>
        )}
      </div>
    </div>
  );
}

function SampleEfficiency({ points }: { points: NonNullable<RunDetail['sample_efficiency']> }): ReactNode {
  const models = Object.keys(points[0] ?? {}).filter((k) => k !== 'n_train');
  const rows = points.map((p) => {
    const row: Record<string, number> = { n_train: p.n_train };
    for (const m of models) {
      const v = p[m];
      row[m] = typeof v === 'number' ? v : (v?.auroc_mean ?? NaN);
    }
    return row;
  });
  return (
    <div className="h-80">
      <ResponsiveContainer>
        <LineChart data={rows} margin={{ top: 8, right: 24, left: -16, bottom: 4 }}>
          <CartesianGrid vertical={false} stroke="var(--color-rule)" />
          <XAxis
            dataKey="n_train"
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={{ stroke: 'var(--color-rule)' }}
            label={{ value: 'training samples', position: 'insideBottom', offset: -2, style: { fill: 'var(--color-ink-60)', fontSize: 11, fontFamily: 'var(--font-mono)' } }}
          />
          <YAxis domain={['auto', 'auto']} tick={AXIS_TICK} tickLine={false} axisLine={false} />
          <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v, name) => [Number(v).toFixed(3), MODEL_LABELS[String(name)] ?? name]} />
          <Legend formatter={(v) => <span style={{ color: 'var(--color-ink-60)', fontSize: 12 }}>{MODEL_LABELS[v] ?? v}</span>} />
          {models.map((m) => (
            <Line
              key={m}
              type="monotone"
              dataKey={m}
              stroke={SERIES_COLORS[m] ?? CLASSICAL_BAR}
              strokeWidth={isQuantum(m) ? 2.5 : 2}
              dot={{ r: 3, strokeWidth: 0, fill: SERIES_COLORS[m] ?? CLASSICAL_BAR }}
              activeDot={{ r: 5 }}
              isAnimationActive
              animationDuration={1100}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function DatasetDetail(): ReactNode {
  const { name } = useParams();
  const { data, error, loading } = useApi<RunDetail>(name ? `/api/runs/${name}` : null);

  if (loading) return <Skeleton className="h-96" />;
  if (error) return <ErrorState>Could not load run for “{name}” ({error}).</ErrorState>;
  if (!data) return null;

  const { dataset, budget, scout, heldout } = data;

  return (
    <div>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Eyebrow>
            <Link to="/dashboard" className="link">Dashboard</Link> / {dataset.name}
          </Eyebrow>
          <PageTitle>{dataset.title}</PageTitle>
          <div className="mt-2 flex flex-wrap gap-2">
            <Badge tone="quantum">{budget.n_qubits} qubits · {budget.method}</Badge>
            {budget.explained_variance != null && (
              <Badge tone="neutral">{(budget.explained_variance * 100).toFixed(1)}% variance kept</Badge>
            )}
            <Badge tone="neutral">{dataset.n_features} features</Badge>
          </div>
        </div>
        <Link to="/check" className={buttonClass('quantum')}>Run a patient check</Link>
      </div>

      <Reveal className="mt-8 grid gap-4 lg:grid-cols-3">
        <Card className="p-5 lg:col-span-2">
          <Eyebrow>Held-out AUROC — quantum vs classical twins</Eyebrow>
          <div className="mt-4">
            <ModelBars heldout={heldout} />
          </div>
        </Card>

        {scout ? (
          <Card className="p-5">
            <Eyebrow className="text-quantum">Advantage scout</Eyebrow>
            <div className="mt-3">
              <Badge tone={scout.quantum_worth_trying ? 'quantum' : 'neutral'}>
                {scout.verdict
                  ? scout.verdict.replaceAll('_', ' ')
                  : `advantage possible: ${scout.quantum_worth_trying ? 'yes' : 'no'}`}
              </Badge>
            </div>
            <div className="mt-4">
              <div className="eyebrow">spectral asymmetry (validated in-platform)</div>
              <div className="mt-1 font-mono text-[28px] leading-none text-ink tnum">
                {fmt(scout.g_asym, 2)}
              </div>
            </div>
            <div className="mt-3">
              <KV k="Huang et al. geometric difference (naive reg.)" v={fmt(scout.g_huang_naive_reg, 2)} />
              <KV k="KTA (quantum kernel)" v={fmt(scout.kta_quantum)} />
              <KV k="KTA (classical kernel)" v={fmt(scout.kta_classical)} />
            </div>
            <p className="mt-4 text-[12px] leading-relaxed text-ink-60">
              The scout compares kernel–target alignment and the spectral asymmetry between
              quantum and classical kernels <em>before</em> training. Large asymmetry with usable
              alignment means the quantum kernel sees structure the classical one cannot — only
              then is quantum worth the hardware time.
            </p>
          </Card>
        ) : (
          <Card className="p-5">
            <Eyebrow className="text-quantum">Image track — operating point</Eyebrow>
            <div className="mt-3">
              {Object.entries(heldout).map(([m, e]) => (
                <div key={m} className="mb-4 last:mb-0">
                  <div className="text-[13px] font-medium text-ink">
                    {MODEL_LABELS[m] ?? m}{' '}
                    {isQuantum(m) && <Badge tone="quantum" className="ml-1">quantum</Badge>}
                  </div>
                  <KV k="Sensitivity" v={fmt(e.sensitivity)} />
                  <KV k="Specificity" v={fmt(e.specificity)} />
                  <KV k="Threshold (from train)" v={fmt(e.threshold_from_train, 2)} />
                </div>
              ))}
            </div>
            <p className="mt-2 text-[12px] leading-relaxed text-ink-60">
              Image-track runs compare a classical CNN against a hybrid quantum neural network
              at a decision threshold chosen on training data only.
            </p>
          </Card>
        )}
      </Reveal>

      {data.sample_efficiency && data.sample_efficiency.length > 0 && (
        <Reveal className="mt-4">
          <Card className="p-5">
            <Eyebrow>Sample efficiency — AUROC vs training-set size</Eyebrow>
            <p className="mt-1 text-[12px] text-ink-60">
              The headline question: does the quantum kernel learn more from fewer patients?
            </p>
            <div className="mt-4">
              <SampleEfficiency points={data.sample_efficiency} />
            </div>
          </Card>
        </Reveal>
      )}

      <Reveal className="mt-4">
        <Card className="p-5">
          <Eyebrow>Held-out metrics — full table</Eyebrow>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="border-b border-rule text-left">
                  <th className="py-2 pr-4 font-medium text-ink-60">Model</th>
                  <th className="py-2 pr-4 font-medium text-ink-60">AUROC</th>
                  <th className="py-2 pr-4 font-medium text-ink-60">Accuracy</th>
                  <th className="py-2 pr-4 font-medium text-ink-60">Spec @ 95% sens</th>
                  <th className="py-2 pr-4 font-medium text-ink-60">Train (s)</th>
                </tr>
              </thead>
              <tbody className="font-mono tnum">
                {Object.entries(heldout).map(([model, m]) => (
                  <tr key={model} className="border-b border-rule last:border-b-0">
                    <td className="py-2 pr-4 font-sans">
                      {MODEL_LABELS[model] ?? model}{' '}
                      {isQuantum(model) && <Badge tone="quantum" className="ml-1">quantum</Badge>}
                    </td>
                    <td className="py-2 pr-4">{fmt(m.auroc)}</td>
                    <td className="py-2 pr-4">{fmt(m.accuracy)}</td>
                    <td className="py-2 pr-4">{fmt(m.spec_at_95sens)}</td>
                    <td className="py-2 pr-4">{m.train_seconds != null ? m.train_seconds.toFixed(1) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </Reveal>
    </div>
  );
}
