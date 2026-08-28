import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { MODEL_LABELS, useApi, type DatasetCard } from '../lib/api';
import { Badge, Card, ErrorState, Eyebrow, PageTitle, Skeleton , ColdStartNote } from '../components/ui';
import { CountUp, Reveal } from '../components/motion';
import { HardwareSection } from '../components/Hardware';

function DatasetTile({ d }: { d: DatasetCard }): ReactNode {
  return (
    <Link to={`/datasets/${d.name}`} className="block">
      <Card className="h-full p-5 transition-colors hover:border-quantum/50">
        <div className="flex items-start justify-between gap-2">
          <div>
            <div className="font-mono text-[11px] text-ink-45 uppercase">{d.name}</div>
            <h3 className="mt-0.5 font-serif text-lg leading-snug font-semibold text-ink">{d.title}</h3>
          </div>
          <div className="flex shrink-0 flex-col items-end gap-1">
            <Badge tone="quantum">{d.n_qubits} qubits</Badge>
            {d.name === 'quantum_synth' && <Badge tone="hold">synthetic benchmark</Badge>}
            {d.image_track && <Badge tone="action">image track</Badge>}
          </div>
        </div>

        <div className="mt-5 flex items-end gap-6">
          <div>
            <div className="eyebrow">Best AUROC</div>
            <div className="mt-1 font-mono text-[28px] leading-none text-ink tnum">
              <CountUp value={d.best_auroc} />
            </div>
            <div className="mt-1 text-[12px] text-ink-60">{MODEL_LABELS[d.best_model] ?? d.best_model}</div>
          </div>
          {d.qsvm_auroc != null && (
            <div>
              <div className="eyebrow">QSVM</div>
              <div className="mt-1 font-mono text-[28px] leading-none text-quantum tnum">
                <CountUp value={d.qsvm_auroc} />
              </div>
              <div className="mt-1 text-[12px] text-ink-60">quantum kernel</div>
            </div>
          )}
        </div>

        {d.deployment_floor != null && d.deployment_floor_auroc != null && (
          <div className="mt-3 font-mono text-[11px] text-ink-45 tnum">
            classical floor: {d.deployment_floor_auroc.toFixed(2)} ({d.deployment_floor}, all features)
          </div>
        )}

        <div className="mt-5 border-t border-rule pt-3">
          {d.quantum_worth_trying ? (
            <Badge tone="quantum">advantage scout: quantum worth trying</Badge>
          ) : (
            <Badge tone="neutral">advantage scout: no quantum edge expected</Badge>
          )}
        </div>
      </Card>
    </Link>
  );
}

export function Dashboard(): ReactNode {
  const { data, error, loading } = useApi<DatasetCard[]>('/api/datasets');

  return (
    <div>
      <Eyebrow>Flagship datasets</Eyebrow>
      <div className="mt-1 flex items-baseline justify-between gap-4">
        <PageTitle>Dashboard</PageTitle>
      </div>
      <p className="mt-2 max-w-2xl text-[14px] text-ink-60">
        Three public clinical benchmarks, each run through the full pipeline: qubit budget,
        advantage scout, quantum models and their classical twins on a locked held-out split.
      </p>

      <div className="mt-6">
        {loading && (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-56" />
            ))}
            <ColdStartNote />
          </div>
        )}
        {error && (
          <ErrorState>
            Could not reach the Q-Nidaan API ({error}). Start the backend on :8000 and reload.
          </ErrorState>
        )}
        {data && (
          <Reveal stagger={0.1} className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {data.map((d) => (
              <DatasetTile key={d.name} d={d} />
            ))}
          </Reveal>
        )}
      </div>

      <div className="mt-8">
        <HardwareSection full />
      </div>
    </div>
  );
}
