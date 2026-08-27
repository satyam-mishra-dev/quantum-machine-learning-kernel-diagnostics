import type { ReactNode } from 'react';
import { useApi } from '../lib/api';
import { Eyebrow } from './ui';
import { CountUp, Reveal } from './motion';

type HardwarePatient = {
  true_label: number;
  sim_score: number;
  hw_score: number;
  sim_prediction: number;
  hw_prediction: number;
  agree: boolean;
};

type HardwareRun = {
  dataset: string;
  backend: string;
  backend_qubits: number;
  n_qubits_used: number;
  shots: number;
  wall_seconds: number;
  patients: HardwarePatient[];
  sim_hw_agreement: number;
};

const SIM_DOT = '#6b6c6f';
const HW_DOT = '#6d28d9';

/** Dumbbell row: sim score (gray) vs hardware score (violet) on a shared scale.
    Noise pulls the hardware dot toward zero; the connecting line shows by how much. */
function ScorePair({ p, max }: { p: HardwarePatient; max: number }): ReactNode {
  const x = (v: number): number => 4 + (Math.abs(v) / max) * 92; // percent, padded
  return (
    <div className="flex items-center gap-3 py-1.5">
      <span className="w-16 shrink-0 font-mono text-[11px] text-ink-60 tnum">y={p.true_label}</span>
      <svg viewBox="0 0 100 10" preserveAspectRatio="none" className="h-4 min-w-0 flex-1">
        <line x1="4" y1="5" x2="96" y2="5" stroke="var(--color-rule)" strokeWidth="0.6" />
        <line x1={x(p.sim_score)} y1="5" x2={x(p.hw_score)} y2="5" stroke={HW_DOT} strokeWidth="1.2" opacity="0.45" />
        <circle cx={x(p.sim_score)} cy="5" r="2.6" fill="var(--color-paper)" stroke={SIM_DOT} strokeWidth="1.4" />
        <circle cx={x(p.hw_score)} cy="5" r="2.6" fill={HW_DOT} />
      </svg>
      <span className="w-28 shrink-0 text-right font-mono text-[11px] text-ink-60 tnum">
        {p.sim_score.toFixed(3)} → {p.hw_score.toFixed(3)}
      </span>
      <span className="w-8 shrink-0 text-right">
        {p.agree ? (
          <span className="font-mono text-[12px] text-credit">✓</span>
        ) : (
          <span className="font-mono text-[12px] text-debit">✗</span>
        )}
      </span>
    </div>
  );
}

function RunReceipt({ run, full }: { run: HardwareRun; full: boolean }): ReactNode {
  const agree = run.patients.filter((p) => p.agree).length;
  const max = Math.max(...run.patients.flatMap((p) => [Math.abs(p.sim_score), Math.abs(p.hw_score)]), 1e-9);
  return (
    <div className="q-gradient-border p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Eyebrow className="text-quantum">Ran on real quantum hardware</Eyebrow>
          <div className="mt-1.5 font-serif text-2xl font-semibold">
            <span className="q-gradient-text">{run.backend}</span>
          </div>
          <div className="mt-1 text-[13px] text-ink-60">
            {run.n_qubits_used} of {run.backend_qubits} qubits · dataset: {run.dataset}
          </div>
        </div>
        <div className="text-right">
          <div className="font-mono text-[28px] leading-none text-ink tnum">
            <CountUp value={agree} decimals={0} />
            <span className="text-ink-45">/{run.patients.length}</span>
          </div>
          <div className="eyebrow mt-1">predictions agree with simulator</div>
        </div>
      </div>

      {full && run.patients.length > 0 && (
        <div className="mt-5 border-t border-rule pt-4">
          <div className="flex items-center justify-between gap-4">
            <Eyebrow>Per-patient scores — simulator vs hardware</Eyebrow>
            <span className="hidden font-mono text-[11px] text-ink-60 sm:inline">
              <span className="mr-1 inline-block h-2 w-2 rounded-full border align-middle" style={{ borderColor: SIM_DOT }} /> sim
              <span className="mr-1 ml-3 inline-block h-2 w-2 rounded-full align-middle" style={{ background: HW_DOT }} /> hardware
            </span>
          </div>
          <div className="mt-2">
            {run.patients.map((p, i) => (
              <ScorePair key={i} p={p} max={max} />
            ))}
          </div>
          <p className="mt-2 text-[12px] text-ink-60">
            Hardware noise shrinks scores toward zero — but the predictions hold.
          </p>
        </div>
      )}

      <p className="mt-4 border-t border-rule pt-3 font-mono text-[11px] text-ink-45">
        {run.shots.toLocaleString()} shots · {run.wall_seconds.toFixed(1)}s wall time · IBM Quantum
      </p>
    </div>
  );
}

/** Hidden entirely when the endpoint is missing, errors, or returns []. */
export function HardwareSection({ full = false }: { full?: boolean }): ReactNode {
  const { data } = useApi<HardwareRun[]>('/api/hardware');
  if (!data || data.length === 0) return null;
  return (
    <Reveal className="space-y-4">
      {data.map((run) => (
        <RunReceipt key={`${run.dataset}-${run.backend}`} run={run} full={full} />
      ))}
    </Reveal>
  );
}
