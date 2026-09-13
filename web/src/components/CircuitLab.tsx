import { ReactNode } from 'react';
import { Card, Eyebrow } from './ui';

export function CircuitLab({ circuit, stage, telemetry }: { circuit: any, stage?: string, telemetry?: any }): ReactNode {
  if (!circuit || !circuit.drawing) return null;
  
  const isExecuting = stage === 'TRAINING_QUANTUM' || stage === 'QUANTUM_EXECUTION' || (telemetry && telemetry.stage === 'TRAINING_QUANTUM');
  const isDone = stage === 'EVALUATING' || stage === 'UNCERTAINTY' || stage === 'EXPLAINABILITY' || stage === 'REPORTING' || stage === 'COMPLETED';
  
  const pulseClass = isExecuting ? 'animate-pulse text-quantum' : isDone ? 'text-quantum' : 'text-quantum-wash';
  const borderClass = isExecuting ? 'border-quantum shadow-[0_0_15px_rgba(138,43,226,0.3)]' : 'border-quantum/30';
  
  const nQubits = circuit.n_qubits || 4;
  const qubits = Array.from({ length: nQubits }, (_, i) => i);

  return (
    <Card className={`mt-6 p-5 bg-[#090b0f] transition-all duration-500 border ${borderClass}`}>
      <div className="flex justify-between items-center mb-6">
        <Eyebrow className={isExecuting ? 'text-quantum animate-pulse' : 'text-quantum'}>
          {isExecuting ? '● LIVE EXECUTION' : isDone ? '✓ EXECUTION COMPLETE' : 'Quantum Circuit Topology'}
        </Eyebrow>
        <div className="flex gap-2 items-center">
            {circuit.execution_type?.includes('QPU') ? (
              <span className="px-2 py-0.5 rounded-sm bg-quantum/20 text-quantum text-[10px] font-bold">REAL IBM QPU</span>
            ) : (
              <span className="px-2 py-0.5 rounded-sm bg-ink-60/20 text-ink-60 text-[10px] font-bold">SIMULATED EXECUTION</span>
            )}
            <span className="text-[11px] font-mono text-ink-60">{circuit.ansatz || 'AngleEmbedding + IsingZZ'}</span>
        </div>
      </div>
      
      {/* Visual Animation Box */}
      <div className="relative mb-8 bg-[#05070a] border border-quantum/20 rounded-sm p-6 overflow-hidden">
        {/* Animated execution scanline if executing */}
        {isExecuting && (
          <div className="absolute top-0 bottom-0 left-0 w-8 bg-gradient-to-r from-transparent via-quantum/30 to-transparent animate-[scan_2s_linear_infinite] z-0" style={{ animation: 'scan 2s linear infinite' }} />
        )}

        <div className="relative z-10 flex flex-col gap-6 font-mono text-xs text-ink-60">
          {qubits.map(q => (
             <div key={q} className="flex items-center gap-4 relative">
                <div className="w-6 text-right font-bold text-quantum-wash">q{q}</div>
                
                {/* The wire */}
                <div className={`flex-1 h-px ${isExecuting ? 'bg-quantum/40' : 'bg-rule'} relative flex items-center`}>
                   
                   {/* Angle Embedding Gate */}
                   <div className={`absolute left-[15%] px-2 py-1 rounded-sm border ${isExecuting ? 'bg-quantum/20 border-quantum text-quantum shadow-[0_0_8px_rgba(138,43,226,0.5)]' : 'bg-wash border-rule text-ink-60'}`}>
                     RX
                   </div>

                   {/* Entanglement Gate (simplified representation) */}
                   {q < nQubits - 1 && (
                     <>
                       <div className={`absolute left-[50%] w-2 h-2 rounded-full ${isExecuting ? 'bg-quantum shadow-[0_0_5px_rgba(138,43,226,1)]' : 'bg-ink-60'}`} />
                       <div className={`absolute left-[50%] top-0 w-px h-6 ${isExecuting ? 'bg-quantum' : 'bg-rule'}`} />
                     </>
                   )}
                   {q > 0 && (
                     <>
                       <div className={`absolute left-[50%] w-3 h-3 rounded-full border-2 ${isExecuting ? 'border-quantum text-quantum bg-[#05070a]' : 'border-ink-60 bg-[#05070a]'} flex items-center justify-center translate-x-[-2px] translate-y-[-1px]`}>
                         <div className="w-1.5 h-px bg-current" />
                         <div className="w-px h-1.5 bg-current absolute" />
                       </div>
                     </>
                   )}

                   {/* Measurement */}
                   <div className={`absolute right-4 px-2 py-1 rounded-sm border ${isExecuting ? 'bg-quantum/10 border-quantum text-quantum' : 'bg-wash border-rule text-ink-60'} flex items-center gap-1`}>
                     <span className="text-[10px]">M</span>
                   </div>
                </div>
             </div>
          ))}
        </div>
      </div>
      
      {isExecuting && telemetry && (
        <div className="mb-6 bg-paper/10 p-4 rounded-sm border border-quantum/50 text-[11px] font-mono text-ink-60">
          <div className="flex justify-between mb-1">
             <span className="text-quantum font-bold">{telemetry.model || 'Quantum'} Kernel Execution</span>
             <span>{telemetry.percentage?.toFixed(1) || 0}%</span>
          </div>
          <div className="h-1.5 w-full bg-wash rounded-full overflow-hidden mb-2">
             <div className="bg-quantum h-full transition-all duration-300" style={{ width: `${Math.min(100, telemetry.percentage || 0)}%` }} />
          </div>
          <div className="flex justify-between mt-2">
             <span>{telemetry.completed || 0} / {telemetry.total || 0} evaluations</span>
             {telemetry.eta_seconds != null ? (
               <span>{telemetry.rate?.toFixed(1) || 0} eval/s · ETA {Math.ceil(telemetry.eta_seconds)}s</span>
             ) : (
               <span>ETA calculating...</span>
             )}
          </div>
        </div>
      )}
      
      <div className="flex justify-between items-end">
        <div className="flex gap-4 text-[11px] font-mono text-ink-60 bg-paper/5 p-2 rounded-sm border border-rule/50">
           <div>Depth: {circuit.depth} layers</div>
           <div>Operations: {circuit.num_operations} total gates</div>
        </div>
        
        <div className="text-[10px] text-ink-60">
          RAW PENNYLANE CIRCUIT DRAW BELOW
        </div>
      </div>
      
      <div className="overflow-x-auto mt-2 bg-[#05070a] p-4 border border-quantum/10 rounded-sm">
        <pre className={`font-mono text-[11px] leading-[14px] transition-colors duration-300 ${pulseClass}`}>
          {circuit.drawing}
        </pre>
      </div>
    </Card>
  );
}
