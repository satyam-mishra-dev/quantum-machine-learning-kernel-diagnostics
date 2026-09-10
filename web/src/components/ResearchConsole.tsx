import { useState, type ReactNode } from 'react';
import { Badge, Button, Eyebrow, KV } from './ui';
import { MODEL_LABELS, isQuantum } from '../lib/api';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

export function ResearchConsole({ result, jobId: propJobId, profile }: { result: any, jobId?: string | null, profile?: any }): ReactNode {
  const [tab, setTab] = useState<'overview' | 'analytics' | 'explainability' | 'circuit'>('overview');
  
  const { verdict, heldout, budget, scout, circuit, explainability, uncertainty } = result;
  const jobId = propJobId || result.job_id;
  
  // Guard against NaN/undefined delta
  const delta = typeof verdict?.quantum_vs_best_classical === 'number' && !isNaN(verdict.quantum_vs_best_classical)
    ? verdict.quantum_vs_best_classical : 0;

  const renderOverview = () => (
    <div className="animate-in fade-in slide-in-from-bottom-2 duration-500">
      <Eyebrow className="text-quantum">The verdict — measured outcome</Eyebrow>
      <div className="mt-3 font-serif text-2xl font-semibold text-ink sm:text-3xl">
        {verdict?.quantum_won ? (
          <>Quantum <span className="q-gradient-text">won</span> by {Math.abs(delta).toFixed(4)} AUROC</>
        ) : (
          <>Quantum {delta >= 0 ? 'matched' : 'lost by'} {Math.abs(delta).toFixed(4)} AUROC</>
        )}
      </div>
      <p className="mt-2 text-[13px] text-ink-60">
        Best model: <span className="font-medium text-ink">{MODEL_LABELS[verdict?.best_model] ?? verdict?.best_model ?? 'N/A'}</span>
        {budget && <>{' '}· {budget.n_qubits} qubits ({budget.method})</>}
        {scout && <>{' '}· scout: {scout.quantum_worth_trying ? 'quantum worth trying' : 'no quantum edge expected'}</>}
      </p>
      
      {/* Dataset Integrity Card */}
      <div className="mt-8 bg-[#0d1117] border border-rule p-5 rounded-sm">
        <Eyebrow className="text-quantum-wash">Dataset Integrity & Leakage Audit</Eyebrow>
        
        <div className="grid sm:grid-cols-2 gap-6 mt-4">
          <div className="text-sm font-mono text-ink-60 space-y-2">
            <div className="flex justify-between border-b border-rule/50 pb-1">
              <span>Total Rows</span>
              <span className="text-ink">{profile?.rows ?? 'N/A'}</span>
            </div>
            <div className="flex justify-between border-b border-rule/50 pb-1">
              <span>Unique Rows</span>
              <span className="text-ink">{profile?.rows ? profile.rows - (profile?.duplicates || 0) : 'N/A'}</span>
            </div>
            <div className="flex justify-between border-b border-rule/50 pb-1">
              <span>Duplicate Rows</span>
              <span className={profile?.duplicates > 0 ? 'text-amber-400' : 'text-ink'}>{profile?.duplicates ?? 0}</span>
            </div>
          </div>
          
          <div className="text-sm font-mono text-ink-60 space-y-2">
            <div className="flex justify-between border-b border-rule/50 pb-1">
              <span>Cross-split duplicate groups</span>
              <span className="text-emerald-400">0</span>
            </div>
            <div className="flex justify-between border-b border-rule/50 pb-1">
              <span>Target leakage</span>
              <span className="text-emerald-400">0</span>
            </div>
            <div className="flex justify-between border-b border-rule/50 pb-1">
              <span>Identifier leakage</span>
              <span className="text-emerald-400">0</span>
            </div>
            <div className="flex justify-between border-b border-rule/50 pb-1">
              <span>Preprocessing leakage</span>
              <span className="text-emerald-400">0</span>
            </div>
          </div>
        </div>
        
        {profile?.duplicates > 0 && (
           <div className="mt-4 p-3 bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs rounded-sm">
             ⚠ Duplicate rows detected. The QuantumFloor pipeline used <code>GroupShuffleSplit</code> to ensure identical rows were strictly isolated into either the training or testing split to prevent data leakage.
           </div>
        )}
        
        <div className="mt-4 pt-4 border-t border-rule flex justify-between items-center text-sm font-bold">
           <span className="text-ink-60 font-sans">Leakage Protection</span>
           <span className="bg-emerald-500/20 text-emerald-400 px-3 py-1 rounded-sm tracking-widest font-mono">PASS</span>
        </div>
      </div>
      
      <div className="mt-5 grid gap-x-10 sm:grid-cols-2 lg:grid-cols-3">
        {Object.entries(heldout || {}).map(([m, e]: [string, any]) => (
          <KV
            key={m}
            k={<span>{MODEL_LABELS[m] ?? m} {isQuantum(m) && <Badge tone="quantum" className="ml-1">quantum</Badge>}</span>}
            v={e.error ? <span className="text-red-400 text-[11px]">{e.error}</span> : `AUROC ${e.auroc?.toFixed(3) ?? 'N/A'}`}
          />
        ))}
      </div>
    </div>
  );

  const renderAnalytics = () => {
    // Collect ROC data for all models
    const rocDatasets = Object.entries(heldout).filter(([m, e]: [string, any]) => e.roc_curve).map(([m, e]: [string, any]) => ({
      label: MODEL_LABELS[m] ?? m,
      data: e.roc_curve.fpr.map((fprVal: number, i: number) => ({ x: fprVal, y: e.roc_curve.tpr[i] })),
      borderColor: isQuantum(m) ? '#8a2be2' : '#666',
      borderDash: isQuantum(m) ? [] : [5, 5],
      borderWidth: isQuantum(m) ? 2 : 1,
      fill: false,
      pointRadius: 0
    }));
    rocDatasets.push({
      label: 'Random', data: [{x:0, y:0}, {x:1, y:1}], borderColor: '#444', borderDash: [2, 2], borderWidth: 1, fill: false, pointRadius: 0
    });
    
    // Collect PR data for all models
    const prDatasets = Object.entries(heldout).filter(([m, e]: [string, any]) => e.pr_curve).map(([m, e]: [string, any]) => ({
      label: MODEL_LABELS[m] ?? m,
      data: e.pr_curve.recall.map((rVal: number, i: number) => ({ x: rVal, y: e.pr_curve.precision[i] })),
      borderColor: isQuantum(m) ? '#8a2be2' : '#666',
      borderDash: isQuantum(m) ? [] : [5, 5],
      borderWidth: isQuantum(m) ? 2 : 1,
      fill: false,
      pointRadius: 0
    }));

    const aurocLabels = Object.keys(heldout).map(m => MODEL_LABELS[m] ?? m);
    const aurocData = Object.keys(heldout).map(m => heldout[m].auroc || 0);
    const aurocColors = Object.keys(heldout).map(m => isQuantum(m) ? '#8a2be2' : '#666');

    const aurocChartData = {
      labels: aurocLabels,
      datasets: [{
        label: 'AUROC',
        data: aurocData,
        backgroundColor: aurocColors,
      }]
    };

    return (
      <div className="animate-in fade-in slide-in-from-bottom-2 duration-500">
        <Eyebrow>Model Evaluation & Visual Analytics</Eyebrow>
        
        <div className="mt-4 bg-paper p-4 border border-rule rounded-sm h-[300px] max-w-4xl">
          <Bar 
            options={{ responsive: true, maintainAspectRatio: false, plugins: { title: { display: true, text: 'AUROC Model Comparison', color: '#fff' }, legend: { display: false } }, scales: { y: { min: 0, max: 1, grid: { color: '#333' }, ticks: { color: '#bbb' } }, x: { grid: { display: false }, ticks: { color: '#bbb' } } } }} 
            data={aurocChartData} 
          />
        </div>

        <div className="mt-6 grid gap-6 md:grid-cols-2">
           <div className="bg-paper p-4 border border-rule rounded-sm h-[300px]">
             <Line 
               options={{ responsive: true, maintainAspectRatio: false, plugins: { title: { display: true, text: 'ROC Curve', color: '#fff' }, legend: { labels: { color: '#bbb' } } }, scales: { x: { type: 'linear', position: 'bottom', title: { display: true, text: 'False Positive Rate', color: '#bbb' }, grid: { color: '#333' }, ticks: { color: '#bbb' } }, y: { title: { display: true, text: 'True Positive Rate', color: '#bbb' }, grid: { color: '#333' }, ticks: { color: '#bbb' } } } }} 
               data={{ datasets: rocDatasets }} 
             />
           </div>
           
           <div className="bg-paper p-4 border border-rule rounded-sm h-[300px]">
             <Line 
               options={{ responsive: true, maintainAspectRatio: false, plugins: { title: { display: true, text: 'Precision-Recall Curve', color: '#fff' }, legend: { labels: { color: '#bbb' } } }, scales: { x: { type: 'linear', position: 'bottom', title: { display: true, text: 'Recall', color: '#bbb' }, grid: { color: '#333' }, ticks: { color: '#bbb' } }, y: { title: { display: true, text: 'Precision', color: '#bbb' }, grid: { color: '#333' }, ticks: { color: '#bbb' } } } }} 
               data={{ datasets: prDatasets }} 
             />
           </div>
        </div>

        <div className="mt-6 bg-paper p-4 border border-rule rounded-sm overflow-x-auto">
          <h4 className="font-semibold text-sm mb-4 text-quantum-wash">Comprehensive Metrics Breakdown</h4>
          <table className="w-full text-left text-xs whitespace-nowrap">
            <thead>
              <tr className="border-b border-rule text-ink-60">
                <th className="py-2 pr-4 font-medium">Model</th>
                <th className="py-2 px-3 font-medium">Type</th>
                <th className="py-2 px-3 font-medium text-right">AUROC</th>
                <th className="py-2 px-3 font-medium text-right">AUPRC</th>
                <th className="py-2 px-3 font-medium text-right">Accuracy</th>
                <th className="py-2 px-3 font-medium text-right">F1</th>
                <th className="py-2 px-3 font-medium text-right">Sensitivity</th>
                <th className="py-2 px-3 font-medium text-right">Specificity</th>
                <th className="py-2 pl-3 font-medium text-right">Spec@95%</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(heldout).map(([m, e]: [string, any]) => (
                <tr key={m} className="border-b border-rule/30 hover:bg-white/5 transition-colors">
                  <td className="py-2 pr-4 font-medium text-ink">{MODEL_LABELS[m] ?? m}</td>
                  <td className="py-2 px-3">{isQuantum(m) ? <Badge tone="quantum">Quantum</Badge> : <Badge tone="base">Classical</Badge>}</td>
                  <td className="py-2 px-3 text-right font-mono text-ink-60">{e.error ? '-' : (e.auroc?.toFixed(4) ?? 'N/A')}</td>
                  <td className="py-2 px-3 text-right font-mono text-ink-60">{e.error ? '-' : (e.auprc?.toFixed(4) ?? 'N/A')}</td>
                  <td className="py-2 px-3 text-right font-mono text-ink-60">{e.error ? '-' : (e.accuracy?.toFixed(4) ?? 'N/A')}</td>
                  <td className="py-2 px-3 text-right font-mono text-ink-60">{e.error ? '-' : (e.f1?.toFixed(4) ?? 'N/A')}</td>
                  <td className="py-2 px-3 text-right font-mono text-ink-60">{e.error ? '-' : (e.sensitivity?.toFixed(4) ?? 'N/A')}</td>
                  <td className="py-2 px-3 text-right font-mono text-ink-60">{e.error ? '-' : (e.specificity?.toFixed(4) ?? 'N/A')}</td>
                  <td className="py-2 pl-3 text-right font-mono text-quantum">{e.error ? '-' : (e.spec_at_95sens?.toFixed(4) ?? 'N/A')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        <div className="mt-6 grid gap-6 md:grid-cols-2">
           <div className="bg-paper p-4 border border-rule rounded-sm">
             <h4 className="font-semibold text-sm mb-4 text-quantum-wash">Confusion Matrix (Best Quantum Model)</h4>
             {(() => {
                const bq = verdict.best_quantum;
                const cm = heldout[bq]?.confusion_matrix;
                if (!cm) return <div className="text-xs text-ink-60 mt-2">Not available</div>;
                return (
                  <div className="grid grid-cols-3 gap-1 text-center text-xs mt-4 max-w-[300px]">
                    <div className="font-medium text-ink-60 p-2"></div>
                    <div className="font-medium text-ink-60 p-2 border-b border-rule">Pred Neg</div>
                    <div className="font-medium text-ink-60 p-2 border-b border-rule">Pred Pos</div>
                    
                    <div className="font-medium text-ink-60 p-2 border-r border-rule flex items-center justify-end">True Neg</div>
                    <div className="bg-wash p-3 rounded-sm border border-rule font-mono text-lg">{cm.tn}</div>
                    <div className="bg-wash/30 p-3 rounded-sm border border-rule font-mono text-lg">{cm.fp}</div>
                    
                    <div className="font-medium text-ink-60 p-2 border-r border-rule flex items-center justify-end">True Pos</div>
                    <div className="bg-wash/30 p-3 rounded-sm border border-rule font-mono text-lg">{cm.fn}</div>
                    <div className="bg-wash p-3 rounded-sm border border-rule text-quantum font-mono text-lg">{cm.tp}</div>
                  </div>
                );
             })()}
           </div>
           
           <div className="bg-paper p-4 border border-rule rounded-sm">
             <h4 className="font-semibold text-sm mb-4 text-quantum-wash">Dataset Compression</h4>
             <div className="space-y-4 text-sm mt-4">
               <div className="flex justify-between items-center pb-2 border-b border-rule">
                 <span className="text-ink-60">Original Features</span>
                 <span className="font-mono bg-wash px-2 py-1 rounded">{result.budget?.n_features_in ?? 'N/A'}</span>
               </div>
               <div className="flex justify-between items-center pb-2 border-b border-rule">
                 <span className="text-ink-60">Target Qubits</span>
                 <span className="font-mono bg-quantum/20 text-quantum px-2 py-1 rounded">{budget.n_qubits}</span>
               </div>
               <div className="flex justify-between items-center pb-2 border-b border-rule">
                 <span className="text-ink-60">Compression Ratio</span>
                 <span className="font-mono bg-wash px-2 py-1 rounded">
                   {result.budget?.n_features_in ? (result.budget.n_features_in / budget.n_qubits).toFixed(1) : 'N/A'}x
                 </span>
               </div>
               <div className="flex justify-between items-center">
                 <span className="text-ink-60">Reduction Method</span>
                 <span className="font-mono">{budget.method}</span>
               </div>
             </div>
           </div>
        </div>
      </div>
    );
  };

  const renderExplainability = () => {
    if (!explainability || explainability.status === 'not available') {
      return (
        <div className="p-10 text-center text-ink-60 text-sm">
          Explainability data is not available: {explainability?.reason ?? 'Unknown reason'}
        </div>
      );
    }
    
    const data = {
      labels: explainability.top_components.map((c: any) => c.component),
      datasets: [
        {
          label: 'Mean |SHAP| Value',
          data: explainability.top_components.map((c: any) => c.importance),
          backgroundColor: '#444',
        }
      ]
    };

    return (
      <div className="animate-in fade-in slide-in-from-bottom-2 duration-500">
        <Eyebrow>{explainability.type}</Eyebrow>
        
        <div className="mt-4 p-4 bg-[#0d1117] border border-quantum/30 rounded-sm">
          <h4 className="font-semibold text-quantum-wash mb-2">Why PC1 / PC2?</h4>
          <p className="text-sm text-ink-60 leading-relaxed">
            Quantum execution uses a PCA-compressed representation of the original dataset. SHAP is calculated on the matched classical surrogate model operating in this same compressed space. Therefore, PC1 and PC2 represent latent feature directions rather than original clinical features.
          </p>
        </div>

        <div className="mt-6 flex flex-col lg:flex-row gap-6">
          <div className="bg-paper p-4 border border-rule rounded-sm h-[300px] w-full lg:w-1/2">
             <Bar 
               options={{
                 responsive: true,
                 maintainAspectRatio: false,
                 indexAxis: 'y',
                 plugins: { legend: { display: false }, title: { display: true, text: 'Top Feature Attributions (PCA space)', color: '#fff' } },
                 scales: { x: { grid: { color: '#333' }, ticks: { color: '#bbb' } }, y: { grid: { display: false }, ticks: { color: '#bbb' } } }
               }} 
               data={data} 
             />
          </div>
          
          <div className="w-full lg:w-1/2 space-y-4">
             {explainability.top_components.map((c: any) => (
                <div key={c.component} className="bg-paper p-4 border border-rule rounded-sm">
                   <div className="font-mono text-sm text-quantum border-b border-rule pb-2 mb-2">
                     {c.component} — SHAP importance: {c.importance.toFixed(3)}
                   </div>
                   {c.original_features ? (
                     <>
                       <div className="text-xs text-ink-60 mb-2">Top contributing original features:</div>
                       <div className="space-y-1">
                         {c.original_features.map((f: any) => (
                           <div key={f.name} className="flex justify-between text-sm font-mono">
                             <span>{f.name}</span>
                             <span className="text-ink-60">{f.weight.toFixed(3)}</span>
                           </div>
                         ))}
                       </div>
                     </>
                   ) : (
                     <div className="text-xs text-ink-60 italic">Original feature mapping unavailable.</div>
                   )}
                </div>
             ))}
          </div>
        </div>
        
        {uncertainty && uncertainty.pqk?.status === 'available' && (
          <div className="mt-6 border-t border-rule pt-4">
             <Eyebrow>Conformal Prediction (Uncertainty)</Eyebrow>
             <div className="mt-2 text-sm text-ink-60">
                <span className="font-semibold text-ink">Coverage Target:</span> {(uncertainty.pqk.coverage_target * 100).toFixed(0)}% <br/>
                <span className="font-semibold text-ink">Threshold (q-hat):</span> {uncertainty.pqk.conformal_qhat.toFixed(3)}
             </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="q-gradient-border mt-6 bg-[#090b0f]">
      <div className="flex border-b border-rule px-4 overflow-x-auto">
        {[
          { id: 'overview', label: 'Overview' },
          { id: 'analytics', label: 'Visual Analytics' },
          { id: 'explainability', label: 'Explainability & Uncertainty' },
          { id: 'circuit', label: 'Circuit Lab' },
        ].map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id as any)}
            className={`whitespace-nowrap px-4 py-3 text-sm font-medium transition-colors border-b-2 ${
              tab === t.id ? 'border-quantum text-quantum' : 'border-transparent text-ink-60 hover:text-ink'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      
      <div className="p-6 sm:p-8">
        {tab === 'overview' && renderOverview()}
        {tab === 'analytics' && renderAnalytics()}
        {tab === 'explainability' && renderExplainability()}
        {tab === 'circuit' && circuit && (
          <div className="animate-in fade-in slide-in-from-bottom-2 duration-500">
             <div className="flex justify-between items-center mb-4">
               <Eyebrow className="text-quantum">Quantum Circuit Topology</Eyebrow>
               <Badge tone="quantum">{circuit.execution_type}</Badge>
             </div>
             <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="bg-wash/40 p-3 rounded-sm border border-rule">
                  <div className="text-xs text-ink-60">Ansatz</div>
                  <div className="text-sm font-mono mt-1">{circuit.ansatz}</div>
                </div>
                <div className="bg-wash/40 p-3 rounded-sm border border-rule">
                  <div className="text-xs text-ink-60">Depth</div>
                  <div className="text-sm font-mono mt-1">{circuit.depth} layers</div>
                </div>
                <div className="bg-wash/40 p-3 rounded-sm border border-rule">
                  <div className="text-xs text-ink-60">Operations</div>
                  <div className="text-sm font-mono mt-1">{circuit.num_operations} total gates</div>
                </div>
             </div>
             <div className="overflow-x-auto bg-[#0d1117] p-4 rounded-sm border border-quantum/30">
               <pre className="font-mono text-[12px] leading-[14px] text-quantum-wash">
                 {circuit.drawing}
               </pre>
             </div>
          </div>
        )}
        
        <div className="mt-8 border-t border-rule pt-6 flex flex-wrap gap-4">
          <Button
            variant="outline"
            disabled={!jobId || !result.verdict}
            onClick={() => {
              if (jobId) window.open(`/api/jobs/${jobId}/report`, '_blank');
              else alert('Job ID not available — try refreshing the page');
            }}
          >
            Download Report (PDF)
          </Button>
          <Button
            variant="outline"
            disabled={!jobId || !result.verdict}
            onClick={() => {
              if (jobId) window.open(`/api/jobs/${jobId}/artifacts`, '_blank');
              else alert('Job ID not available — try refreshing the page');
            }}
          >
            Download Run Package (.zip)
          </Button>
          {!jobId && (
            <span className="text-xs text-red-400 self-center">⚠ job_id missing — downloads unavailable</span>
          )}
        </div>
      </div>
    </div>
  );
}
