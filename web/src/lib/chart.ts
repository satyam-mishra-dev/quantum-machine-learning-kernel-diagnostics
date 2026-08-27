// Chart palette — validated with the dataviz six-checks script against paper
// (#fbfaf7): CVD separation, chroma floor, normal-vision floor, contrast all pass.
export const SERIES_COLORS: Record<string, string> = {
  qsvm: '#6d28d9', // quantum accent
  vqc: '#6d28d9',
  logreg: '#b45309',
  svm_rbf: '#1f4fd8',
  random_forest: '#0f8a55',
};
export const CLASSICAL_BAR = '#6b6c6f'; // muted — identity carried by axis labels
export const QUANTUM_BAR = '#6d28d9';
export const SHAP_UP = '#b3232e'; // pushes risk up (debit carmine)
export const SHAP_DOWN = '#1f4fd8'; // pushes risk down (action blue)

export const TOOLTIP_STYLE = {
  background: 'var(--color-paper)',
  border: '1px solid var(--color-rule)',
  borderRadius: 4,
  fontFamily: 'var(--font-mono)',
  fontSize: 12,
};

export const AXIS_TICK = { fill: 'var(--color-ink-60)', fontSize: 11, fontFamily: 'var(--font-mono)' };
