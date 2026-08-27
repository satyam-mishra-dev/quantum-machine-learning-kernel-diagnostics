import { useEffect, useState } from 'react';

// ---- API shapes (mirror qnidaan/app.py) -----------------------------------

export type DatasetCard = {
  name: string;
  title: string;
  n_qubits: number;
  best_model: string;
  best_auroc: number;
  qsvm_auroc: number | null;
  quantum_worth_trying: boolean;
};

export type ModelEval = {
  auroc: number;
  accuracy: number;
  spec_at_95sens?: number;
  train_seconds?: number;
  conformal_qhat?: number;
};

export type Scout = {
  kta_quantum: number;
  kta_classical: number;
  geometric_difference: number;
  quantum_worth_trying: boolean;
  verdict?: string; // "advantage_possible" | "classical_will_match"
};

export type RunDetail = {
  dataset: { name: string; title: string; n_features: number; n_train?: number; n_test?: number; prevalence?: number };
  budget: { n_qubits: number; method: string; explained_variance: number | null };
  scout: Scout;
  heldout: Record<string, ModelEval>;
  sample_efficiency?: Array<
    { n_train: number } & Record<string, { auroc_mean: number; auroc_std: number } | number>
  >;
};

export type Prediction = {
  risk_probability: number;
  prediction: number;
  conformal_set: number[];
  confident: boolean;
  flag_for_review: boolean;
  coverage_guarantee: string;
  top_features: { feature: string; shap: number }[];
  explanation_source: string;
  model: string;
  dataset: string;
};

export type Verdict = {
  best_model: string;
  quantum_won: boolean;
  qsvm_vs_best_classical: number;
};

export type JobStatus = {
  status: 'running' | 'done' | 'error';
  error?: string;
  dataset?: { title: string; n_features: number };
  result?: {
    budget: RunDetail['budget'];
    scout: Scout;
    heldout: Record<string, ModelEval>;
    verdict: Verdict;
  };
};

// ---- fetch helpers --------------------------------------------------------

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = String(body.detail);
    } catch {
      /* keep status text */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

/** Tiny data hook: loading / error / data. Refetches when path changes. */
export function useApi<T>(path: string | null): {
  data: T | null;
  error: string | null;
  loading: boolean;
} {
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>({
    data: null,
    error: null,
    loading: !!path,
  });
  useEffect(() => {
    if (!path) return;
    let alive = true;
    setState({ data: null, error: null, loading: true });
    api<T>(path)
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((e: Error) => alive && setState({ data: null, error: e.message, loading: false }));
    return () => {
      alive = false;
    };
  }, [path]);
  return state;
}

export const MODEL_LABELS: Record<string, string> = {
  logreg: 'Logistic Regression',
  svm_rbf: 'SVM (RBF)',
  random_forest: 'Random Forest',
  qsvm: 'QSVM',
  vqc: 'VQC',
};

export const isQuantum = (m: string): boolean => m === 'qsvm' || m === 'vqc';
