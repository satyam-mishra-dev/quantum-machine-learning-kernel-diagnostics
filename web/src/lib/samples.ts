// Hardcoded "load sample patient" values — real rows from the public datasets
// (WDBC row 0, Cleveland row 0, Pima row 0), keyed by exact feature name.

const WDBC_NAMES = [
  'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
  'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
  'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
  'area error', 'smoothness error', 'compactness error', 'concavity error',
  'concave points error', 'symmetry error', 'fractal dimension error',
  'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
  'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry',
  'worst fractal dimension',
];
const WDBC_VALUES = [
  17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
  1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
  25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
];

export const SAMPLE_PATIENTS: Record<string, Record<string, number>> = {
  wdbc: Object.fromEntries(WDBC_NAMES.map((n, i) => [n, WDBC_VALUES[i]])),
  cleveland: {
    age: 63, sex: 1, cp: 1, trestbps: 145, chol: 233, fbs: 1, restecg: 2,
    thalach: 150, exang: 0, oldpeak: 2.3, slope: 3, ca: 0, thal: 6,
  },
  pima: {
    pregnancies: 6, glucose: 148, bp: 72, skin: 35, insulin: 0,
    bmi: 33.6, pedigree: 0.627, age: 50,
  },
};
