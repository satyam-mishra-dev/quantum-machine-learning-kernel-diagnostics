import pandas as pd
import numpy as np
import io

def load_csv(path_or_buf, target_column: str):
    df = pd.read_csv(path_or_buf)
    
    # Validation 1: Empty file
    if df.empty:
        raise ValueError("Dataset is empty.")
    
    # Validation 2: Target missing
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column!r} not found.")

    original_rows = len(df)
    original_cols = len(df.columns)
    
    # Missing values profiling
    missing_total = df.isna().sum().sum()
    missing_pct = (missing_total / (original_rows * original_cols)) * 100
    missing_per_col = df.isna().sum().to_dict()
    
    # Duplicates profiling
    duplicates = int(df.duplicated().sum())
    
    # Target validation
    df = df.dropna(subset=[target_column])
    target_missing = original_rows - len(df)
    
    y_raw = df.pop(target_column)
    classes = y_raw.unique()
    if len(classes) != 2:
        raise ValueError(f"Target must be binary. Found {len(classes)} classes.")
    
    # Class distribution
    class_counts = y_raw.value_counts().to_dict()
    c1, c2 = list(class_counts.values())
    imbalance_ratio = max(c1, c2) / min(c1, c2) if min(c1, c2) > 0 else float('inf')
    
    classes_sorted = sorted(classes)
    y = (y_raw == classes_sorted[1]).astype(int).to_numpy()
    
    # Feature types
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(exclude='number').columns
    
    # Constant features
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    
    # Transformation Log
    # For phase 1 MVP, we will still drop non-numeric to keep the pipeline intact,
    # but we will loudly log it.
    df_clean = df.dropna()
    na_dropped = len(df) - len(df_clean)
    
    X = df_clean.select_dtypes("number")
    dropped_non_numeric = list(set(df.columns) - set(X.columns))
    
    transform_log = {
        "original_rows": original_rows,
        "rows_removed_missing_target": target_missing,
        "rows_removed_missing_features": na_dropped,
        "final_rows": len(X),
        "original_features": original_cols - 1,
        "dropped_non_numeric": dropped_non_numeric,
        "final_features": X.shape[1]
    }
    
    profile = {
        "rows": original_rows,
        "columns": original_cols,
        "target_column": target_column,
        "classes": len(classes),
        "missing_pct": missing_pct,
        "missing_per_col": missing_per_col,
        "duplicates": duplicates,
        "numerical_features": len(num_cols),
        "categorical_features": len(cat_cols),
        "constant_features": len(constant_cols),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "imbalance_ratio": imbalance_ratio,
        "modality": "TABULAR"
    }

    return X.to_numpy(dtype=float), y[df_clean.index] if na_dropped > 0 else y, {
        "name": "custom",
        "title": "Uploaded dataset",
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "positive_class": str(classes_sorted[1]),
        "profile": profile,
        "transform_log": transform_log
    }

# Test
csv_data = io.StringIO('''age,salary,city,disease
25,50000,NY,1
30,60000,,0
,70000,SF,1
40,,SF,0
25,50000,NY,1
50,80000,LA,
''')

X, y, meta = load_csv(csv_data, 'disease')
import json
print(json.dumps(meta, indent=2))
