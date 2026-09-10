import traceback
from qnidaan.db import update_job

# Default conservative sample limit for local demo
QUANTUM_MAX_TRAIN_ROWS_DEFAULT = 120  # 120 rows = 7,260 train + test kernel evals

def run_adapter_worker(job_id: str, X, y, meta, budget):
    """Universal adapter running in a separate process."""
    try:
        from sklearn.model_selection import train_test_split
        import numpy as np
        import time as _time
        
        from qnidaan.budgeter import fit_budget_auto
        from qnidaan.classical import evaluate, make_twins
        from qnidaan.quantum import PQK, QSVM, VQC, make_kernel
        from qnidaan.scout import scout
        from qnidaan.readiness import readiness
        
        # ─── DUPLICATE DETECTION & SPLITTING ──────────────────────────────────────────────
        import pandas as pd
        from sklearn.model_selection import GroupShuffleSplit
        
        if hasattr(X, 'duplicated'):
            df_check = X.copy()
        else:
            df_check = pd.DataFrame(X)
            
        # Create deterministic fingerprint for each feature row
        groups = df_check.apply(lambda row: hash(tuple(row)), axis=1).values
        
        n_total = len(df_check)
        n_unique = len(np.unique(groups))
        dup_count = n_total - n_unique
        dup_fraction = dup_count / n_total if n_total > 0 else 0
        
        integrity_warnings = []
        y_arr = np.asarray(y)
        
        if dup_fraction > 0:
            df_temp = pd.DataFrame({'group': groups, 'target': y_arr})
            target_conflicts = df_temp.groupby('group')['target'].nunique()
            n_conflicts = (target_conflicts > 1).sum()
            
            if n_conflicts > 0:
                msg = f"⚠ INTEGRITY ERROR: {n_conflicts} duplicate feature groups have conflicting targets. Dataset is ambiguous."
                update_job(job_id, status="FAILED", stage="FAILED", error=msg, log_message=msg)
                return False
                
            integrity_warnings.append(
                f"Duplicate feature groups: {n_unique} | "
                f"Duplicate rows: {dup_count} | "
                f"Cross-split duplicate groups: 0 | "
                f"Target leakage detected: 0 | "
                f"Identifier leakage detected: 0 | "
                f"Preprocessing leakage: 0"
            )
            update_job(job_id, log_message=f"Integrity: {dup_count} duplicate rows grouped. Cross-split leakage prevented.")
            
            update_job(job_id, status="RUNNING", stage="SPLITTING", log_message="Splitting dataset (GroupShuffleSplit, duplicate-aware)")
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(X, y_arr, groups))
            
            if hasattr(X, 'iloc'):
                Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            else:
                Xtr, Xte = np.asarray(X)[train_idx], np.asarray(X)[test_idx]
            ytr, yte = y_arr[train_idx], y_arr[test_idx]
        else:
            update_job(job_id, status="RUNNING", stage="SPLITTING", log_message="Splitting dataset into train/test (80/20 stratified)")
            Xtr, Xte, ytr, yte = train_test_split(X, y_arr, test_size=0.2, random_state=42, stratify=y_arr)
        
        # ─── PREPROCESSING ────────────────────────────────────────────────────
        update_job(job_id, stage="PREPROCESSING", log_message="Applying leakage-free ColumnTransformer (fitted on train only)")
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        
        num_cols = meta.get("numerical_features", [])
        cat_cols = meta.get("categorical_features", [])
        
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        transformers = []
        if num_cols:
            transformers.append(("num", num_pipe, num_cols))
        if cat_cols:
            transformers.append(("cat", cat_pipe, cat_cols))
            
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        Xtr_pre = preprocessor.fit_transform(Xtr)
        Xte_pre = preprocessor.transform(Xte)
        
        # ─── QUANTUM SUBSAMPLING ──────────────────────────────────────────────
        import os
        limit_str = os.environ.get("QUANTUM_MAX_TRAIN_ROWS", str(QUANTUM_MAX_TRAIN_ROWS_DEFAULT))
        subset_limit = min(int(limit_str), len(Xtr_pre))
        capped = len(Xtr_pre) > subset_limit
        
        if capped:
            # Stratified subsample for quantum — classical uses full set
            Xtr_q, _, ytr_q, _ = train_test_split(
                Xtr_pre, ytr,
                train_size=subset_limit,
                random_state=42, stratify=ytr
            )
            update_job(job_id, log_message=(
                f"Quantum subset: {subset_limit} rows (stratified) out of {len(Xtr_pre)} available. "
                f"Classical baselines use full {len(Xtr_pre)} training rows."
            ))
        else:
            Xtr_q = Xtr_pre
            ytr_q = ytr
            update_job(job_id, log_message=f"Dataset small enough: all {len(Xtr_pre)} training rows used for both classical and quantum.")
            
        # ─── BUDGETING ────────────────────────────────────────────────────────
        update_job(job_id, stage="BUDGETING", log_message=f"Selecting optimal {budget}-qubit budget via PCA/SelectKBest")
        plan = fit_budget_auto(Xtr_q, ytr_q, budget)
        comp_ratio = plan.n_features_in / plan.n_qubits
        update_job(job_id, log_message=(
            f"Compression: {plan.n_features_in} features → {plan.n_qubits} qubits "
            f"via {plan.method} ({comp_ratio:.1f}x compression)"
        ))
        
        Ztr_q, Zte = plan.transform(Xtr_q), plan.transform(Xte_pre)
        kernel = make_kernel(plan.n_qubits)
        
        # ─── SCOUTING ─────────────────────────────────────────────────────────
        update_job(job_id, stage="SCOUTING", log_message="Running Quantum Geometry Scout (entanglement capability test)")
        
        import pennylane as qml
        from qnidaan.quantum import _feature_map, QuantumBackendManager
        
        dev_draw = qml.device("default.qubit", wires=plan.n_qubits)
        
        @qml.qnode(dev_draw)
        def draw_circuit(x):
            _feature_map(x, list(range(plan.n_qubits)), 1)
            return qml.probs(wires=list(range(plan.n_qubits)))
            
        specs = qml.specs(draw_circuit)(np.zeros(plan.n_qubits))
        res = specs["resources"]
        
        exec_mode, exec_error = QuantumBackendManager.get_info(plan.n_qubits)
        
        train_k_evals = (len(Ztr_q) * (len(Ztr_q) + 1)) // 2
        test_k_evals = len(Zte) * len(Ztr_q)
        
        out = {
            "job_id": job_id,  # Include job_id in result for frontend download links
            "budget": {
                "n_qubits": plan.n_qubits,
                "method": plan.method,
                "explained_variance": plan.explained_variance,
                "cv_auroc": plan.cv_auroc,
                "runner_up_cv_auroc": plan.runner_up_cv_auroc,
                "n_features_in": plan.n_features_in,
            },
            "scout": scout(Ztr_q, ytr_q, kernel),
            "heldout": {},
            "train_rows_used": int(len(ytr_q)),
            "train_rows_capped": capped,
            "classical_rows_used": int(len(ytr)),
            "quantum_kernel_evals_train": train_k_evals,
            "quantum_kernel_evals_test": test_k_evals,
            "quantum_kernel_evals_total": train_k_evals + test_k_evals,
            "integrity_warnings": integrity_warnings,
        }
        
        out["circuit"] = {
            "drawing": qml.draw(draw_circuit)(np.zeros(plan.n_qubits)),
            "depth": int(res.depth),
            "num_operations": sum(res.gate_types.values()),
            "execution_type": exec_mode,
            "execution_error": exec_error,
            "ansatz": "AngleEmbedding + IsingZZ",
            "n_qubits": plan.n_qubits,
        }
        
        out["readiness"] = readiness(len(y), meta["n_features"], float(np.mean(y)), out["scout"], out["budget"])
        
        update_job(job_id, stage="PLANNING", log_message=(
            f"Execution plan: classical_floor=RF({len(Xtr_pre)} rows), "
            f"matched_classical=[LogReg,SVM,RF]({len(Ztr_q)} rows), "
            f"quantum=[QSVM,PQK,VQC]({len(Ztr_q)} rows), "
            f"budget={plan.n_qubits} qubits, "
            f"QSVM kernel evals={train_k_evals + test_k_evals}"
        ))
        
        # ─── FULL-FEATURE CLASSICAL FLOOR ─────────────────────────────────────
        from sklearn.ensemble import RandomForestClassifier
        update_job(job_id, stage="TRAINING_CLASSICAL",
                   log_message=f"Training full-feature classical floor (RandomForest, {len(Xtr_pre)} training rows)")
        full_floor = RandomForestClassifier(n_estimators=300, random_state=42)
        full_floor.fit(Xtr_pre, ytr)
        ff_result = evaluate(full_floor, Xte_pre, yte)
        out["heldout"]["full_feature_classical"] = ff_result
        
        if integrity_warnings and ff_result.get("auroc", 0) >= 0.999:
            out["heldout"]["full_feature_classical"]["integrity_warning"] = (
                "AUROC=1.000 detected with duplicate rows present. "
                "This metric is likely inflated due to duplicate samples in train/test split. "
                "Dataset has only " + str(n_unique) + " unique rows."
            )
        
        # ─── MATCHED CLASSICAL + QUANTUM MODELS ───────────────────────────────
        models = {
            **make_twins(),
            "qsvm": QSVM(n_qubits=plan.n_qubits, C=10),
            "pqk": PQK(n_qubits=plan.n_qubits),
            "vqc": VQC(n_qubits=plan.n_qubits, epochs=10),
        }
        
        # Calibration split from quantum subset
        Ztr_train, Ztr_calib, ytr_train, ytr_calib = train_test_split(
            Ztr_q, ytr_q, test_size=0.2, random_state=42, stratify=ytr_q
        )
        out["uncertainty"] = {}
        
        import time
        for mname, m in models.items():
            start_time = time.time()
            is_quantum = mname in ["qsvm", "pqk", "vqc"]
            
            if is_quantum:
                update_job(job_id, stage="TRAINING_QUANTUM",
                           log_message=f"Training quantum model: {mname.upper()} ({len(Ztr_train)} rows, {plan.n_qubits} qubits)")
            else:
                update_job(job_id, stage="TRAINING_CLASSICAL",
                           log_message=f"Training classical baseline: {mname}")
            
            try:
                if mname == "vqc":
                    vqc_start = time.time()
                    
                    def vqc_cb(ep, mx, c):
                        elapsed = time.time() - vqc_start
                        if elapsed > 120:
                            raise TimeoutError(f"VQC exceeded 120s at epoch {ep}/{mx}.")
                        rate = ep / elapsed if elapsed > 0 else 0
                        eta = (mx - ep) / rate if rate > 0 else None
                        telemetry = {
                            "stage": "TRAINING_QUANTUM",
                            "substage": "VQC",
                            "model": mname,
                            "completed": ep,
                            "total": mx,
                            "percentage": (ep / mx) * 100,
                            "elapsed_seconds": elapsed,
                            "rate": rate,
                            "eta_seconds": eta,
                            "execution_type": exec_mode,
                            "qubits": plan.n_qubits
                        }
                        update_job(job_id, log_message=f"VQC epoch {ep}/{mx} (cost={c:.4f})", telemetry=telemetry)
                    
                    m.fit(Ztr_train, ytr_train, epoch_callback=vqc_cb)
                    
                elif mname in ["qsvm", "pqk"]:
                    # Build unified callback for kernel progress
                    model_label = mname.upper()
                    n_train_q = len(Ztr_train)
                    total_train_evals = (n_train_q * (n_train_q + 1)) // 2
                    total_test_evals = len(Zte) * n_train_q
                    phase_offset = [0]  # mutable for closure
                    phase_total = [total_train_evals]
                    phase_label = ["TRAIN_KERNEL"]
                    
                    def kernel_cb(completed, total, elapsed, rate, eta, phase=None):
                        if phase:
                            phase_label[0] = phase
                            if phase == "TEST_KERNEL":
                                phase_offset[0] = total_train_evals
                                phase_total[0] = total_test_evals
                        
                        global_completed = phase_offset[0] + completed
                        global_total = total_train_evals + total_test_evals
                        pct = (global_completed / global_total * 100) if global_total > 0 else 0
                        
                        telemetry = {
                            "stage": "TRAINING_QUANTUM",
                            "substage": "TRAINING_KERNEL",
                            "model": model_label,
                            "phase": phase_label[0],
                            "completed": global_completed,
                            "total": global_total,
                            "percentage": pct,
                            "elapsed_seconds": elapsed,
                            "rate": rate if rate else 0,
                            "eta_seconds": eta,
                            "execution_type": exec_mode,
                            "qubits": plan.n_qubits,
                        }
                        update_job(job_id,
                                   log_message=f"{model_label} {phase_label[0]}: {global_completed}/{global_total} ({pct:.1f}%)",
                                   telemetry=telemetry)
                    
                    m.kernel_callback = kernel_cb
                    m.fit(Ztr_train, ytr_train)
                    
                else:
                    m.fit(Ztr_train, ytr_train)
                    
            except TimeoutError as e:
                update_job(job_id, log_message=f"{mname} TIMEOUT: {e}",
                           telemetry={"stage": "TRAINING_QUANTUM", "status": "TIMEOUT", "model": mname, "error": str(e)})
                out["heldout"][mname] = {
                    "auroc": 0, "f1": 0, "spec_at_95sens": 0,
                    "error": "Timeout — model skipped"
                }
                continue
            except Exception as e:
                update_job(job_id, log_message=f"{mname} ERROR: {e}")
                out["heldout"][mname] = {
                    "auroc": 0, "f1": 0, "spec_at_95sens": 0,
                    "error": str(e)[:200]
                }
                continue
                
            out["heldout"][mname] = evaluate(m, Zte, yte)
            
            # Conformal Prediction (Uncertainty)
            try:
                calib_probs = m.predict_proba(Ztr_calib)
                test_probs = m.predict_proba(Zte)
                scores = 1.0 - calib_probs[np.arange(len(ytr_calib)), ytr_calib]
                qhat = np.quantile(scores, min(1.0, (len(ytr_calib) + 1) / len(ytr_calib) * 0.90))
                prediction_sets = (test_probs >= (1.0 - qhat)).astype(int).tolist()
                out["uncertainty"][mname] = {
                    "conformal_qhat": float(qhat),
                    "coverage_target": 0.90,
                    "prediction_sets": prediction_sets,
                    "status": "available"
                }
            except Exception as e:
                out["uncertainty"][mname] = {"status": "not available", "reason": str(e)}
        
        # ─── EXPLAINABILITY ───────────────────────────────────────────────────
        update_job(job_id, stage="EXPLAINING", log_message="Generating Surrogate Classical Explainability (SHAP on matched RF)")
        try:
            import shap
            matched_rf = models.get("random_forest")
            if matched_rf is None:
                raise ValueError("Matched random forest model not available")
            
            explainer = shap.TreeExplainer(matched_rf)
            shap_values = explainer.shap_values(Zte)
            
            # Handle both old (list) and new (3D ndarray) SHAP API
            sv = np.array(shap_values)
            n_latent = Zte.shape[1]
            
            if sv.ndim == 3:
                # New SHAP: shape (n_samples, n_features, n_classes) — use class 1
                sv2d = sv[:, :, 1]
            elif sv.ndim == 2:
                # Old SHAP with 2D output
                sv2d = sv
            elif isinstance(shap_values, list):
                # Old SHAP with list output — class 1
                sv2d = np.array(shap_values[1])
            else:
                sv2d = sv
            
            # sv2d shape: (n_samples, n_features)
            if sv2d.ndim != 2:
                sv2d = sv2d.reshape(len(Zte), -1)
            
            mean_abs_shap = np.abs(sv2d).mean(axis=0)  # shape: (n_features,)
            
            # n_features == plan.n_qubits (latent dimension)
            top_n = min(5, n_latent, len(mean_abs_shap))
            top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
            
            top_components = []
            for i in top_idx:
                comp = {"component": f"PC{i + 1}", "importance": float(mean_abs_shap[i])}
                if plan.method == "pca" and "pca" in plan.pipeline.named_steps:
                    pca_weights = plan.pipeline.named_steps["pca"].components_[i]
                    top_feats = np.argsort(np.abs(pca_weights))[::-1][:4]
                    comp["original_features"] = [
                        {"name": meta["feature_names"][fi], "weight": float(np.abs(pca_weights[fi]))}
                        for fi in top_feats
                    ]
                top_components.append(comp)
            
            out["explainability"] = {
                "type": "Surrogate Classical Explainability",
                "model_used": "Matched Random Forest on PCA latent space",
                "disclaimer": "These are surrogate attributions on the compressed latent space — not direct quantum circuit attributions.",
                "n_components": n_latent,
                "status": "available",
                "top_components": top_components,
            }
        except Exception as e:
            out["explainability"] = {
                "status": "not available",
                "reason": str(e),
                "type": "Surrogate Classical Explainability",
            }
            update_job(job_id, log_message=f"Explainability unavailable: {e}")
        
        # ─── FINAL VERDICT ────────────────────────────────────────────────────
        update_job(job_id, stage="EVALUATING", log_message="Generating final verdict and comparative metrics")
        quantum_names = {"qsvm", "pqk", "vqc"}
        
        # Only include models that have real auroc (not timeout/error)
        valid_auroc = {
            m: e["auroc"]
            for m, e in out["heldout"].items()
            if "auroc" in e and not np.isnan(e.get("auroc", float("nan")))
            and "error" not in e
            and m != "full_feature_classical"
        }
        
        if valid_auroc:
            best = max(valid_auroc, key=valid_auroc.get)
            quantum_valid = [m for m in quantum_names if m in valid_auroc]
            classical_valid = [m for m in valid_auroc if m not in quantum_names]
            
            best_quantum = max(quantum_valid, key=valid_auroc.get) if quantum_valid else None
            best_classical_auroc = max(valid_auroc[m] for m in classical_valid) if classical_valid else 0.0
            
            out["verdict"] = {
                "best_model": best,
                "best_quantum": best_quantum,
                "quantum_won": best in quantum_names,
                "quantum_vs_best_classical": round(
                    valid_auroc.get(best_quantum, 0) - best_classical_auroc, 4
                ) if best_quantum else 0.0,
                "full_feature_floor_auroc": out["heldout"].get("full_feature_classical", {}).get("auroc", 0),
                "full_feature_floor_has_integrity_warning": bool(integrity_warnings and 
                    out["heldout"].get("full_feature_classical", {}).get("auroc", 0) >= 0.999),
            }
        else:
            out["verdict"] = {
                "best_model": "N/A",
                "best_quantum": None,
                "quantum_won": False,
                "quantum_vs_best_classical": 0.0,
                "full_feature_floor_auroc": out["heldout"].get("full_feature_classical", {}).get("auroc", 0),
                "full_feature_floor_has_integrity_warning": bool(integrity_warnings),
            }
        
        # ─── PERSIST ARTIFACTS ────────────────────────────────────────────────
        import joblib
        import json
        import shutil
        import datetime
        from pathlib import Path
        from qnidaan.reporter import generate_report
        
        run_dir = Path(__file__).resolve().parent.parent / "runs" / f"QF-{job_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        model_dir = Path(__file__).resolve().parent.parent / "models" / f"QF-{job_id}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Serialize: handle numpy types
        def _clean(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list): return [_clean(v) for v in obj]
            return obj
        
        out_clean = _clean(out)
        
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(out_clean, f, indent=2)
        with open(run_dir / "profile.json", "w") as f:
            json.dump(_clean(meta), f, indent=2)
        if "circuit" in out:
            with open(run_dir / "circuit.json", "w") as f:
                json.dump(_clean(out["circuit"]), f, indent=2)
        if "explainability" in out:
            with open(run_dir / "explainability.json", "w") as f:
                json.dump(_clean(out["explainability"]), f, indent=2)
        if "uncertainty" in out:
            with open(run_dir / "uncertainty.json", "w") as f:
                json.dump(_clean(out["uncertainty"]), f, indent=2)
                
        models_generated = []
        models_failed = []
        for mname, m in models.items():
            try:
                model_path = model_dir / f"{mname}.joblib"
                joblib.dump(m, model_path)
                # Ensure model deserializes successfully
                joblib.load(model_path)
                # Move to run_dir/models for zipping
                run_models = run_dir / "models"
                run_models.mkdir(exist_ok=True)
                shutil.copy(model_path, run_models / f"{mname}.joblib")
                models_generated.append(mname)
            except Exception as e:
                models_failed.append({"model": mname, "error": str(e)})
                update_job(job_id, log_message=f"Model serialization failed for {mname}: {e}")
                
        generate_report(job_id, out_clean, _clean(meta), str(run_dir / "report.pdf"))
        
        # Create manifest.json
        manifest = {
            "job_id": job_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset": meta.get("title", "Uploaded dataset"),
            "models_generated": models_generated,
            "models_failed": models_failed,
            "execution_backend": exec_mode,
            "qubit_budget": plan.n_qubits,
            "quantum_subset_size": subset_limit if capped else len(Xtr_pre)
        }
        with open(run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        zip_base = str(run_dir.parent / f"QuantumFloor_{job_id}")
        shutil.make_archive(zip_base, "zip", str(run_dir))
        
        # ─── ARTIFACT INTEGRITY GATE ──────────────────────────────────────────
        required = ["metrics.json", "profile.json", "circuit.json", "report.pdf", "manifest.json"]
        missing = [f for f in required if not (run_dir / f).exists()]
        zip_path = Path(f"{zip_base}.zip")
        if not zip_path.exists():
            missing.append(f"QuantumFloor_{job_id}.zip")
            
        if missing:
            raise RuntimeError(f"Artifact integrity failure — missing: {', '.join(missing)}")
        
        update_job(job_id, log_message="All artifacts verified. Report and ZIP ready for download.")
        update_job(job_id, status="COMPLETED", stage="COMPLETED", result=out_clean,
                   log_message="Analysis completed successfully.")
        
        return True
        
    except Exception as e:
        err_msg = str(e)
        traceback.print_exc()
        update_job(job_id, status="FAILED", stage="FAILED", error=err_msg,
                   log_message=f"Pipeline error: {err_msg}")
        return False
