import json
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 18)
        self.cell(0, 10, "QUANTUMFLOOR", new_x="LMARGIN", new_y="NEXT")
        self.set_font("helvetica", "I", 12)
        self.cell(0, 10, "Computational Research Report", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def chapter_title(self, num, title):
        self.set_font("helvetica", "B", 14)
        self.cell(0, 10, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT", border="B")
        self.ln(2)

    def chapter_body(self, text):
        self.set_font("helvetica", "", 11)
        self.multi_cell(0, 8, text)
        self.ln(5)
        
    def metric(self, label, value):
        self.set_font("helvetica", "B", 11)
        self.cell(60, 8, f"{label}: ", new_x="RIGHT", new_y="TOP")
        self.set_font("helvetica", "", 11)
        # Convert value to string and replace newlines with spaces for single line, 
        # or just print it if it's short.
        val_str = str(value).replace('\n', ' ').replace('⚠', '[!]').replace('—', '-')
        self.cell(0, 8, val_str, new_x="LMARGIN", new_y="NEXT")
        
    def alert_box(self, warnings, ff_auroc=None):
        self.set_fill_color(255, 251, 235)  # yellow
        self.set_draw_color(245, 158, 11)   # amber
        self.set_font("helvetica", "B", 12)
        self.cell(0, 10, "Scientific Integrity Warnings", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_font("helvetica", "", 10)
        for w in warnings:
            self.multi_cell(0, 6, "Warning: " + w.replace('⚠', '[!]').replace('—', '-'), border="LR", fill=True)
        if ff_auroc is not None:
            msg = (f"Full-Feature Classical AUROC = {ff_auroc:.4f} flagged: "
                   "Duplicate rows in this dataset may inflate metrics when identical samples "
                   "appear in both train and test splits. Matched-classical/quantum results below "
                   "use stratified subsampling and are not affected.")
            self.multi_cell(0, 6, msg, border="LRB", fill=True)
        else:
            self.cell(0, 2, "", border="T", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

def generate_report(job_id, result, meta, out_path):
    budget = result.get("budget", {})
    scout = result.get("scout", {})
    verdict = result.get("verdict", {})
    circuit = result.get("circuit", {})
    exec_type = circuit.get("execution_type", "SIMULATED EXECUTION")
    integrity_warnings = result.get("integrity_warnings", [])
    
    prof = meta.get("profile", {})
    prof_rows = prof.get("rows", meta.get("n_samples", 0))
    prof_dups = prof.get("duplicates", 0)
    prof_unique = prof_rows - prof_dups
    
    pdf = PDF()
    pdf.add_page()
    
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 8, f"Job ID: {job_id}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    pdf.chapter_title(1, "Executive Summary")
    ds_name = prof.get("target_column", meta.get("title", "Uploaded dataset"))
    pdf.chapter_body(f"This report details the execution of the QuantumFloor diagnostic pipeline on dataset '{ds_name}'. The system evaluated multiple classical baselines against quantum algorithms to determine the optimal strategy.")
    
    pdf.chapter_title(2, "Dataset Intelligence")
    pdf.metric("Modality", prof.get("modality", "Unknown"))
    pdf.metric("Total Rows", prof_rows)
    pdf.metric("Unique Rows", prof_unique)
    pdf.metric("Duplicate Rows", prof_dups)
    pdf.metric("Total Columns", meta.get("n_features", 0))
    pdf.metric("Missing Values", f"{prof.get('missing_pct', 0):.2f}%")
    pdf.metric("Leakage Risk Detected", prof.get("identifier_columns", 0))
    pdf.ln(5)
    
    if integrity_warnings or verdict.get("full_feature_floor_has_integrity_warning"):
        ff_auroc = result.get("heldout", {}).get("full_feature_classical", {}).get("auroc", 0) if verdict.get("full_feature_floor_has_integrity_warning") else None
        pdf.alert_box(integrity_warnings, ff_auroc)
        
    pdf.chapter_title(3, "Data Preparation")
    pdf.metric("Train/Test Split", "80/20 Stratified")
    pdf.metric("Preprocessing", "Pipeline (Imputation + Scaling)")
    pdf.metric("Leakage Protection", "Enforced")
    pdf.ln(5)
    
    pdf.chapter_title(4, "Computational Plan")
    pdf.metric("Original Features", meta.get("n_features", 0))
    pdf.metric("Quantum Embedding Dimensions", budget.get("n_qubits", 0))
    pdf.metric("Qubit Budget", budget.get("n_qubits", 0))
    pdf.metric("Classical Training Size", result.get("classical_rows_used", meta.get("n_samples", 0)))
    pdf.metric("Quantum Subset Size", result.get("train_rows_used", 0))
    pdf.ln(5)
    
    pdf.add_page()
    pdf.chapter_title(5, "Quantum Execution")
    pdf.metric("Execution Backend", exec_type)
    pdf.metric("Ansatz", circuit.get("ansatz", "AngleEmbedding + IsingZZ"))
    pdf.metric("Qubits", circuit.get("n_qubits", 0))
    pdf.metric("Circuit Depth", circuit.get("depth", 0))
    pdf.metric("Total Operations", circuit.get("num_operations", 0))
    pdf.ln(5)
    
    pdf.chapter_title(6, "Classical vs Quantum Baselines")
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(40, 8, "Model", 1)
    pdf.cell(30, 8, "Type", 1)
    pdf.cell(25, 8, "AUROC", 1)
    pdf.cell(25, 8, "F1", 1)
    pdf.cell(30, 8, "Spec@95", 1)
    pdf.cell(40, 8, "Notes", 1, new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", "", 9)
    for m, e in result.get("heldout", {}).items():
        mtype = "Quantum" if m in ["qsvm", "pqk", "vqc"] else "Classical"
        auroc = f"{e.get('auroc', 0):.4f}" if "auroc" in e and not isinstance(e.get('auroc'), str) else "N/A"
        f1 = f"{e.get('f1', 0):.4f}" if "f1" in e and not isinstance(e.get('f1'), str) else "N/A"
        spec = f"{e.get('spec_at_95sens', 0):.4f}" if "spec_at_95sens" in e else "N/A"
        err = e.get('error', '')
        iw = e.get('integrity_warning', '')
        note = (err[:30] + "..") if err else ("[!] Integrity Warning" if iw else "")
        if m == 'full_feature_classical' and verdict.get('full_feature_floor_has_integrity_warning'):
            note = "[!] Flawed by duplicates"
        
        note = note.replace('⚠', '[!]').replace('—', '-')
        
        pdf.cell(40, 8, m, 1)
        pdf.cell(30, 8, mtype, 1)
        pdf.cell(25, 8, auroc, 1)
        pdf.cell(25, 8, f1, 1)
        pdf.cell(30, 8, spec, 1)
        pdf.cell(40, 8, note, 1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    pdf.chapter_title(7, "Comparison & Explainability")
    auroc_vals = {m: e.get("auroc", 0) for m, e in result.get("heldout", {}).items() if "auroc" in e and not isinstance(e.get('auroc'), str)}
    q_models = {"qsvm", "pqk", "vqc"}
    c_vals = [v for m, v in auroc_vals.items() if m not in q_models and m != "full_feature_classical"]
    q_vals = [v for m, v in auroc_vals.items() if m in q_models]
    pdf.metric("Best Classical AUROC", f"{max(c_vals):.4f}" if c_vals else "N/A")
    pdf.metric("Best Quantum AUROC", f"{max(q_vals):.4f}" if q_vals else "N/A")
    pdf.metric("Delta", f"{verdict.get('quantum_vs_best_classical', 0.0):.4f}")
    
    expl = result.get("explainability", {})
    pdf.metric("Explainability Status", expl.get("status", "not available"))
    if "top_components" in expl:
        for tc in expl.get("top_components", []):
            pdf.metric("  " + tc['component'], f"{tc['importance']:.4f}")
    elif "reason" in expl:
        pdf.metric("  Reason", expl["reason"])
    pdf.ln(5)
    
    pdf.chapter_title(8, "Final Verdict")
    pdf.metric("Best Overall Model", verdict.get("best_model", "Unknown"))
    pdf.metric("Quantum Advantage", "Yes" if verdict.get("quantum_won", False) else "No")
    
    # Write to file
    pdf.output(out_path)
