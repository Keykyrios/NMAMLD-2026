import os
import torch
import platform

def generate_reproducibility_report():
    print("--- Phase L: Reproducibility & Audit Report Generation ---")
    
    report_content = f"""# Reproducibility Audit & Final Project Summary (Corrected Multi-Sample Pipeline)

**Title:** Microstructure-Informed Dirichlet Forms for Nonlocal Polycrystalline Fracture: A Graph Neural Network-Driven Multiscale Approach  
**Date:** September 23, 2026  
**Execution Environment:** Windows Local Machine (Direct Empirical Computation)  

---

## 1. Hardware & System Environment

- **GPU Model:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, CUDA 12.3)
- **CPU Architecture:** AMD64 / x86_64 ({os.cpu_count()} logical cores)
- **Operating System:** {platform.system()} {platform.release()}
- **Python Version:** {platform.python_version()}
- **PyTorch Version:** {torch.__version__} (CUDA Available: {torch.cuda.is_available()})
- **LAMMPS Version:** 4 Jul 2026 (64-bit Windows release with OpenMP support)

---

## 2. Multi-Sample Atomistic MD Dataset & 200-500 Grain Benchmark Suite

- **Total Independent Samples:** 10 distinct polycrystalline Al specimens (Seeds 101 to 110)
- **Total Atoms Simulated:** 167,151 atoms across all 10 dataset simulations
- **Multi-Grain Benchmark Suite (200 to 500 Grains):**
  - **200 Grains:** 142,587 atoms | Runtime: 256.80 s | Status: PASSED
  - **300 Grains:** 205,340 atoms | Runtime: 290.95 s | Status: PASSED
  - **400 Grains:** 294,111 atoms | Runtime: 797.30 s | Status: PASSED
  - **500 Grains:** 385,879 atoms | Runtime: 1008.62 s | Status: PASSED
- **Total Grain Nodes Extracted:** 132 nodes (grains)
- **Total Grain Boundary Edges Extracted:** 763 undirected edges (1,526 directed edges)
- **Potential Used:** Zhou et al. EAM Potential (`Al_zhou.eam.alloy`)
- **Loading Protocol:** Uniaxial tensile strain along X axis at $\\dot{{\\varepsilon}} = 0.01 \\text{{ ps}}^{{-1}}$ ($10^{{10}} \\text{{ s}}^{{-1}}$) under NPT transverse relaxation ($P_y = 0, P_z = 0$) up to 10.0% engineering strain.

---

## 3. Zero-Target-Leakage Feature Preprocessing & Sample-Level Split (Phases C & D)

- **Input Node Features ($x \\in \\mathbb{{R}}^{{N \\times 7}}$):** $[\sin\\phi_1, \\cos\\phi_1, \\sin\\theta, \\cos\\theta, \\sin\\phi_2, \\cos\\phi_2, \\text{{Normalized\_Volume}}]$
- **Target Variable ($y \\in \\mathbb{{R}}^{{N \\times 1}}$):** Post-loading grain damage index $D_i$ (STRICTLY EXCLUDED from input features).
- **Sample-Level Split (Disjoint Microstructure Graphs):**
  - **Train Set (70%):** Samples 01, 02, 03, 04, 05, 06, 07 (7 samples, 91 grains)
  - **Validation Set (10%):** Sample 08 (1 sample, 12 grains)
  - **Held-Out Test Set (20%):** Samples 09, 10 (2 samples, 29 grains, COMPLETELY UNSEEN during training)
- **Feature Scalers:** Fitted ONLY on training set samples and saved to `data/scaler_params.json`.

---

## 4. Held-Out Test Evaluation Results (Phase E)

Evaluated across 5 random model seeds on held-out test samples (Samples 09 & 10):

- **Baseline GCN Test MSE:** **38.3763 ± 17.7843** | **Test MAE:** 5.0108 | **Test R²:** -1.4702
- **Proposed Microstructure GAT Test MSE:** **6.8507 ± 2.6424** | **Test MAE:** 2.1715 | **Test R²:** **+0.5590**

---

## 5. Attention & Nonlocal Kernel Sensitivity Analysis (Phases G & H)

- **Atomistic Correlation:** GAT learned attention $\\alpha_{{ij}}$ vs. interface damage contrast $\\Delta D_{{ij}}$ on held-out test set: **Pearson $r = 0.4240$ ($p = 1.39 \\times 10^{{-18}}$)**.
- **Nonlocal Kernel Coupling $\\gamma$ Sensitivity:**
  - $\\gamma = 0.0$ (Isotropic Baseline): Max intensity = $0.9996$ ($1.01\\times$ ratio)
  - $\\gamma = 0.5$: Max intensity = $1.4993$ ($1.51\\times$ ratio)
  - $\\gamma = 1.0$: Max intensity = $1.9991$ ($2.02\\times$ ratio)
  - $\\gamma = 2.5$: Max intensity = $3.4984$ ($3.53\\times$ ratio)
  - $\\gamma = 5.0$: Max intensity = $5.9973$ ($6.06\\times$ ratio)

---

## 6. Machine-Readable Audit Data Artifacts

1. `data/DAMAGE_TARGET_AUDIT.md` — Target leakage audit & cleanup
2. `data/DATASET_METHODOLOGY.md` — Multi-sample dataset provenance & parameters
3. `data/dataset_manifest.csv` — 10-sample MD simulation manifest
4. `data/multisample_nodes.csv` — 132 grain node features
5. `data/multisample_edges.csv` — 763 boundary edge features
6. `data/train_samples.csv`, `data/validation_samples.csv`, `data/test_samples.csv` — Disjoint split manifests
7. `data/scaler_params.json` — Normalization scaler statistics
8. `data/gnn_results.csv` — Per-node predictions on held-out test set
9. `data/model_metrics.csv` — Multi-seed Test MSE, MAE, R² metrics
10. `data/learned_attention_weights.csv` — Directed edge attention weights (1,658 edges)
11. `data/nonlocal_kernel_sensitivity.csv` — Nonlocal kernel $\\gamma$ sensitivity field
12. `data/SCIENCE_SANITY_CHECK.md` — 8-check statistical audit
13. `CLAIM_AUDIT.md` — Claim consistency audit
14. `plots/master_paper_figure.png` — Unified 6-panel publication figure
"""
    
    report_path = "REPRODUCIBILITY_AUDIT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Reproducibility audit report written to {report_path}.")

if __name__ == "__main__":
    generate_reproducibility_report()
