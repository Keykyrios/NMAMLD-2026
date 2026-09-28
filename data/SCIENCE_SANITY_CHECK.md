# Scientific & Statistical Sanity Check Report

**Date:** September 23, 2026  
**Status:** ALL 8 CRITICAL SCIENTIFIC SANITY CHECKS PASSED  

---

## Summary Table

| Check ID & Description | Status | Evidence & Metrics |
|---|---|---|
| **1. NaN / Inf Free Verification** | **PASS** | Found 0 NaN/Inf values across all tabular datasets. |
| **2. Sample-Level Split Disjointness (Zero Contamination)** | **PASS** | Train={np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6), np.int64(7)}, Val={np.int64(8)}, Test={np.int64(9), np.int64(10)}. Intersections: Train-Val=set(), Train-Test=set(), Val-Test=set(). |
| **3. Target Leakage Elimination in Predictor Vector** | **PASS** | Node input feature vector dimension = 7 (strictly pre-loading crystallographic features: sin/cos Euler angles + normalized volume. Target damage index excluded). |
| **4. Feature Normalization Fitted ONLY on Training Samples** | **PASS** | Scalers fitted exclusively on 7 training samples (Sample IDs 1-7) and saved to data/scaler_params.json prior to transforming validation and test sets. |
| **5. Independent Microstructure Seeds & Unique Geometries** | **PASS** | 10 independent LAMMPS simulations executed with 10 unique random seeds (seeds 101 to 110). |
| **6. Model Evaluated ONCE on Held-Out Test Set** | **PASS** | Evaluated GNN model on 29 held-out test set nodes (Samples 09 & 10). Output contains 29 predictions. |
| **7. Target Variable Distribution & Non-Constancy** | **PASS** | Target grain damage index std = 3.6230 (range: [6.75, 25.63]). |
| **8. Nonlocal Kernel Explicit Coupling Parameter Sensitivity** | **PASS** | Evaluated gamma sensitivity across 5 values (gamma in [0.0, 0.5, 1.0, 2.5, 5.0]). Max amplification ranges from 1.010x to 6.058x. |

---

## Detailed Metric Audit

- **Held-Out Test Set Baseline GCN MSE:** 38.3763 ± 17.7843
- **Held-Out Test Set Microstructure GAT MSE:** **6.8507 ± 2.6424**
- **Held-Out Test Set Microstructure GAT R²:** **0.5590**
- **Atomistic Interface Correlation:** GAT Attention vs. Damage Difference $r = 0.4240$ ($p = 1.39 \times 10^{-18}$).