import pandas as pd
import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import sys
sys.path.append("gnn")
from correct_multisample_gnn import HybridMPNN_GAT, build_pyg_graph, device

def run_attention_and_kernel_analysis():
    print("--- Phases G-I: Attention Analysis, Nonlocal Kernel Sensitivity & Atomistic Validation ---")

    # 1. Load Data & Scalers
    nodes_df = pd.read_csv("data/multisample_nodes.csv")
    edges_df = pd.read_csv("data/multisample_edges.csv")

    with open("data/scaler_params.json", "r") as f:
        scalers = json.load(f)

    node_mean = np.array(scalers["node_scaler_mean"])
    node_std  = np.array(scalers["node_scaler_std"])
    edge_mean = np.array(scalers["edge_scaler_mean"])
    edge_std  = np.array(scalers["edge_scaler_std"])
    target_mean = float(scalers["target_mean"])
    target_std  = float(scalers["target_std"])

    train_ids = [1, 2, 3, 4, 5, 6, 7]
    val_ids   = [8]
    test_ids  = [9, 10]

    # 2. Load the Best GAT Model saved by correct_multisample_gnn.py so the
    #    attention weights come from exactly the model that produced the
    #    reported held-out test metrics.
    ckpt_path = "data/checkpoints/best_gat_model.pt"
    gat = HybridMPNN_GAT(in_channels=9, hidden_channels=64, edge_dim=4, out_channels=1, heads=4).to(device)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} not found. Run gnn/correct_multisample_gnn.py first — "
            "it saves the best-of-5-seeds model that this script must reuse."
        )
    state_dict = torch.load(ckpt_path, map_location=device)
    gat.load_state_dict(state_dict)
    print("Loaded best GAT model checkpoint from", ckpt_path)

    gat.eval()

    # 3. PHASE G: Extract Attention Weights Across Splits
    all_attention_records = []
    
    def extract_split_attention(sample_ids, split_name):
        for sid in sample_ids:
            g = build_pyg_graph(sid, nodes_df, edges_df, node_mean, node_std, edge_mean, edge_std, target_mean, target_std).to(device)
            with torch.no_grad():
                out, (edge_index_2, alpha) = gat(g.x, g.edge_index, g.edge_attr, return_attention_weights=True)

            # Build proper node_index -> grain_id mapping
            sample_nodes = nodes_df[nodes_df["sample_id"] == sid].sort_values("grain_id").reset_index(drop=True)
            idx_to_gid = sample_nodes["grain_id"].values  # idx_to_gid[node_idx] = grain_id

            edges_sample = edges_df[edges_df["sample_id"] == sid].copy().reset_index(drop=True)
            alpha_np = alpha.squeeze().cpu().numpy()
            e_idx = edge_index_2.cpu().numpy()

            for i in range(e_idx.shape[1]):
                # GATConv adds self-loops before attention; drop them so the
                # saved weights contain only real grain-boundary edges (a
                # self-loop would otherwise be logged with distance/misorientation
                # defaults of 0.0 and pollute the correlation analysis)
                if e_idx[0, i] == e_idx[1, i]:
                    continue
                u = int(idx_to_gid[e_idx[0, i]])
                v = int(idx_to_gid[e_idx[1, i]])
                w = float(alpha_np[i])

                # Match with edge features
                match = edges_sample[((edges_sample["source_grain"] == u) & (edges_sample["target_grain"] == v)) |
                                     ((edges_sample["source_grain"] == v) & (edges_sample["target_grain"] == u))]
                
                misorient = float(match["misorientation_deg"].iloc[0]) if len(match) > 0 else 0.0
                dist = float(match["distance_A"].iloc[0]) if len(match) > 0 else 0.0
                gb_energy = float(match["gb_interface_energy_Jm2"].iloc[0]) if len(match) > 0 else 0.0

                all_attention_records.append({
                    "split": split_name,
                    "sample_id": sid,
                    "source_grain": u,
                    "target_grain": v,
                    "attention_weight": w,
                    "distance_A": dist,
                    "misorientation_deg": misorient,
                    "gb_interface_energy_Jm2": gb_energy
                })

    extract_split_attention(train_ids, "train")
    extract_split_attention(val_ids, "val")
    extract_split_attention(test_ids, "test")

    att_df = pd.DataFrame(all_attention_records)
    att_df.to_csv("data/learned_attention_weights.csv", index=False)
    print(f"Extracted attention weights for {len(att_df)} directed edges across Train/Val/Test splits.")

    # 4. PHASE I: Physical Feature Validation & Correlations
    test_att = att_df[att_df["split"] == "test"]
    
    # Correlation with misorientation angle (pre-loading physical feature)
    r_misorient, p_misorient = pearsonr(test_att["misorientation_deg"], test_att["attention_weight"])
    rho_misorient, p_spearman = spearmanr(test_att["misorientation_deg"], test_att["attention_weight"])
    
    # Correlation with GB energy (pre-loading physical feature)
    r_gb_energy, p_gb_energy = pearsonr(test_att["gb_interface_energy_Jm2"], test_att["attention_weight"])

    # Correlation with distance (pre-loading physical feature)
    r_dist, p_dist = pearsonr(test_att["distance_A"], test_att["attention_weight"])

    print("\n==========================================")
    print("  PHYSICAL ATOMISTIC CORRELATION ANALYSIS ")
    print("  (Correlations with pre-loading features)")
    print("==========================================")
    print(f"GAT Attention vs Misorientation (Test Set): Pearson r = {r_misorient:.4f}, Spearman rho = {rho_misorient:.4f} (p = {p_spearman:.4e})")
    print(f"GAT Attention vs GB Energy     (Test Set): Pearson r = {r_gb_energy:.4f} (p = {p_gb_energy:.4e})")
    print(f"GAT Attention vs Distance      (Test Set): Pearson r = {r_dist:.4f} (p = {p_dist:.4e})")

    # 5. PHASE H: Nonlocal Kernel Gamma Sensitivity Analysis
    gamma_values = [0.0, 0.5, 1.0, 2.5, 5.0]
    sensitivity_records = []

    # Select representative held-out test sample 09
    test_sample_nodes = nodes_df[nodes_df["sample_id"] == 9].copy().sort_values("grain_id").reset_index(drop=True)
    test_sample_att   = att_df[att_df["sample_id"] == 9]

    alpha_map = {}
    for _, row in test_sample_att.iterrows():
        alpha_map[(int(row["source_grain"]), int(row["target_grain"]))] = float(row["attention_weight"])

    centroids = test_sample_nodes[["centroid_x", "centroid_y", "centroid_z"]].values
    grain_ids = test_sample_nodes["grain_id"].values
    ref_idx = 0
    ref_gid = int(grain_ids[ref_idx])
    x0 = centroids[ref_idx]

    # Scale grid to match the box size of the test sample
    box_size = max(centroids.max(axis=0) - centroids.min(axis=0)) * 1.2
    grid_size = 80
    x_range = np.linspace(0, box_size, grid_size)
    y_range = np.linspace(0, box_size, grid_size)
    X, Y = np.meshgrid(x_range, y_range)
    horizon_delta = box_size * 0.15  # ~15% of box size

    for gamma in gamma_values:
        C_vals = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                pt = np.array([X[i, j], Y[i, j], x0[2]])
                r = np.linalg.norm(pt - x0)
                if 0.1 < r <= horizon_delta:
                    c0 = np.exp(-(r / horizon_delta)**2)
                    target_idx = int(np.argmin([np.linalg.norm(pt - c) for c in centroids]))
                    target_gid = int(grain_ids[target_idx])
                    alpha_ij = 1.0 if target_gid == ref_gid else alpha_map.get((ref_gid, target_gid), 0.0)
                    C_vals[i, j] = c0 * (1.0 + gamma * alpha_ij)

        max_c = float(np.max(C_vals))
        mean_c = float(np.mean(C_vals[C_vals > 0])) if np.any(C_vals > 0) else 0.0
        std_c = float(np.std(C_vals[C_vals > 0])) if np.any(C_vals > 0) else 0.0

        sensitivity_records.append({
            "gamma": gamma,
            "max_kernel_intensity": max_c,
            "mean_nonzero_kernel": mean_c,
            "std_nonzero_kernel": std_c,
            "max_amplification_ratio": round(max_c / (np.exp(-0.01) + 1e-8), 3)
        })

    sens_df = pd.DataFrame(sensitivity_records)
    sens_df.to_csv("data/nonlocal_kernel_sensitivity.csv", index=False)
    print("\nSaved nonlocal kernel gamma sensitivity to data/nonlocal_kernel_sensitivity.csv:")
    print(sens_df)

if __name__ == "__main__":
    run_attention_and_kernel_analysis()
