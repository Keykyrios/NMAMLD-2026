import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
import json
import os
import copy
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

# 1. Baseline GCN Model (Node features only) — deeper with dropout
class BaselineGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv3(h, edge_index))
        return self.fc(h)

# 2. Hybrid MPNN + GAT Architecture with MLP Decoder Head (Eq. 1 in main (1).tex)
class HybridMPNN_GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, out_channels, heads=4, dropout=0.2):
        super().__init__()
        # MPNN + GAT Message Passing Layer 1
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim, concat=True, dropout=dropout)
        # GAT Message Passing Layer 2
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=edge_dim, concat=True, dropout=dropout)
        # GAT Message Passing Layer 3
        self.gat3 = GATConv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=edge_dim, concat=False, dropout=dropout)
        
        self.dropout = dropout
        
        # MLP Decoder Head mapping learned graph representations to output kernel parameters & damage predictions
        self.mlp_decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, return_attention_weights=False):
        h, (edge_index_1, alpha1) = self.gat1(x, edge_index, edge_attr, return_attention_weights=True)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h, (edge_index_2, alpha2) = self.gat2(h, edge_index, edge_attr, return_attention_weights=True)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h, (edge_index_3, alpha3) = self.gat3(h, edge_index, edge_attr, return_attention_weights=True)
        
        out = self.mlp_decoder(h)

        if return_attention_weights:
            return out, (edge_index_3, alpha3)
        return out

def build_pyg_graph(sample_id, df_nodes, df_edges, node_scaler_mean, node_scaler_std, edge_scaler_mean, edge_scaler_std, target_mean=0.0, target_std=1.0):
    sample_nodes = df_nodes[df_nodes["sample_id"] == sample_id].copy().sort_values("grain_id").reset_index(drop=True)
    sample_edges = df_edges[df_edges["sample_id"] == sample_id].copy()
    num_grains = len(sample_nodes)

    # Count node degree (number of grain boundaries per grain) — purely structural, pre-loading
    degree_count = np.zeros(num_grains)
    for _, row in sample_edges.iterrows():
        u = int(row["source_grain"]) - 1
        v = int(row["target_grain"]) - 1
        if 0 <= u < num_grains:
            degree_count[u] += 1
        if 0 <= v < num_grains:
            degree_count[v] += 1

    # Node Features (9 features):
    #   6 Euler sin/cos + Taylor factor + atom_count + node_degree
    # The feature set consists purely of pre-loading geometric and crystallographic properties.
    p1 = np.radians(sample_nodes["euler_phi1"].values)
    th = np.radians(sample_nodes["euler_theta"].values)
    p2 = np.radians(sample_nodes["euler_phi2"].values)
    taylor_m = sample_nodes["taylor_factor"].values
    atom_ct  = sample_nodes["atom_count"].values.astype(float)

    raw_node_feats = np.column_stack([
        np.sin(p1), np.cos(p1),
        np.sin(th), np.cos(th),
        np.sin(p2), np.cos(p2),
        taylor_m,
        atom_ct,
        degree_count
    ])

    norm_node_feats = (raw_node_feats - node_scaler_mean) / (node_scaler_std + 1e-8)
    x = torch.tensor(norm_node_feats, dtype=torch.float)

    # Target y: log-transformed grain_damage_index (handles heavy right tail)
    raw_damage = sample_nodes["grain_damage_index"].values
    log_damage = np.log1p(raw_damage)  # log(1 + x) to handle values near 0
    # Standardize using train-set statistics
    y_norm = (log_damage - target_mean) / (target_std + 1e-8)
    y = torch.tensor(y_norm, dtype=torch.float).unsqueeze(1)

    # Edges (4 features - distance, misorientation sin/cos, Read-Shockley GB energy gamma_GB)
    edge_src, edge_tgt = [], []
    raw_edge_attrs = []

    for _, row in sample_edges.iterrows():
        u = int(row["source_grain"]) - 1
        v = int(row["target_grain"]) - 1
        dist = float(row["distance_A"])
        misorient_rad = np.radians(float(row["misorientation_deg"]))
        gb_energy = float(row["gb_interface_energy_Jm2"])

        feat = [dist, np.sin(misorient_rad), np.cos(misorient_rad), gb_energy]
        edge_src.extend([u, v])
        edge_tgt.extend([v, u])
        raw_edge_attrs.extend([feat, feat])

    edge_index = torch.tensor([edge_src, edge_tgt], dtype=torch.long)
    raw_edge_attrs = np.array(raw_edge_attrs)

    norm_edge_attrs = (raw_edge_attrs - edge_scaler_mean) / (edge_scaler_std + 1e-8)
    edge_attr = torch.tensor(norm_edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, sample_id=sample_id,
                raw_damage=torch.tensor(raw_damage, dtype=torch.float))

def run_experiment():
    print("--- Phases C-F: Multi-Sample Hybrid MPNN+GAT+MLP Training & Held-Out Test Evaluation ---")
    print(f"Training on: {device}")
    
    nodes_df = pd.read_csv("data/multisample_nodes.csv")
    edges_df = pd.read_csv("data/multisample_edges.csv")
    
    train_sample_ids = [1, 2, 3, 4, 5, 6, 7]
    val_sample_ids   = [8]
    test_sample_ids  = [9, 10]

    pd.DataFrame({"sample_id": train_sample_ids}).to_csv("data/train_samples.csv", index=False)
    pd.DataFrame({"sample_id": val_sample_ids}).to_csv("data/validation_samples.csv", index=False)
    pd.DataFrame({"sample_id": test_sample_ids}).to_csv("data/test_samples.csv", index=False)

    train_nodes = nodes_df[nodes_df["sample_id"].isin(train_sample_ids)]
    train_edges = edges_df[edges_df["sample_id"].isin(train_sample_ids)]

    # Compute node degree for training set scaler
    def compute_degree(nodes_sub, edges_sub):
        degrees = []
        for sid in nodes_sub["sample_id"].unique():
            sn = nodes_sub[nodes_sub["sample_id"] == sid].sort_values("grain_id").reset_index(drop=True)
            se = edges_sub[edges_sub["sample_id"] == sid]
            deg = np.zeros(len(sn))
            for _, row in se.iterrows():
                u, v = int(row["source_grain"]) - 1, int(row["target_grain"]) - 1
                if 0 <= u < len(sn): deg[u] += 1
                if 0 <= v < len(sn): deg[v] += 1
            degrees.extend(deg)
        return np.array(degrees)

    train_degrees = compute_degree(train_nodes, train_edges)

    p1 = np.radians(train_nodes["euler_phi1"].values)
    th = np.radians(train_nodes["euler_theta"].values)
    p2 = np.radians(train_nodes["euler_phi2"].values)
    train_node_raw = np.column_stack([
        np.sin(p1), np.cos(p1), np.sin(th), np.cos(th), np.sin(p2), np.cos(p2),
        train_nodes["taylor_factor"].values,
        train_nodes["atom_count"].values.astype(float),
        train_degrees
    ])

    node_scaler_mean = np.mean(train_node_raw, axis=0)
    node_scaler_std  = np.std(train_node_raw, axis=0)

    # Target transform statistics (log1p, then standardize) — fitted on TRAIN only
    train_damage = train_nodes["grain_damage_index"].values
    train_log_damage = np.log1p(train_damage)
    target_mean = float(np.mean(train_log_damage))
    target_std  = float(np.std(train_log_damage))

    raw_edge_list = []
    for _, row in train_edges.iterrows():
        feat = [
            float(row["distance_A"]),
            np.sin(np.radians(float(row["misorientation_deg"]))),
            np.cos(np.radians(float(row["misorientation_deg"]))),
            float(row["gb_interface_energy_Jm2"])
        ]
        raw_edge_list.extend([feat, feat])
    train_edge_raw = np.array(raw_edge_list)

    edge_scaler_mean = np.mean(train_edge_raw, axis=0)
    edge_scaler_std  = np.std(train_edge_raw, axis=0)

    scaler_params = {
        "node_scaler_mean": node_scaler_mean.tolist(),
        "node_scaler_std": node_scaler_std.tolist(),
        "edge_scaler_mean": edge_scaler_mean.tolist(),
        "edge_scaler_std": edge_scaler_std.tolist(),
        "target_mean": target_mean,
        "target_std": target_std
    }
    with open("data/scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=2)

    train_graphs = [build_pyg_graph(sid, nodes_df, edges_df, node_scaler_mean, node_scaler_std, edge_scaler_mean, edge_scaler_std, target_mean, target_std).to(device) for sid in train_sample_ids]
    val_graphs   = [build_pyg_graph(sid, nodes_df, edges_df, node_scaler_mean, node_scaler_std, edge_scaler_mean, edge_scaler_std, target_mean, target_std).to(device) for sid in val_sample_ids]
    test_graphs  = [build_pyg_graph(sid, nodes_df, edges_df, node_scaler_mean, node_scaler_std, edge_scaler_mean, edge_scaler_std, target_mean, target_std).to(device) for sid in test_sample_ids]

    model_seeds = [42, 43, 44, 45, 46]
    
    gcn_test_mses, gcn_test_maes, gcn_test_r2s = [], [], []
    gat_test_mses, gat_test_maes, gat_test_r2s = [], [], []

    last_gat_preds, last_gat_targets = [], []

    for seed in model_seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        gcn = BaselineGCN(in_channels=9, hidden_channels=64, out_channels=1).to(device)
        opt_gcn = torch.optim.Adam(gcn.parameters(), lr=0.003, weight_decay=1e-4)
        sched_gcn = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_gcn, patience=30, factor=0.5, min_lr=1e-5)
        
        gat = HybridMPNN_GAT(in_channels=9, hidden_channels=64, edge_dim=4, out_channels=1, heads=4).to(device)
        opt_gat = torch.optim.Adam(gat.parameters(), lr=0.003, weight_decay=1e-4)
        sched_gat = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_gat, patience=30, factor=0.5, min_lr=1e-5)

        criterion = nn.MSELoss()  # MSE in log-standardized space
        best_gcn_val_loss = float('inf')
        best_gat_val_loss = float('inf')
        best_gcn_state = None
        best_gat_state = None
        patience_counter_gcn = 0
        patience_counter_gat = 0
        max_patience = 80

        for epoch in range(1, 501):
            # --- Train GCN ---
            gcn.train()
            for g in train_graphs:
                opt_gcn.zero_grad()
                out = gcn(g.x, g.edge_index)
                l = criterion(out, g.y)
                l.backward()
                opt_gcn.step()

            # --- Train GAT ---
            gat.train()
            for g in train_graphs:
                opt_gat.zero_grad()
                out = gat(g.x, g.edge_index, g.edge_attr)
                l = criterion(out, g.y)
                l.backward()
                opt_gat.step()

            # --- Validate for early stopping ---
            gcn.eval()
            gat.eval()
            with torch.no_grad():
                val_loss_gcn = sum(criterion(gcn(g.x, g.edge_index), g.y).item() for g in val_graphs) / len(val_graphs)
                val_loss_gat = sum(criterion(gat(g.x, g.edge_index, g.edge_attr), g.y).item() for g in val_graphs) / len(val_graphs)

            sched_gcn.step(val_loss_gcn)
            sched_gat.step(val_loss_gat)

            if val_loss_gcn < best_gcn_val_loss:
                best_gcn_val_loss = val_loss_gcn
                best_gcn_state = copy.deepcopy(gcn.state_dict())
                patience_counter_gcn = 0
            else:
                patience_counter_gcn += 1

            if val_loss_gat < best_gat_val_loss:
                best_gat_val_loss = val_loss_gat
                best_gat_state = copy.deepcopy(gat.state_dict())
                patience_counter_gat = 0
            else:
                patience_counter_gat += 1

            if patience_counter_gcn >= max_patience and patience_counter_gat >= max_patience:
                print(f"    [Seed {seed}] Early stop at epoch {epoch}")
                break

        # Restore best models
        if best_gcn_state is not None:
            gcn.load_state_dict(best_gcn_state)
        if best_gat_state is not None:
            gat.load_state_dict(best_gat_state)

        gcn.eval()
        gat.eval()

        gcn_preds, gat_preds, true_targets = [], [], []

        with torch.no_grad():
            for g in test_graphs:
                p_gcn_norm = gcn(g.x, g.edge_index).squeeze().cpu().numpy()
                p_gat_norm = gat(g.x, g.edge_index, g.edge_attr).squeeze().cpu().numpy()

                # Inverse transform: unstandardize then expm1
                p_gcn_log = p_gcn_norm * target_std + target_mean
                p_gat_log = p_gat_norm * target_std + target_mean
                p_gcn_raw = np.expm1(p_gcn_log)  # inverse of log1p
                p_gat_raw = np.expm1(p_gat_log)
                t_y = g.raw_damage.cpu().numpy()  # original scale

                gcn_preds.extend(p_gcn_raw)
                gat_preds.extend(p_gat_raw)
                true_targets.extend(t_y)

        gcn_preds = np.array(gcn_preds)
        gat_preds = np.array(gat_preds)
        true_targets = np.array(true_targets)

        # Clip negative predictions (expm1 can produce negatives from noisy log-space preds)
        gcn_preds = np.clip(gcn_preds, 0, None)
        gat_preds = np.clip(gat_preds, 0, None)

        gcn_test_mses.append(mean_squared_error(true_targets, gcn_preds))
        gcn_test_maes.append(mean_absolute_error(true_targets, gcn_preds))
        gcn_test_r2s.append(r2_score(true_targets, gcn_preds))

        gat_test_mses.append(mean_squared_error(true_targets, gat_preds))
        gat_test_maes.append(mean_absolute_error(true_targets, gat_preds))
        gat_test_r2s.append(r2_score(true_targets, gat_preds))

        # Also compute R² in log-space (the space the model actually trains in)
        log_true = np.log1p(true_targets)
        log_gcn  = np.log1p(gcn_preds)
        log_gat  = np.log1p(gat_preds)
        if seed == model_seeds[0]:  # Print once
            print(f"    [Log-space R²] GCN: {r2_score(log_true, log_gcn):.4f}, GAT: {r2_score(log_true, log_gat):.4f}")

        last_gat_preds = gat_preds
        last_gat_targets = true_targets

    test_node_records = []
    node_idx = 0
    for g in test_graphs:
        for i in range(len(g.y)):
            test_node_records.append({
                "sample_id": g.sample_id,
                "node_idx": i + 1,
                "true_target": float(last_gat_targets[node_idx]),
                "gat_predicted_target": float(last_gat_preds[node_idx])
            })
            node_idx += 1

    pd.DataFrame(test_node_records).to_csv("data/gnn_results.csv", index=False)

    metrics_summary = {
        "Baseline_GCN_Test_MSE_mean": float(np.mean(gcn_test_mses)),
        "Baseline_GCN_Test_MSE_std":  float(np.std(gcn_test_mses)),
        "Baseline_GCN_Test_MAE_mean": float(np.mean(gcn_test_maes)),
        "Baseline_GCN_Test_R2_mean":  float(np.mean(gcn_test_r2s)),
        "Microstructure_GAT_Test_MSE_mean": float(np.mean(gat_test_mses)),
        "Microstructure_GAT_Test_MSE_std":  float(np.std(gat_test_mses)),
        "Microstructure_GAT_Test_MAE_mean": float(np.mean(gat_test_maes)),
        "Microstructure_GAT_Test_R2_mean":  float(np.mean(gat_test_r2s))
    }

    pd.DataFrame([metrics_summary]).to_csv("data/model_metrics.csv", index=False)
    print("\n==========================================")
    print("  HELD-OUT TEST EVALUATION (MPNN+GAT+MLP) ")
    print("==========================================")
    print(f"Baseline GCN Test MSE        : {np.mean(gcn_test_mses):.4f} ± {np.std(gcn_test_mses):.4f}")
    print(f"Baseline GCN Test MAE        : {np.mean(gcn_test_maes):.4f}")
    print(f"Baseline GCN Test R²         : {np.mean(gcn_test_r2s):.4f}")
    print("------------------------------------------")
    print(f"Hybrid MPNN+GAT Test MSE     : {np.mean(gat_test_mses):.4f} ± {np.std(gat_test_mses):.4f}")
    print(f"Hybrid MPNN+GAT Test MAE     : {np.mean(gat_test_maes):.4f}")
    print(f"Hybrid MPNN+GAT Test R²      : {np.mean(gat_test_r2s):.4f}")
    print("==========================================")

if __name__ == "__main__":
    run_experiment()
