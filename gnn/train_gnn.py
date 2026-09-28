import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 1. Baseline Simple GCN Model
class BaselineGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.fc(h)

# 2. Microstructure-Informed Graph Attention Network (GAT with Edge Attributes)
class MicrostructureGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, out_channels, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=edge_dim, concat=False)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, return_attention_weights=False):
        # Layer 1
        h, (edge_index_1, alpha1) = self.gat1(x, edge_index, edge_attr, return_attention_weights=True)
        h = F.elu(h)
        # Layer 2
        h, (edge_index_2, alpha2) = self.gat2(h, edge_index, edge_attr, return_attention_weights=True)
        out = self.fc(h)

        if return_attention_weights:
            return out, (edge_index_2, alpha2)
        return out

def train_and_evaluate():
    print("--- Phase 7: GNN & Graph Attention Model Training ---")
    
    # Load PyG Data
    data_path = os.path.join("data", "microstructure_graph.pt")
    graph_data = torch.load(data_path, weights_only=False)
    print(f"Loaded Graph Data: {graph_data}")

    x = graph_data.x
    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr
    y = graph_data.y

    in_channels = x.shape[1]
    edge_dim = edge_attr.shape[1]
    hidden_channels = 32

    # Instantiate Models
    baseline_model = BaselineGCN(in_channels, hidden_channels, out_channels=1)
    gat_model = MicrostructureGAT(in_channels, hidden_channels, edge_dim, out_channels=1, heads=4)

    optimizer_base = torch.optim.Adam(baseline_model.parameters(), lr=0.005, weight_decay=1e-4)
    optimizer_gat = torch.optim.Adam(gat_model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.MSELoss()

    epochs = 200
    baseline_losses = []
    gat_losses = []

    print("\nBeginning GNN Model Training (200 Epochs)...")
    for epoch in range(1, epochs + 1):
        # Train Baseline GCN
        baseline_model.train()
        optimizer_base.zero_grad()
        out_base = baseline_model(x, edge_index)
        loss_base = criterion(out_base, y)
        loss_base.backward()
        optimizer_base.step()
        baseline_losses.append(loss_base.item())

        # Train GAT
        gat_model.train()
        optimizer_gat.zero_grad()
        out_gat = gat_model(x, edge_index, edge_attr)
        loss_gat = criterion(out_gat, y)
        loss_gat.backward()
        optimizer_gat.step()
        gat_losses.append(loss_gat.item())

        if epoch % 40 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Baseline GCN Loss: {loss_base.item():.6f} | Microstructure GAT Loss: {loss_gat.item():.6f}")

    # Extract Learned Attention Weights
    gat_model.eval()
    with torch.no_grad():
        final_pred, (att_edge_index, att_weights) = gat_model(x, edge_index, edge_attr, return_attention_weights=True)

    att_weights = att_weights.squeeze().numpy()
    att_edges = att_edge_index.numpy()

    # Save attention weights to CSV
    att_df = pd.DataFrame({
        "source_grain": att_edges[0] + 1,
        "target_grain": att_edges[1] + 1,
        "learned_attention_alpha": att_weights
    })
    att_csv = os.path.join("data", "learned_attention_weights.csv")
    att_df.to_csv(att_csv, index=False)
    print(f"\nSaved learned GAT attention weights ({len(att_df)} edges) to {att_csv}.")

    # Plot Training Loss Comparison
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), baseline_losses, color="#7f8c8d", linestyle="--", lw=2, label="Baseline GCN")
    plt.plot(range(1, epochs + 1), gat_losses, color="#8e44ad", lw=2.5, label="Microstructure GAT (Proposed)")
    plt.xlabel("Training Epochs", fontsize=12, fontweight="bold")
    plt.ylabel("MSE Loss", fontsize=12, fontweight="bold")
    plt.title("Phase 7: GNN Training Loss Convergence Comparison", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join("plots", "phase7_gnn_training_loss.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Training loss comparison plot saved to {fig_path}.")

    print("\n==========================================")
    print("       PHASE 7 GNN EVALUATION SUMMARY     ")
    print("==========================================")
    print(f"Final Baseline GCN Loss : {baseline_losses[-1]:.6f}")
    print(f"Final GAT Loss          : {gat_losses[-1]:.6f}")
    print(f"Mean Learned Attention  : {np.mean(att_weights):.4f} (std: {np.std(att_weights):.4f})")

if __name__ == "__main__":
    train_and_evaluate()
