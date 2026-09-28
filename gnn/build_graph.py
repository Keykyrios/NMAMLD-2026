import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def build_microstructure_graph():
    print("--- Phase 6: Graph Representation Construction ---")
    
    # 1. Read Node & Edge Datasets
    nodes_df = pd.read_csv(os.path.join("data", "graph_nodes.csv"))
    edges_df = pd.read_csv(os.path.join("data", "graph_edges.csv"))
    
    num_nodes = len(nodes_df)
    print(f"Loaded {num_nodes} graph nodes (grains) and {len(edges_df)} undirected edges.")

    # 2. Construct Node Feature Matrix X
    # Encode Euler angles as sin/cos pairs for periodic boundary stability
    p1 = np.radians(nodes_df["euler_phi1"].values)
    th = np.radians(nodes_df["euler_theta"].values)
    p2 = np.radians(nodes_df["euler_phi2"].values)

    # Node features: Euler angle sin/cos pairs + normalized volume (NO grain_damage_index — that is the target)
    node_feats = np.column_stack([
        np.sin(p1), np.cos(p1),
        np.sin(th), np.cos(th),
        np.sin(p2), np.cos(p2),
        nodes_df["volume_est"].values / 1e5      # Normalized volume
    ])
    x = torch.tensor(node_feats, dtype=torch.float)

    # 3. Construct Edge Index & Features (Bidirectional)
    edge_src = []
    edge_tgt = []
    edge_attr_list = []

    for _, row in edges_df.iterrows():
        u = int(row["source_grain"]) - 1 # 0-indexed
        v = int(row["target_grain"]) - 1
        dist = float(row["distance_A"]) / 50.0
        misorient_rad = np.radians(float(row["misorientation_deg"]))

        # Edge features: distance, misorientation sin/cos (NO interface_damage_diff — target leakage)
        edge_feat = [dist, np.sin(misorient_rad), np.cos(misorient_rad)]

        # Edge u -> v
        edge_src.append(u)
        edge_tgt.append(v)
        edge_attr_list.append(edge_feat)

        # Edge v -> u (Undirected)
        edge_src.append(v)
        edge_tgt.append(u)
        edge_attr_list.append(edge_feat)

    edge_index = torch.tensor([edge_src, edge_tgt], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    # 4. Target Label y (Grain Damage Index)
    y = torch.tensor(nodes_df["grain_damage_index"].values, dtype=torch.float).unsqueeze(1)

    # 5. Create PyG Data object
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    torch.save(pyg_data, os.path.join("data", "microstructure_graph.pt"))
    
    print("\n==========================================")
    print("      PYTORCH GEOMETRIC GRAPH CREATED     ")
    print("==========================================")
    print(pyg_data)
    print(f"Node feature dimension (x)        : {x.shape}")
    print(f"Edge index dimension (edge_index)  : {edge_index.shape}")
    print(f"Edge feature dimension (edge_attr): {edge_attr.shape}")
    print(f"Target label dimension (y)        : {y.shape}")

    # 6. Plot 3D Grain Graph Network
    os.makedirs("plots", exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    centroids = nodes_df[["centroid_x", "centroid_y", "centroid_z"]].values
    damage = nodes_df["grain_damage_index"].values

    # Scatter nodes (color-coded by grain damage index)
    sc = ax.scatter(
        centroids[:, 0], centroids[:, 1], centroids[:, 2],
        c=damage, cmap="plasma", s=250, edgecolors='black', depthshade=True
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Grain Damage Index", fontsize=11, fontweight="bold")

    # Annotate node IDs
    for i in range(num_nodes):
        ax.text(centroids[i, 0]+1, centroids[i, 1]+1, centroids[i, 2]+1, f"G{i+1}", fontsize=10, fontweight="bold")

    # Draw edges
    for _, row in edges_df.iterrows():
        u = int(row["source_grain"]) - 1
        v = int(row["target_grain"]) - 1
        p1 = centroids[u]
        p2 = centroids[v]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="gray", alpha=0.5, lw=1.2)

    ax.set_xlabel("X Position [Å]", fontsize=11)
    ax.set_ylabel("Y Position [Å]", fontsize=11)
    ax.set_zlabel("Z Position [Å]", fontsize=11)
    ax.set_title("Phase 6: 3D Microstructure Grain Interaction Graph", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig_path = os.path.join("plots", "phase6_microstructure_graph.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Graph visualization saved to {fig_path}.")

if __name__ == "__main__":
    build_microstructure_graph()
