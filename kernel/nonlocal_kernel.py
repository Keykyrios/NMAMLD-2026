import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def compute_nonlocal_kernels():
    print("--- Phase 8: Microstructure-Aware Nonlocal Interaction Kernel ---")

    # 1. Load Nodes and Learned GAT Attention Weights
    nodes_df = pd.read_csv(os.path.join("data", "graph_nodes.csv"))
    att_df = pd.read_csv(os.path.join("data", "learned_attention_weights.csv"))

    # Create lookup map for alpha(i, j)
    alpha_map = {}
    for _, row in att_df.iterrows():
        u = int(row["source_grain"])
        v = int(row["target_grain"])
        alpha_map[(u, v)] = float(row["learned_attention_alpha"])

    num_grains = len(nodes_df)
    centroids = nodes_df[["centroid_x", "centroid_y", "centroid_z"]].values

    # 2. Setup 2D Spatial Cross-Section Grid (Z = 35.0 Å)
    grid_size = 100
    x_range = np.linspace(0, 80.0, grid_size)
    y_range = np.linspace(0, 80.0, grid_size)
    X, Y = np.meshgrid(x_range, y_range)

    # Reference source point x_0 in Grain 1
    ref_grain = 1
    x0 = centroids[ref_grain - 1]
    print(f"Reference Point x0 (Grain {ref_grain} Centroid): [{x0[0]:.2f}, {x0[1]:.2f}, {x0[2]:.2f}] Å")

    # Nonlocal horizon parameter
    horizon_delta = 25.0 # Å
    gamma_weight = 2.5   # Microstructure coupling constant

    C_iso = np.zeros((grid_size, grid_size))
    C_informed = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            pt = np.array([X[i, j], Y[i, j], x0[2]])
            r = np.linalg.norm(pt - x0)

            if r <= horizon_delta and r > 0.1:
                # Isotropic Gaussian nonlocal kernel
                c0 = np.exp(- (r / horizon_delta)**2)
                C_iso[i, j] = c0

                # Determine nearest target grain for spatial point pt
                dists_to_grains = [np.linalg.norm(pt - c) for c in centroids]
                target_grain = int(np.argmin(dists_to_grains)) + 1

                # Retrieve learned attention weight between ref_grain and target_grain
                if target_grain == ref_grain:
                    alpha_ij = 1.0 # Intra-grain coherence
                else:
                    alpha_ij = alpha_map.get((ref_grain, target_grain), 0.0)

                # Microstructure-Informed Dirichlet Form Kernel
                C_informed[i, j] = c0 * (1.0 + gamma_weight * alpha_ij)

    # 3. Export Kernel Data CSV
    kernel_data = []
    for i in range(grid_size):
        for j in range(grid_size):
            kernel_data.append({
                "x": X[i, j],
                "y": Y[i, j],
                "C_isotropic": C_iso[i, j],
                "C_microstructure_informed": C_informed[i, j]
            })
    df_kernel = pd.DataFrame(kernel_data)
    csv_path = os.path.join("data", "nonlocal_kernel_field.csv")
    df_kernel.to_csv(csv_path, index=False)
    print(f"Saved nonlocal kernel field spatial data to {csv_path}.")

    # 4. Plot 2D Comparison Figure
    os.makedirs("plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Isotropic Kernel
    im1 = ax1.imshow(C_iso, extent=[0, 80, 0, 80], origin="lower", cmap="viridis")
    ax1.scatter(x0[0], x0[1], color="red", marker="*", s=200, label="Reference Point $x_0$")
    ax1.set_title("Standard Isotropic Nonlocal Kernel $C_{iso}(\\mathbf{x}, \\mathbf{y})$", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X Position [Å]", fontsize=11)
    ax1.set_ylabel("Y Position [Å]", fontsize=11)
    ax1.legend(loc="upper right")
    fig.colorbar(im1, ax=ax1, label="Kernel Weight Intensity")

    # Microstructure-Informed Kernel
    im2 = ax2.imshow(C_informed, extent=[0, 80, 0, 80], origin="lower", cmap="magma")
    ax2.scatter(x0[0], x0[1], color="cyan", marker="*", s=200, label="Reference Point $x_0$")

    # Overlay grain boundary centroids
    ax2.scatter(centroids[:, 0], centroids[:, 1], color="white", s=40, edgecolors="black", label="Grain Centroids")
    ax2.set_title("Proposed Microstructure-Informed Kernel $C_{informed}(\\mathbf{x}, \\mathbf{y}; \\alpha_{ij})$", fontsize=12, fontweight="bold")
    ax2.set_xlabel("X Position [Å]", fontsize=11)
    ax2.set_ylabel("Y Position [Å]", fontsize=11)
    ax2.legend(loc="upper right")
    fig.colorbar(im2, ax=ax2, label="Microstructure-Weighted Intensity")

    plt.tight_layout()
    fig_path = os.path.join("plots", "phase8_nonlocal_kernel.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Nonlocal kernel spatial comparison plot saved to {fig_path}.")

    print("\n==========================================")
    print("      PHASE 8 KERNEL DEMONSTRATION SUMMARY")
    print("==========================================")
    print(f"Isotropic Kernel Max Intensity    : {np.max(C_iso):.4f}")
    print(f"Informed Kernel Max Intensity     : {np.max(C_informed):.4f}")
    print(f"Informed/Isotropic Max Ratio      : {np.max(C_informed)/np.max(C_iso):.2f}x")

if __name__ == "__main__":
    compute_nonlocal_kernels()
