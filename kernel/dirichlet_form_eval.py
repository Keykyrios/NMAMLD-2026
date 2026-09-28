import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

def evaluate_dirichlet_form_eq1():
    print("--- Evaluating Dirichlet Form Energy Functional (Eq. 1 from main (1).tex) ---")
    
    # 1. Load Nodes, Edges, and Learned GAT Attention Weights
    nodes_df = pd.read_csv(os.path.join("data", "multisample_nodes.csv"))
    edges_df = pd.read_csv(os.path.join("data", "multisample_edges.csv"))
    att_df   = pd.read_csv(os.path.join("data", "learned_attention_weights.csv"))

    # Focus on held-out test sample 09
    sample_id = 9
    sample_nodes = nodes_df[nodes_df["sample_id"] == sample_id].copy().sort_values("grain_id").reset_index(drop=True)
    sample_edges = edges_df[edges_df["sample_id"] == sample_id]
    sample_att   = att_df[att_df["sample_id"] == sample_id]

    num_grains = len(sample_nodes)
    centroids  = sample_nodes[["centroid_x", "centroid_y", "centroid_z"]].values
    damage_vec = sample_nodes["grain_damage_index"].values

    # Normalize damage to w(x) in [0, 1) range for degradation term (1 - w(x))^alpha
    max_d = np.max(damage_vec) + 1e-5
    omega = damage_vec / max_d # Local damage variable w(x) in [0, 1)

    alpha_deg = 2.0  # Damage degradation exponent alpha
    gamma_coup = 2.5 # Coupling parameter gamma

    # Create attention lookup map alpha(i, j)
    alpha_map = {}
    for _, row in sample_att.iterrows():
        u = int(row["source_grain"])
        v = int(row["target_grain"])
        alpha_map[(u, v)] = float(row["attention_weight"])

    # 2. Evaluate Spatial Field over 2D Grid Section
    # Determine box size dynamically from centroid range (not hardcoded)
    box_size = max(centroids.max(axis=0) - centroids.min(axis=0)) * 1.2  # 20% margin
    box_size = max(box_size, 80.0)  # Floor at 80 A
    grid_size = 80
    x_range = np.linspace(0, box_size, grid_size)
    y_range = np.linspace(0, box_size, grid_size)
    X, Y = np.meshgrid(x_range, y_range)

    # Reference point x_0 in Grain 1
    ref_grain = 1
    x0 = centroids[ref_grain - 1]
    w_x0 = omega[ref_grain - 1]

    horizon_delta = box_size * 0.15 # Horizon delta scaled to ~15% of box

    w_hat_field = np.zeros((grid_size, grid_size))
    dirichlet_integrand = np.zeros((grid_size, grid_size))
    resistance_metric   = np.zeros((grid_size, grid_size))

    # Applied displacement field u(x) = strain * x (linear displacement test function)
    strain_test = 0.10
    u_x0 = strain_test * x0[0]

    for i in range(grid_size):
        for j in range(grid_size):
            pt = np.array([X[i, j], Y[i, j], x0[2]])
            r = np.linalg.norm(pt - x0)

            if 0.1 < r <= horizon_delta:
                # Fractional / Gaussian spatial envelope
                c0 = np.exp(-(r / horizon_delta)**2)

                # Nearest grain assignment for spatial point y = pt
                target_g = int(np.argmin([np.linalg.norm(pt - c) for c in centroids])) + 1
                w_y = omega[target_g - 1]

                alpha_ij = 1.0 if target_g == ref_grain else alpha_map.get((ref_grain, target_g), 0.0)

                # Anisotropic weight kernel w_hat(x, y; G)
                w_hat = c0 * (1.0 + gamma_coup * alpha_ij)
                w_hat_field[i, j] = w_hat

                # Displacement difference |u(x) - u(y)|^2
                u_y = strain_test * pt[0]
                du_sq = (u_x0 - u_y)**2

                # Equation 1 Integrand: |u(x) - u(y)|^2 * w_hat(x,y;G) * (1 - w(x))^alpha * (1 - w(y))^alpha
                degrad_term = ((1.0 - w_x0)**alpha_deg) * ((1.0 - w_y)**alpha_deg)
                integrand_val = du_sq * w_hat * degrad_term
                dirichlet_integrand[i, j] = integrand_val

                # Effective Resistance Metric R_eff(x, y) = 1 / (w_hat * degrad_term + 1e-12)
                resistance_metric[i, j] = 1.0 / (w_hat * degrad_term + 1e-8)

    # Calculate Total Dirichlet Form Energy E_GNN(u, u)
    dx = box_size / grid_size
    dy = box_size / grid_size
    E_GNN_val = float(np.sum(dirichlet_integrand) * dx * dy)

    print("\n==========================================")
    print("  DIRICHLET FORM E_GNN(u, u) EVALUATION   ")
    print("==========================================")
    print(f"Evaluated Sample ID                 : {sample_id} (Held-Out Test Set)")
    print(f"Total Dirichlet Energy E_GNN(u, u)  : {E_GNN_val:.6f} eV")
    print(f"Max Anisotropic Kernel w_hat(x,y;G) : {np.max(w_hat_field):.4f}")
    print(f"Max Integrated Integrand            : {np.max(dirichlet_integrand):.6f}")
    print(f"Max Resistance Metric R_eff(x, y)   : {np.max(resistance_metric):.2f} (Degenerates along damaged GBs)")

    # Save DataFrame Output
    df_eval = pd.DataFrame({
        "sample_id": sample_id,
        "E_GNN_energy_eV": [E_GNN_val],
        "max_w_hat": [np.max(w_hat_field)],
        "max_resistance_R_eff": [np.max(resistance_metric)],
        "alpha_degradation_exp": [alpha_deg],
        "gamma_coupling": [gamma_coup]
    })
    csv_path = os.path.join("data", "dirichlet_form_evaluation.csv")
    df_eval.to_csv(csv_path, index=False)
    print(f"Saved Dirichlet form metrics to {csv_path}.")

    # Plot Eq 1 Integrand and Resistance Metric
    os.makedirs("plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    im1 = ax1.imshow(dirichlet_integrand, extent=[0, box_size, 0, box_size], origin="lower", cmap="inferno")
    ax1.scatter(x0[0], x0[1], color="cyan", marker="*", s=200, label="$x_0$ Ref Point")
    ax1.set_title("Eq. (1) Dirichlet Form Integrand $\\mathcal{E}_{GNN}(u,u)$", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X Position [Å]", fontsize=11)
    ax1.set_ylabel("Y Position [Å]", fontsize=11)
    ax1.legend(loc="upper right")
    fig.colorbar(im1, ax=ax1, label="Energy Density Integrand")

    im2 = ax2.imshow(np.log10(resistance_metric + 1.0), extent=[0, box_size, 0, box_size], origin="lower", cmap="plasma")
    ax2.scatter(x0[0], x0[1], color="cyan", marker="*", s=200, label="$x_0$ Ref Point")
    ax2.set_title("Induced Resistance Metric $\\log_{10}(R_{eff}(x, y))$ (Capacity Theory)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X Position [Å]", fontsize=11)
    ax2.set_ylabel("Y Position [Å]", fontsize=11)
    ax2.legend(loc="upper right")
    fig.colorbar(im2, ax=ax2, label=r"$\log_{10}(R_{eff})$ (Degenerates at fracture)")

    plt.tight_layout()
    fig_path = os.path.join("plots", "fig7_dirichlet_form_equation1.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Saved Equation 1 Dirichlet form plot to {fig_path}.")

if __name__ == "__main__":
    evaluate_dirichlet_form_eq1()
