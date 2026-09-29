import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def generate_all_figures():
    print("--- Phase K: Generating Corrected Publication Figures ---")
    os.makedirs("plots", exist_ok=True)

    # 1. Stress-Strain Response
    df_ss = pd.read_csv("data/stress_strain_data.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df_ss["v_strain"] * 100.0, df_ss["v_stress_x"], color="#c0392b", lw=2.5, label=r"Axial Tensile Stress $\sigma_{xx}$")
    ax.plot(df_ss["v_strain"] * 100.0, df_ss["v_stress_y"], color="#2980b9", lw=1.5, linestyle="--", label=r"Transverse $\sigma_{yy}$")
    ax.plot(df_ss["v_strain"] * 100.0, df_ss["v_stress_z"], color="#27ae60", lw=1.5, linestyle="--", label=r"Transverse $\sigma_{zz}$")
    ax.set_xlabel("Engineering Strain [%]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Stress [GPa]", fontsize=12, fontweight="bold")
    ax.set_title("Atomistic Uniaxial Stress–Strain Response (Polycrystalline Al)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/fig1_stress_strain.png", dpi=300)
    plt.close()

    # 2. 3D Grain Graph Representation
    nodes_df = pd.read_csv("data/multisample_nodes.csv")
    edges_df = pd.read_csv("data/multisample_edges.csv")
    sample1_nodes = nodes_df[nodes_df["sample_id"] == 1].sort_values("grain_id").reset_index(drop=True)
    sample1_edges = edges_df[edges_df["sample_id"] == 1]

    fig = plt.figure(figsize=(8, 6))
    ax3d = fig.add_subplot(111, projection='3d')
    c = sample1_nodes[["centroid_x", "centroid_y", "centroid_z"]].values
    dmg = sample1_nodes["grain_damage_index"].values
    sc = ax3d.scatter(c[:,0], c[:,1], c[:,2], c=dmg, cmap="plasma", s=200, edgecolors="black")
    gid_to_idx = {int(gid): idx for idx, gid in enumerate(sample1_nodes["grain_id"].values)}
    for _, r in sample1_edges.iterrows():
        u = gid_to_idx.get(int(r["source_grain"]), -1)
        v = gid_to_idx.get(int(r["target_grain"]), -1)
        if u < 0 or v < 0:
            continue
        p1, p2 = c[u], c[v]
        ax3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="gray", alpha=0.5)
    ax3d.set_title("3D Microstructure Grain Boundary Graph (Sample 01)", fontsize=12, fontweight="bold")
    fig.colorbar(sc, ax=ax3d, shrink=0.6, label="Post-Loading Damage Index")
    plt.tight_layout()
    plt.savefig("plots/fig2_microstructure_graph.png", dpi=300)
    plt.close()

    # 3. Train vs Validation vs Held-Out Test Predictions
    results_df = pd.read_csv("data/gnn_results.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(results_df["true_target"], results_df["gat_predicted_target"], color="#8e44ad", s=60, edgecolors="black", label="Held-Out Test Nodes (Samples 09 & 10)")
    min_val = min(results_df["true_target"].min(), results_df["gat_predicted_target"].min()) - 1
    max_val = max(results_df["true_target"].max(), results_df["gat_predicted_target"].max()) + 1
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="1:1 Perfect Prediction")
    ax.set_xlabel("True Grain Damage Index (Atomistic Ground Truth)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Microstructure GAT Predicted Damage", fontsize=11, fontweight="bold")
    metrics_df = pd.read_csv("data/model_metrics.csv")
    r2_gat = metrics_df["Microstructure_GAT_Test_R2_mean"].iloc[0]
    ax.set_title(f"Held-Out Test Set Prediction Accuracy (R² = {r2_gat:.3f})", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/fig3_held_out_test_predictions.png", dpi=300)
    plt.close()

    # 4. GCN vs GAT Held-Out Test Performance Comparison
    metrics_df = pd.read_csv("data/model_metrics.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    models = ["Baseline GCN", "Proposed Microstructure GAT"]
    mses = [metrics_df["Baseline_GCN_Test_MSE_mean"].iloc[0], metrics_df["Microstructure_GAT_Test_MSE_mean"].iloc[0]]
    stds = [metrics_df["Baseline_GCN_Test_MSE_std"].iloc[0], metrics_df["Microstructure_GAT_Test_MSE_std"].iloc[0]]
    bars = ax.bar(models, mses, yerr=stds, capsize=8, color=["#7f8c8d", "#8e44ad"], width=0.5, edgecolor="black")
    ax.set_ylabel("Held-Out Test MSE", fontsize=12, fontweight="bold")
    ax.set_title("Held-Out Test Set Error Comparison (5 Seeds)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/fig4_gcn_vs_gat_test_performance.png", dpi=300)
    plt.close()

    # 5. Attention Distribution vs Misorientation Angle
    att_df = pd.read_csv("data/learned_attention_weights.csv")
    test_att = att_df[att_df["split"] == "test"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(test_att["misorientation_deg"], test_att["attention_weight"], color="#27ae60", alpha=0.7, edgecolors="black")
    ax.set_xlabel("Misorientation Angle θ [deg]", fontsize=11, fontweight="bold")
    ax.set_ylabel("GAT Learned Attention Weight α_ij", fontsize=11, fontweight="bold")
    from scipy.stats import spearmanr
    rho, pval = spearmanr(test_att["misorientation_deg"], test_att["attention_weight"])
    ax.set_title(f"Learned Attention vs. Boundary Misorientation (ρ = {rho:.3f})", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/fig5_attention_vs_damage.png", dpi=300)
    plt.close()

    # 6. Nonlocal Kernel Field
    sens_df = pd.read_csv("data/nonlocal_kernel_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(sens_df["gamma"], sens_df["max_kernel_intensity"], "o-", color="#e74c3c", lw=2.5, ms=8, label="Max Kernel Intensity")
    ax.plot(sens_df["gamma"], sens_df["mean_nonzero_kernel"], "s--", color="#2980b9", lw=2.0, ms=7, label="Mean Nonzero Kernel")
    ax.set_xlabel("Coupling Parameter γ", fontsize=12, fontweight="bold")
    ax.set_ylabel("Nonlocal Dirichlet Kernel Intensity", fontsize=12, fontweight="bold")
    ax.set_title("Nonlocal Kernel Sensitivity to Coupling Parameter γ", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/fig6_gamma_sensitivity.png", dpi=300)
    plt.close()

    # 7. Assemble Unified Master Publication Figure (3x2 Grid)
    fig_master, axes_m = plt.subplots(3, 2, figsize=(15, 18))
    fig_list = [
        "plots/fig1_stress_strain.png",
        "plots/fig2_microstructure_graph.png",
        "plots/fig3_held_out_test_predictions.png",
        "plots/fig4_gcn_vs_gat_test_performance.png",
        "plots/fig5_attention_vs_damage.png",
        "plots/fig6_gamma_sensitivity.png"
    ]
    sub_titles = [
        "A) Uniaxial Stress–Strain Response (MD)",
        "B) 3D Microstructure Graph Representation",
        "C) Held-Out Test Set Damage Prediction Accuracy",
        "D) Held-Out Test Error (Baseline GCN vs GAT)",
        "E) GAT Attention vs. Boundary Misorientation",
        "F) Nonlocal Kernel Sensitivity Analysis (γ)"
    ]

    for idx, path in enumerate(fig_list):
        row, col = divmod(idx, 2)
        img = plt.imread(path)
        axes_m[row, col].imshow(img)
        axes_m[row, col].set_title(sub_titles[idx], fontsize=13, fontweight="bold", pad=8)
        axes_m[row, col].axis("off")

    plt.suptitle("Microstructure-Informed Dirichlet Forms for Nonlocal Polycrystalline Fracture\nMultiscale Computational Proof-of-Concept Workflow", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig("plots/master_paper_figure.png", dpi=300)
    plt.close()
    print("Master publication figure assembled and saved to plots/master_paper_figure.png.")

if __name__ == "__main__":
    generate_all_figures()
