import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def assemble_master_figure():
    print("--- Phase 9: Assembling Unified Master Publication Figure ---")
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    img_paths = [
        os.path.join("plots", "phase3_equilibration.png"),
        os.path.join("plots", "phase4_stress_strain.png"),
        os.path.join("plots", "phase6_microstructure_graph.png"),
        os.path.join("plots", "phase7_gnn_training_loss.png"),
        os.path.join("plots", "phase8_nonlocal_kernel.png")
    ]
    
    titles = [
        "A) Polycrystalline Al NPT Thermal Equilibration",
        "B) Uniaxial Stress–Strain Response (MD)",
        "C) 3D Microstructure Graph Representation",
        "D) GNN Loss Convergence (Baseline GCN vs GAT)",
        "E) Microstructure-Informed Dirichlet Form Kernel"
    ]
    
    ax_flat = axes.flatten()
    
    for i, path in enumerate(img_paths):
        if os.path.exists(path):
            img = mpimg.imread(path)
            ax_flat[i].imshow(img)
            ax_flat[i].set_title(titles[i], fontsize=14, fontweight="bold", pad=10)
            ax_flat[i].axis("off")

    # Turn off unused 6th subplot
    ax_flat[5].axis("off")
    
    plt.suptitle("Microstructure-Informed Dirichlet Forms for Nonlocal Polycrystalline Fracture\nMultiscale Computational Proof-of-Concept Workflow", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    master_path = os.path.join("plots", "master_paper_figure.png")
    plt.savefig(master_path, dpi=300)
    print(f"Unified master publication figure saved to {master_path}.")

if __name__ == "__main__":
    assemble_master_figure()
