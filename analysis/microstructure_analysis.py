import numpy as np
import json
import os
import pandas as pd

def calculate_fcc_taylor_factor(R):
    """
    Calculates the Taylor factor M for an FCC crystal orientation R under uniaxial loading along X.
    Uses the 12 FCC {111}<110> slip systems.
    """
    # 4 {111} plane normals
    normals = np.array([
        [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1]
    ]) / np.sqrt(3)

    # 3 <110> directions per plane -> 12 slip systems
    directions = np.array([
        [ 1, -1,  0], [ 1,  0, -1], [ 0,  1, -1],
        [ 1, -1,  0], [ 1,  0,  1], [ 0,  1,  1],
        [ 1,  1,  0], [ 1,  0, -1], [ 0,  1,  1],
        [ 1,  1,  0], [ 1,  0,  1], [ 0,  1, -1]
    ]) / np.sqrt(2)

    # Loading direction in sample reference frame: d = [1, 0, 0]
    d_sample = np.array([1.0, 0.0, 0.0])
    
    # Rotate loading direction into crystal reference frame
    d_crystal = R.T @ d_sample

    schmid_factors = []
    for n, b in zip(normals, directions):
        m = abs(np.dot(n, d_crystal) * np.dot(b, d_crystal))
        schmid_factors.append(m)

    max_schmid = max(schmid_factors)
    if max_schmid < 1e-4:
        return 3.06
    return float(np.clip(1.0 / max_schmid, 2.0, 4.5))

def calculate_read_shockley_gb_energy(misorientation_deg, gamma_0=0.56, theta_m=15.0):
    """
    Calculates the Read-Shockley grain boundary interface energy (in J/m^2) as a function of misorientation angle.
    """
    theta = abs(misorientation_deg)
    if theta <= 1e-3:
        return 0.0
    if theta >= theta_m:
        return float(gamma_0)
    
    ratio = theta / theta_m
    return float(gamma_0 * ratio * (1.0 - np.log(ratio)))

def parse_lammpstrj(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    timestep_indices = [i for i, line in enumerate(lines) if "ITEM: TIMESTEP" in line]
    last_idx = timestep_indices[-1]

    box_idx = last_idx + 5
    box_x = [float(x) for x in lines[box_idx].split()]
    box_y = [float(x) for x in lines[box_idx+1].split()]
    box_z = [float(x) for x in lines[box_idx+2].split()]
    box_len = [box_x[1] - box_x[0], box_y[1] - box_y[0], box_z[1] - box_z[0]]

    atoms_idx = last_idx + 8
    atom_lines = lines[atoms_idx+1:]
    
    data = []
    for l in atom_lines:
        if "ITEM: TIMESTEP" in l:
            break
        parts = l.split()
        if len(parts) >= 5:
            data.append([int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

    df_atoms = pd.DataFrame(data, columns=["id", "type", "x", "y", "z"]).sort_values("id").reset_index(drop=True)
    return box_len, df_atoms

def calculate_misorientation(R1, R2):
    R_rel = np.array(R1) @ np.array(R2).T
    trace = np.trace(R_rel)
    val = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta_rad = np.arccos(val)
    return float(np.degrees(theta_rad))

def analyze_microstructure():
    print("--- Microstructure & Feature Extraction with Taylor Factors & GB Energies ---")
    
    with open("data/grains_metadata.json", "r") as f:
        grains_info = json.load(f)
    num_grains = len(grains_info)

    traj_path = os.path.join("simulations", "tensile.lammpstrj")
    box_len, df_deformed = parse_lammpstrj(traj_path)

    node_features = []
    for g in grains_info:
        g_id = g["grain_id"]
        atoms_g = df_deformed[df_deformed["type"] == g_id]
        
        centroid = atoms_g[["x", "y", "z"]].mean().values
        std_pos = atoms_g[["x", "y", "z"]].std().values
        
        R = np.array(g["rotation_matrix"])
        taylor_M = calculate_fcc_taylor_factor(R)
        
        node_features.append({
            "grain_id": g_id,
            "atom_count": len(atoms_g),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "euler_phi1": g["euler_angles_deg"][0],
            "euler_theta": g["euler_angles_deg"][1],
            "euler_phi2": g["euler_angles_deg"][2],
            "taylor_factor": taylor_M,
            "volume_est": float(np.prod(std_pos * 2.0)),
            "grain_damage_index": float(np.mean(std_pos))
        })

    df_nodes = pd.DataFrame(node_features)

    edge_features = []
    cutoff_dist = 45.0

    for i in range(num_grains):
        for j in range(i + 1, num_grains):
            g1 = grains_info[i]
            g2 = grains_info[j]

            c1 = np.array([df_nodes.loc[i, "centroid_x"], df_nodes.loc[i, "centroid_y"], df_nodes.loc[i, "centroid_z"]])
            c2 = np.array([df_nodes.loc[j, "centroid_x"], df_nodes.loc[j, "centroid_y"], df_nodes.loc[j, "centroid_z"]])

            delta = np.abs(c1 - c2)
            delta = np.where(delta > 0.5 * np.array(box_len), np.array(box_len) - delta, delta)
            dist = float(np.linalg.norm(delta))

            if dist <= cutoff_dist:
                misorient = calculate_misorientation(g1["rotation_matrix"], g2["rotation_matrix"])
                gb_energy = calculate_read_shockley_gb_energy(misorient)
                
                # Edge features: distance, misorientation, GB energy
                # NOTE: interface_damage_diff REMOVED — it is |D_i - D_j| derived from the target
                edge_features.append({
                    "source_grain": g1["grain_id"],
                    "target_grain": g2["grain_id"],
                    "distance_A": dist,
                    "misorientation_deg": misorient,
                    "gb_interface_energy_Jm2": gb_energy
                })

    df_edges = pd.DataFrame(edge_features)

    nodes_csv = os.path.join("data", "graph_nodes.csv")
    edges_csv = os.path.join("data", "graph_edges.csv")
    df_nodes.to_csv(nodes_csv, index=False)
    df_edges.to_csv(edges_csv, index=False)
    print(f"Extracted node features (including Taylor Factor M) and edge features (including Read-Shockley GB Energy γ_GB). Saved to {nodes_csv} and {edges_csv}.")

if __name__ == "__main__":
    analyze_microstructure()
