import subprocess
import os
import json
import time
import pandas as pd
import numpy as np
from scipy.spatial import KDTree

from microstructure_analysis import parse_lammpstrj, calculate_misorientation, calculate_fcc_taylor_factor, calculate_read_shockley_gb_energy, compute_voronoi_adjacency

lmp_path = r"C:\Users\mitra\AppData\Local\LAMMPS 64-bit 4Jul2026\bin\lmp.exe"
data_dir = "data"
sim_dir = "simulations"
checkpoint_dir = os.path.join(data_dir, "checkpoints")

def get_box_len_for_grains(num_grains):
    """
    Returns an appropriate simulation box length (in Angstroms) for a given grain count.
    Calibrated from existing benchmark data:
      200 grains -> 160 A, 300 -> 180, 400 -> 200, 500 -> 220
    Uses linear interpolation / extrapolation from these reference points.
    """
    ref_grains = [200, 300, 400, 500]
    ref_box    = [160.0, 180.0, 200.0, 220.0]
    return float(np.interp(num_grains, ref_grains, ref_box))

def save_checkpoint(sample_id, result):
    """Save a per-sample checkpoint JSON so we can resume after crash."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"sample_{sample_id:02d}_complete.json")
    with open(ckpt_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [CHECKPOINT] Saved checkpoint for sample {sample_id:02d} -> {ckpt_path}")

def load_checkpoint(sample_id):
    """Load a checkpoint if it exists. Returns the result dict or None."""
    ckpt_path = os.path.join(checkpoint_dir, f"sample_{sample_id:02d}_complete.json")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            return json.load(f)
    return None

def run_single_simulation_sample(sample_id, seed, num_grains):
    """
    Full pipeline for one polycrystalline specimen:
      1. Generate Voronoi polycrystal LAMMPS data file
      2. Run LAMMPS MD (minimize + equilibrate + tensile)
      3. Extract node features (Euler angles, Taylor factor, volume, damage index)
      4. Extract edge features (distance, misorientation, GB energy) — NO damage_diff
      5. Return results dict
    """
    box_len = get_box_len_for_grains(num_grains)
    atom_count = 0  # Will be set during polycrystal generation or header read

    print(f"\n==================================================")
    print(f"   STARTING SAMPLE {sample_id:02d} | Seed: {seed} | Grains: {num_grains} | Box: {box_len:.0f} A")
    print(f"==================================================")
    
    sample_prefix = f"sample_{sample_id:02d}"
    data_filename = f"{sample_prefix}_polycrystal.data"
    meta_filename = f"{sample_prefix}_metadata.json"
    
    data_path = os.path.join(data_dir, data_filename)
    meta_path = os.path.join(data_dir, meta_filename)
    
    # --- 1. Generate Voronoi polycrystal ---
    np.random.seed(seed)
    grain_seeds = np.random.uniform(0, box_len, size=(num_grains, 3))
    
    grains_info = []
    rotation_matrices = []
    for g_id in range(num_grains):
        phi1 = float(np.random.uniform(0, 360))
        theta = float(np.degrees(np.arccos(np.random.uniform(-1, 1))))  # proper SO(3) sampling
        phi2 = float(np.random.uniform(0, 360))
        
        p1, th, p2 = np.radians(phi1), np.radians(theta), np.radians(phi2)
        R_z1 = np.array([[np.cos(p1), -np.sin(p1), 0], [np.sin(p1), np.cos(p1), 0], [0, 0, 1]])
        R_x  = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
        R_z2 = np.array([[np.cos(p2), -np.sin(p2), 0], [np.sin(p2), np.cos(p2), 0], [0, 0, 1]])
        R = R_z2 @ R_x @ R_z1
        
        rotation_matrices.append(R)
        grains_info.append({
            "grain_id": g_id + 1,
            "seed_pos": grain_seeds[g_id].tolist(),
            "euler_angles_deg": [phi1, theta, phi2],
            "rotation_matrix": R.tolist()
        })

    lattice_param = 4.05
    n_unit_cells = int(np.ceil(box_len / lattice_param))
    fcc_basis = np.array([[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]) * lattice_param
    
    # Only regenerate polycrystal data file if it does not already exist
    if not os.path.exists(data_path):
        print(f"  Generating Voronoi polycrystal ({num_grains} grains, {box_len:.0f} A box)...")

        # Generate FCC lattice grid with padding for rotation coverage.
        # Rotated FCC sites belonging to a grain's Voronoi cell can lie up to
        # ~0.7 * seed_spacing away from the seed (corners of the cell), and
        # corner seeds need coverage outside the box, so pad scales with the
        # characteristic seed spacing instead of being a fixed constant.
        seed_spacing = (box_len ** 3 / num_grains) ** (1.0 / 3.0)
        pad = int(np.ceil(0.7 * seed_spacing / lattice_param)) + 1
        # How far each grain's lattice may protrude past its Voronoi boundary
        # (A). Set to ~2x the FCC nn spacing so both lattices interpenetrate
        # ~1.5 A past the shared boundary; the overlap-removal step then
        # deduplicates, leaving an atomically-graded GB instead of a vacuum
        # slab. Validated by scratch LAMMPS minimization (E/atom within ~2%
        # of bulk FCC) during the audit.
        boundary_overlap = 3.0
        grid_coords = []
        for i in range(-pad, n_unit_cells + pad + 1):
            for j in range(-pad, n_unit_cells + pad + 1):
                for k in range(-pad, n_unit_cells + pad + 1):
                    base_pos = np.array([i, j, k]) * lattice_param
                    for b in fcc_basis:
                        grid_coords.append(base_pos + b)
        grid_coords = np.array(grid_coords)

        # Build periodic Voronoi assignment tree (27 periodic images)
        extended_seeds, extended_ids = [], []
        for sx in [-box_len, 0.0, box_len]:
            for sy in [-box_len, 0.0, box_len]:
                for sz in [-box_len, 0.0, box_len]:
                    shift = np.array([sx, sy, sz])
                    for g_id, s in enumerate(grain_seeds):
                        extended_seeds.append(s + shift)
                        extended_ids.append(g_id + 1)
        extended_seeds = np.array(extended_seeds)
        extended_ids = np.array(extended_ids)
        voronoi_tree = KDTree(extended_seeds)

        # Per-grain lattice generation: rotate FIRST, then Voronoi-assign.
        # Each grain generates its own oriented FCC lattice; only atoms
        # whose nearest Voronoi seed (after rotation) is this grain are kept.
        # This avoids the gap/overlap artifacts of assign-then-rotate.
        all_pos, all_gids = [], []
        for g_idx in range(num_grains):
            g_id = g_idx + 1
            R = rotation_matrices[g_idx]
            seed = grain_seeds[g_idx]

            # Rotate the global lattice into this grain's crystallographic orientation
            centered = grid_coords - seed
            rotated = (R @ centered.T).T + seed

            # Box filter: keep only atoms inside [0, box_len)^3
            in_box = np.all((rotated >= 0) & (rotated < box_len), axis=1)
            candidates = rotated[in_box]

            # Voronoi assignment on ROTATED positions: keep atoms whose
            # nearest grain seed is this grain, PLUS atoms within
            # boundary_overlap of this grain's cell (second-nearest seed is
            # this grain and only slightly farther than the nearest). Both
            # neighboring lattices therefore interpenetrate in the boundary
            # band and the overlap-removal step below deduplicates them,
            # filling the sub-atomic vacuum gaps that would otherwise remain
            # where two mismatched FCC lattices meet (~20% void without this).
            dd, nearest_idx = voronoi_tree.query(candidates, k=2)
            assigned_ids = extended_ids[nearest_idx[:, 0]]
            own_mask = (assigned_ids == g_id) | (
                (extended_ids[nearest_idx[:, 1]] == g_id)
                & ((dd[:, 1] - dd[:, 0]) < boundary_overlap)
            )

            own_atoms = candidates[own_mask]
            all_pos.append(own_atoms)
            all_gids.extend([g_id] * len(own_atoms))

            if (g_idx + 1) % 50 == 0 or g_idx == num_grains - 1:
                print(f"    Grain {g_idx + 1}/{num_grains}: {len(own_atoms)} atoms")

        final_pos = np.vstack(all_pos)
        final_gids = np.array(all_gids)
        print(f"  Total atoms before overlap removal: {len(final_pos)}")

        # Remove overlapping atoms at grain boundaries.
        # Fixed: only remove j if partner i is still kept,
        # preventing chain over-deletion at multi-atom boundary clusters.
        pos_tree = KDTree(final_pos)
        pairs = pos_tree.query_pairs(r=2.1)
        remove_idx = set()
        for (i, j) in sorted(pairs):
            if i not in remove_idx and j not in remove_idx:
                remove_idx.add(j)
        keep_mask = np.array([idx not in remove_idx for idx in range(len(final_pos))])
        clean_pos = final_pos[keep_mask]
        clean_gids = final_gids[keep_mask]

        for g in grains_info:
            g["atom_count"] = int(np.sum(clean_gids == g["grain_id"]))

        with open(meta_path, "w") as f:
            json.dump(grains_info, f, indent=2)

        with open(data_path, "w") as f:
            f.write(f"# Polycrystalline Al Sample {sample_id}\n\n")
            f.write(f"{len(clean_pos)} atoms\n{num_grains} atom types\n\n")
            f.write(f"0.0 {box_len:.6f} xlo xhi\n0.0 {box_len:.6f} ylo yhi\n0.0 {box_len:.6f} zlo zhi\n\nMasses\n\n")
            for g_id in range(1, num_grains + 1):
                f.write(f"{g_id} 26.981539\n")
            f.write("\nAtoms # atomic\n\n")
            for i, (p, g_id) in enumerate(zip(clean_pos, clean_gids)):
                f.write(f"{i+1} {g_id} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        
        atom_count = len(clean_pos)
        print(f"  Generated {data_path} ({atom_count} atoms, {num_grains} grains)")
    else:
        print(f"  Polycrystal data file already exists: {data_path}. Skipping generation.")
        # Reload metadata
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                grains_info = json.load(f)
        # Count atoms from file header
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.endswith("atoms"):
                    atom_count = int(line.split()[0])
                    break

    # --- 2. Run LAMMPS MD ---
    in_path = os.path.join(sim_dir, f"{sample_prefix}_run.in")
    log_path = os.path.join(sim_dir, f"{sample_prefix}.log")
    traj_path = os.path.join(sim_dir, f"{sample_prefix}_tensile.lammpstrj")
    
    pair_coeffs = " ".join(["Al"] * num_grains)

    # Validate trajectory if it exists — must have 11 frames (dumps at steps 0,1000,...,10000)
    traj_valid = False
    if os.path.exists(traj_path):
        with open(traj_path, "r") as f:
            frame_count = sum(1 for line in f if "ITEM: TIMESTEP" in line)
        if frame_count >= 11:
            traj_valid = True
            print(f"  Trajectory validated: {traj_path} ({frame_count} frames). Skipping MD run.")
        else:
            print(f"  [WARNING] Incomplete trajectory detected ({frame_count}/11 frames). Deleting and re-running.")
            os.remove(traj_path)

    if not traj_valid:
        lammps_script = f"""# Sample {sample_id:02d} Simulation Pipeline ({num_grains} grains, {box_len:.0f} A box)
units           metal
dimension       3
boundary        p p p
atom_style      atomic

read_data       ../data/{data_filename}

pair_style      eam/alloy
pair_coeff      * * ../data/Al_zhou.eam.alloy {pair_coeffs}

neighbor        2.0 bin
neigh_modify    every 1 delay 5 check yes

# Stage 1: Energy Minimization
minimize        1.0e-15 1.0e-15 1000 10000

# Stage 2: NPT Equilibration
reset_timestep  0
timestep        0.001
velocity        all create 300.0 {seed} mom yes rot yes dist gaussian
fix             npt_eq all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0
run             1500
unfix           npt_eq

# Stage 3: Tensile Loading (10,000 steps -> 10% strain)
reset_timestep  0
dump            d_tensile all custom 1000 {sample_prefix}_tensile.lammpstrj id type x y z ix iy iz
fix             deform_x all deform 1 x erate 0.01 remap x
fix             npt_tr all npt temp 300.0 300.0 0.1 y 0.0 0.0 1.0 z 0.0 0.0 1.0
run             10000
"""
        with open(in_path, "w") as f:
            f.write(lammps_script)

        print(f"  Running LAMMPS MD for sample {sample_id:02d} ({num_grains} grains, ~{atom_count} atoms)...")
        print(f"  This may take several hours for large grain counts.")
        start_time = time.time()
        cmd = [lmp_path, "-pk", "omp", "8", "-sf", "omp", "-in", f"{sample_prefix}_run.in", "-log", f"{sample_prefix}.log"]
        result = subprocess.run(cmd, cwd=sim_dir, capture_output=True, text=True)
        runtime = time.time() - start_time

        if result.returncode != 0:
            print(f"  [ERROR] Sample {sample_id:02d} LAMMPS run failed!")
            print(f"  STDERR: {result.stderr[:500]}")
            return None
        
        print(f"  LAMMPS completed in {runtime:.1f} s ({runtime/3600:.2f} hours)")
    else:
        runtime = 0.0

    # --- 3. Validate trajectory before extracting features ---
    print(f"  Validating trajectory integrity...")
    with open(traj_path, "r") as f:
        frame_count = sum(1 for line in f if "ITEM: TIMESTEP" in line)
    if frame_count < 11:
        print(f"  [FATAL] Trajectory has only {frame_count}/11 frames. Cannot extract features.")
        print(f"  Deleting incomplete trajectory so next run will redo LAMMPS.")
        os.remove(traj_path)
        return None
    print(f"  Trajectory OK: {frame_count} frames.")

    # --- 4. Extract node features ---
    print(f"  Extracting microstructure features from trajectory...")
    box_bounds, df_deformed = parse_lammpstrj(traj_path)

    sample_nodes = []
    for g in grains_info:
        g_id = g["grain_id"]
        atoms_g = df_deformed[df_deformed["type"] == g_id]
        
        if len(atoms_g) == 0:
            continue
        
        centroid = atoms_g[["x", "y", "z"]].mean().values
        std_pos = atoms_g[["x", "y", "z"]].std().values
        
        R = np.array(g["rotation_matrix"])
        taylor_M = calculate_fcc_taylor_factor(R)
        
        sample_nodes.append({
            "sample_id": sample_id,
            "grain_id": g_id,
            "atom_count": len(atoms_g),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "euler_phi1": g["euler_angles_deg"][0],
            "euler_theta": g["euler_angles_deg"][1],
            "euler_phi2": g["euler_angles_deg"][2],
            "taylor_factor": taylor_M,
            # Physical volume estimate: atoms * atomic volume of FCC Al.
            # (The old prod(2*std_pos) estimate collapsed for grains that
            # straddle a periodic boundary and is shape-artifact-prone.)
            "volume_est": float(len(atoms_g) * (lattice_param ** 3) / 4.0),
            "grain_damage_index": float(np.mean(std_pos))
        })

    # --- 5. Extract edge features (Voronoi face-adjacent grain pairs only) ---
    sample_edges = []

    # Adjacency from the periodic Voronoi tessellation of the grain seeds:
    # edges are true grain-boundary contacts (shared Voronoi faces), not all
    # pairs within some arbitrary distance cutoff.
    seeds_arr = np.array([g["seed_pos"] for g in grains_info])
    adjacency = compute_voronoi_adjacency(seeds_arr, box_len)

    # Build centroid array for efficient distance computation
    centroids = np.array([[n["centroid_x"], n["centroid_y"], n["centroid_z"]] for n in sample_nodes])
    node_grain_ids = [n["grain_id"] for n in sample_nodes]
    centroid_of_gid = dict(zip(node_grain_ids, centroids))

    for gid_a, gid_b in adjacency:
        # Both endpoint grains must have survived feature extraction
        if gid_a not in centroid_of_gid or gid_b not in centroid_of_gid:
            continue
        c1 = centroid_of_gid[gid_a]
        c2 = centroid_of_gid[gid_b]

        # Minimum image convention for periodic boundaries
        delta = np.abs(c1 - c2)
        delta = np.where(delta > 0.5 * np.array(box_bounds), np.array(box_bounds) - delta, delta)
        dist = float(np.linalg.norm(delta))

        g1_info = grains_info[gid_a - 1]
        g2_info = grains_info[gid_b - 1]
        misorient = calculate_misorientation(g1_info["rotation_matrix"], g2_info["rotation_matrix"])
        gb_energy = calculate_read_shockley_gb_energy(misorient)

        # EDGE FEATURES: distance, misorientation, GB energy
        sample_edges.append({
            "sample_id": sample_id,
            "source_grain": gid_a,
            "target_grain": gid_b,
            "distance_A": dist,
            "misorientation_deg": misorient,
            "gb_interface_energy_Jm2": gb_energy
        })

    print(f"  Extracted {len(sample_nodes)} nodes, {len(sample_edges)} edges for sample {sample_id:02d}")

    # --- 6. Data integrity checks before returning ---
    if len(sample_nodes) < num_grains * 0.5:
        print(f"  [FATAL] Only {len(sample_nodes)} nodes extracted for {num_grains} grains. Data corrupt.")
        return None
    if len(sample_edges) == 0:
        print(f"  [FATAL] Zero edges extracted. Data corrupt.")
        return None
    damage_values = [n["grain_damage_index"] for n in sample_nodes]
    damage_std = np.std(damage_values)
    if damage_std < 1e-10:
        print(f"  [WARNING] All damage indices are identical (std={damage_std:.2e}). Check if trajectory has deformation.")
    print(f"  Data integrity check PASSED. Damage index range: [{min(damage_values):.4f}, {max(damage_values):.4f}], std={damage_std:.4f}")

    return {
        "sample_id": sample_id,
        "seed": seed,
        "num_grains": num_grains,
        "atom_count": atom_count,
        "runtime_s": runtime,
        "nodes": sample_nodes,
        "edges": sample_edges
    }

def main():
    print("=" * 60)
    print("  MULTI-SAMPLE DATASET PIPELINE (200-500 GRAINS)")
    print("  With per-sample checkpointing for crash recovery")
    print("=" * 60)
    
    configs = [
        {"sample_id": 1,  "seed": 101, "num_grains": 200},
        {"sample_id": 2,  "seed": 102, "num_grains": 250},
        {"sample_id": 3,  "seed": 103, "num_grains": 300},
        {"sample_id": 4,  "seed": 104, "num_grains": 350},
        {"sample_id": 5,  "seed": 105, "num_grains": 400},
        {"sample_id": 6,  "seed": 106, "num_grains": 250},
        {"sample_id": 7,  "seed": 107, "num_grains": 500},
        {"sample_id": 8,  "seed": 108, "num_grains": 300},  # Validation
        {"sample_id": 9,  "seed": 109, "num_grains": 400},  # Test 1
        {"sample_id": 10, "seed": 110, "num_grains": 500},  # Test 2
    ]

    all_manifest = []
    all_nodes = []
    all_edges = []

    for cfg in configs:
        sid = cfg["sample_id"]
        
        # Check for existing checkpoint
        ckpt = load_checkpoint(sid)
        if ckpt is not None:
            # Re-validate: make sure the trajectory file is still complete
            traj_path = os.path.join(sim_dir, f"sample_{sid:02d}_tensile.lammpstrj")
            if os.path.exists(traj_path):
                with open(traj_path, "r") as f:
                    frame_count = sum(1 for line in f if "ITEM: TIMESTEP" in line)
                if frame_count < 11:
                    print(f"\n[CHECKPOINT INVALID] Sample {sid:02d} checkpoint exists but trajectory has only {frame_count}/11 frames.")
                    print(f"  Deleting bad checkpoint and re-running.")
                    os.remove(os.path.join(checkpoint_dir, f"sample_{sid:02d}_complete.json"))
                    os.remove(traj_path)
                    # Fall through to run_single_simulation_sample below
                else:
                    print(f"\n[CHECKPOINT] Sample {sid:02d} validated ({frame_count} frames). Loading from checkpoint.")
                    all_manifest.append({
                        "sample_id": ckpt["sample_id"],
                        "seed": ckpt["seed"],
                        "num_grains": ckpt["num_grains"],
                        "atom_count": ckpt["atom_count"],
                        "wall_clock_s": round(ckpt["runtime_s"], 2),
                        "potential": "Al_zhou.eam.alloy",
                        "strain_rate": "0.01 ps^-1",
                        "max_strain": "0.10"
                    })
                    all_nodes.extend(ckpt["nodes"])
                    all_edges.extend(ckpt["edges"])
                    continue
            else:
                print(f"\n[CHECKPOINT INVALID] Sample {sid:02d} checkpoint exists but trajectory file is missing.")
                print(f"  Deleting bad checkpoint and re-running.")
                os.remove(os.path.join(checkpoint_dir, f"sample_{sid:02d}_complete.json"))
                # Fall through to run_single_simulation_sample below
        
        # Run the full pipeline for this sample
        res = run_single_simulation_sample(cfg["sample_id"], cfg["seed"], cfg["num_grains"])
        if res is not None:
            all_manifest.append({
                "sample_id": res["sample_id"],
                "seed": res["seed"],
                "num_grains": res["num_grains"],
                "atom_count": res["atom_count"],
                "wall_clock_s": round(res["runtime_s"], 2),
                "potential": "Al_zhou.eam.alloy",
                "strain_rate": "0.01 ps^-1",
                "max_strain": "0.10"
            })
            all_nodes.extend(res["nodes"])
            all_edges.extend(res["edges"])
            
            # Save checkpoint for this sample
            save_checkpoint(sid, res)
            
            # Also save incremental CSVs after each sample (crash safety)
            pd.DataFrame(all_manifest).to_csv(os.path.join(data_dir, "dataset_manifest.csv"), index=False)
            pd.DataFrame(all_nodes).to_csv(os.path.join(data_dir, "multisample_nodes.csv"), index=False)
            pd.DataFrame(all_edges).to_csv(os.path.join(data_dir, "multisample_edges.csv"), index=False)
            print(f"  [CHECKPOINT] Incremental CSVs saved ({len(all_manifest)} samples complete)")
        else:
            print(f"  [ERROR] Sample {sid:02d} failed. Continuing to next sample.")

    # Final CSV export
    df_manifest = pd.DataFrame(all_manifest)
    df_nodes = pd.DataFrame(all_nodes)
    df_edges = pd.DataFrame(all_edges)

    df_manifest.to_csv(os.path.join(data_dir, "dataset_manifest.csv"), index=False)
    df_nodes.to_csv(os.path.join(data_dir, "multisample_nodes.csv"), index=False)
    df_edges.to_csv(os.path.join(data_dir, "multisample_edges.csv"), index=False)

    print("\n==========================================")
    print("      MULTI-SAMPLE DATASET GENERATED      ")
    print("==========================================")
    print(df_manifest)
    print(f"\nTotal Independent Samples  : {len(df_manifest)}")
    print(f"Total Grain Nodes Extracted : {len(df_nodes)} (Includes Taylor Factor M)")
    print(f"Total Grain Boundary Edges  : {len(df_edges)} (Includes Read-Shockley GB Energy gamma_GB)")


if __name__ == "__main__":
    main()
