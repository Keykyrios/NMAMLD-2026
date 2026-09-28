import numpy as np
import json
import os
from scipy.spatial import KDTree

def euler_to_rotation_matrix(phi1, theta, phi2):
    """
    Convert Bunge Euler angles (in degrees) to 3D rotation matrix R.
    Z-X-Z convention.
    """
    p1 = np.radians(phi1)
    th = np.radians(theta)
    p2 = np.radians(phi2)

    R_z1 = np.array([
        [np.cos(p1), -np.sin(p1), 0],
        [np.sin(p1),  np.cos(p1), 0],
        [0,           0,          1]
    ])
    R_x  = np.array([
        [1, 0,           0],
        [0, np.cos(th), -np.sin(th)],
        [0, np.sin(th),  np.cos(th)]
    ])
    R_z2 = np.array([
        [np.cos(p2), -np.sin(p2), 0],
        [np.sin(p2),  np.cos(p2), 0],
        [0,           0,          1]
    ])
    return R_z2 @ R_x @ R_z1

def create_polycrystal(
    box_length=80.0,
    num_grains=12,
    lattice_param=4.05,
    min_dist=2.1,
    random_seed=42
):
    np.random.seed(random_seed)
    print(f"--- Generating Polycrystalline Al Microstructure ---")
    print(f"Box dimensions : {box_length} x {box_length} x {box_length} Å")
    print(f"Number of grains: {num_grains}")
    print(f"Random seed    : {random_seed}")

    # 1. Generate grain seed points within box
    grain_seeds = np.random.uniform(0, box_length, size=(num_grains, 3))

    # 2. Generate random Euler angles for each grain
    grains_info = []
    rotation_matrices = []
    for g_id in range(num_grains):
        phi1 = float(np.random.uniform(0, 360))
        theta = float(np.random.uniform(0, 180))
        phi2 = float(np.random.uniform(0, 360))
        R = euler_to_rotation_matrix(phi1, theta, phi2)
        rotation_matrices.append(R)
        grains_info.append({
            "grain_id": g_id + 1,
            "seed_pos": grain_seeds[g_id].tolist(),
            "euler_angles_deg": [phi1, theta, phi2],
            "rotation_matrix": R.tolist()
        })

    # 3. Create candidate FCC grid
    n_unit_cells = int(np.ceil(box_length / lattice_param))
    fcc_basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ]) * lattice_param

    grid_coords = []
    for i in range(-2, n_unit_cells + 2):
        for j in range(-2, n_unit_cells + 2):
            for k in range(-2, n_unit_cells + 2):
                base_pos = np.array([i, j, k]) * lattice_param
                for b in fcc_basis:
                    grid_coords.append(base_pos + b)
    grid_coords = np.array(grid_coords)

    # 4. Periodic distance assignment to nearest grain seed (Voronoi cell)
    # Replicate grain seeds 3x3x3 for periodic boundary conditions
    extended_seeds = []
    extended_grain_ids = []
    for shift_x in [-box_length, 0.0, box_length]:
        for shift_y in [-box_length, 0.0, box_length]:
            for shift_z in [-box_length, 0.0, box_length]:
                shift = np.array([shift_x, shift_y, shift_z])
                for g_id, seed in enumerate(grain_seeds):
                    extended_seeds.append(seed + shift)
                    extended_grain_ids.append(g_id + 1)

    extended_seeds = np.array(extended_seeds)
    extended_grain_ids = np.array(extended_grain_ids)
    tree = KDTree(extended_seeds)

    # 5. Assign atoms to grains and filter box boundaries
    _, nearest_indices = tree.query(grid_coords)
    assigned_grains = extended_grain_ids[nearest_indices]

    final_positions = []
    final_grain_ids = []

    for pos, g_id in zip(grid_coords, assigned_grains):
        R = rotation_matrices[g_id - 1]
        seed_pos = grain_seeds[g_id - 1]
        
        # Local position relative to grain seed
        rel_pos = pos - seed_pos
        # Rotate lattice orientation
        rot_pos = R @ rel_pos + seed_pos

        # Wrap / Keep within box [0, box_length)
        if 0.0 <= rot_pos[0] < box_length and \
           0.0 <= rot_pos[1] < box_length and \
           0.0 <= rot_pos[2] < box_length:
            final_positions.append(rot_pos)
            final_grain_ids.append(g_id)

    final_positions = np.array(final_positions)
    final_grain_ids = np.array(final_grain_ids)

    # 6. Remove overlapping atoms at grain boundaries (dist < min_dist)
    print(f"Candidate atoms generated: {len(final_positions)}. Filtering grain boundary overlaps...")
    pos_tree = KDTree(final_positions)
    pairs = pos_tree.query_pairs(r=min_dist)

    remove_indices = set()
    for (i, j) in pairs:
        remove_indices.add(j)

    keep_mask = np.array([i not in remove_indices for i in range(len(final_positions))])
    clean_positions = final_positions[keep_mask]
    clean_grain_ids = final_grain_ids[keep_mask]

    print(f"Final valid atom count after overlap filtering: {len(clean_positions)} atoms.")

    # 7. Update grain metadata atom counts
    for g in grains_info:
        g["atom_count"] = int(np.sum(clean_grain_ids == g["grain_id"]))

    # Save JSON metadata
    os.makedirs("data", exist_ok=True)
    metadata_path = os.path.join("data", "grains_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(grains_info, f, indent=2)
    print(f"Saved grain metadata to {metadata_path}.")

    # 8. Write LAMMPS Data File
    data_path = os.path.join("data", "polycrystal_Al.data")
    with open(data_path, "w") as f:
        f.write("# Polycrystalline Aluminum Data File for LAMMPS\n")
        f.write(f"# Grains: {num_grains}, Box: {box_length}x{box_length}x{box_length} A\n\n")
        f.write(f"{len(clean_positions)} atoms\n")
        f.write(f"{num_grains} atom types\n\n")
        f.write(f"0.0 {box_length:.6f} xlo xhi\n")
        f.write(f"0.0 {box_length:.6f} ylo yhi\n")
        f.write(f"0.0 {box_length:.6f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for g_id in range(1, num_grains + 1):
            f.write(f"{g_id} 26.981539\n")
        f.write("\nAtoms # atomic\n\n")
        for i, (pos, g_id) in enumerate(zip(clean_positions, clean_grain_ids)):
            f.write(f"{i+1} {g_id} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

    print(f"Successfully exported LAMMPS data file to {data_path}.")
    return len(clean_positions), num_grains

if __name__ == "__main__":
    create_polycrystal()
