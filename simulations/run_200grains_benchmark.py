import subprocess
import os
import time
import numpy as np
from scipy.spatial import KDTree
import json

lmp_path = r"C:\Users\mitra\AppData\Local\LAMMPS 64-bit 4Jul2026\bin\lmp.exe"
data_dir = "data"
sim_dir = "simulations"

def generate_200grain_data():
    box_length = 160.0
    num_grains = 200
    random_seed = 999
    
    print(f"Generating 200-grain polycrystalline Al specimen ({box_length}x{box_length}x{box_length} Å)...")
    np.random.seed(random_seed)
    grain_seeds = np.random.uniform(0, box_length, size=(num_grains, 3))
    
    grains_info = []
    rotation_matrices = []
    for g_id in range(num_grains):
        phi1 = float(np.random.uniform(0, 360))
        theta = float(np.random.uniform(0, 180))
        phi2 = float(np.random.uniform(0, 360))
        
        p1, th, p2 = np.radians(phi1), np.radians(theta), np.radians(phi2)
        R_z1 = np.array([[np.cos(p1), -np.sin(p1), 0], [np.sin(p1), np.cos(p1), 0], [0, 0, 1]])
        R_x  = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
        R_z2 = np.array([[np.cos(p2), -np.sin(p2), 0], [np.sin(p2), np.cos(p2), 0], [0, 0, 1]])
        R = R_z2 @ R_x @ R_z1
        rotation_matrices.append(R)

    lattice_param = 4.05
    n_unit_cells = int(np.ceil(box_length / lattice_param))
    fcc_basis = np.array([[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]) * lattice_param
    
    grid_coords = []
    for i in range(-1, n_unit_cells + 1):
        for j in range(-1, n_unit_cells + 1):
            for k in range(-1, n_unit_cells + 1):
                base_pos = np.array([i, j, k]) * lattice_param
                for b in fcc_basis:
                    grid_coords.append(base_pos + b)
    grid_coords = np.array(grid_coords)

    extended_seeds, extended_ids = [], []
    for sx in [-box_length, 0.0, box_length]:
        for sy in [-box_length, 0.0, box_length]:
            for sz in [-box_length, 0.0, box_length]:
                shift = np.array([sx, sy, sz])
                for g_id, s in enumerate(grain_seeds):
                    extended_seeds.append(s + shift)
                    extended_ids.append(g_id + 1)

    tree = KDTree(np.array(extended_seeds))
    _, nearest = tree.query(grid_coords)
    assigned = np.array(extended_ids)[nearest]

    final_pos, final_gids = [], []
    for pos, g_id in zip(grid_coords, assigned):
        R = rotation_matrices[g_id - 1]
        s_pos = grain_seeds[g_id - 1]
        rot_pos = R @ (pos - s_pos) + s_pos
        if 0.0 <= rot_pos[0] < box_length and 0.0 <= rot_pos[1] < box_length and 0.0 <= rot_pos[2] < box_length:
            final_pos.append(rot_pos)
            final_gids.append(g_id)

    final_pos = np.array(final_pos)
    final_gids = np.array(final_gids)

    pos_tree = KDTree(final_pos)
    pairs = pos_tree.query_pairs(r=2.1)
    remove_idx = set(j for (i, j) in pairs)
    keep_mask = np.array([i not in remove_idx for i in range(len(final_pos))])
    clean_pos = final_pos[keep_mask]
    clean_gids = final_gids[keep_mask]

    data_path = os.path.join(data_dir, "polycrystal_200grains.data")
    with open(data_path, "w") as f:
        f.write(f"# 200-Grain Polycrystalline Al Geometry\n\n")
        f.write(f"{len(clean_pos)} atoms\n{num_grains} atom types\n\n")
        f.write(f"0.0 {box_length:.6f} xlo xhi\n0.0 {box_length:.6f} ylo yhi\n0.0 {box_length:.6f} zlo zhi\n\nMasses\n\n")
        for g_id in range(1, num_grains + 1):
            f.write(f"{g_id} 26.981539\n")
        f.write("\nAtoms # atomic\n\n")
        for i, (p, g_id) in enumerate(zip(clean_pos, clean_gids)):
            f.write(f"{i+1} {g_id} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    print(f"Exported {data_path} ({len(clean_pos)} atoms, {num_grains} grains).")
    return len(clean_pos), num_grains

def run_200grain_benchmark():
    num_atoms, num_grains = generate_200grain_data()
    pair_coeffs = " ".join(["Al"] * num_grains)

    in_content = f"""# 200-Grain Polycrystalline Al Benchmark Simulation
units           metal
dimension       3
boundary        p p p
atom_style      atomic

read_data       ../data/polycrystal_200grains.data

pair_style      eam/alloy
pair_coeff      * * ../data/Al_zhou.eam.alloy {pair_coeffs}

neighbor        2.0 bin
neigh_modify    every 1 delay 5 check yes

# Stage 1: Energy Minimization
minimize        1.0e-15 1.0e-15 300 3000

# Stage 2: NPT Thermal Equilibration
reset_timestep  0
timestep        0.001
velocity        all create 300.0 999 mom yes rot yes dist gaussian
fix             npt_eq all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0
run             300
"""
    with open(os.path.join(sim_dir, "benchmark_200grains.in"), "w") as f:
        f.write(in_content)

    print("Executing 200-grain polycrystalline Al LAMMPS benchmark...")
    start_time = time.time()
    cmd = [lmp_path, "-pk", "omp", "8", "-sf", "omp", "-in", "benchmark_200grains.in", "-log", "benchmark_200grains.log"]
    res = subprocess.run(cmd, cwd=sim_dir, capture_output=True, text=True)
    runtime = time.time() - start_time

    if res.returncode == 0:
        print(f"==========================================")
        print(f"   200-GRAIN BENCHMARK SUCCESSFULLY PASSED")
        print(f"==========================================")
        print(f"Atom Count    : {num_atoms} atoms")
        print(f"Grain Count   : {num_grains} grains")
        print(f"Runtime       : {runtime:.2f} seconds")
    else:
        print("[ERROR] 200-grain benchmark failed!")
        print(res.stderr)

if __name__ == "__main__":
    run_200grain_benchmark()
