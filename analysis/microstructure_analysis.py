import numpy as np
import json
import os
import pandas as pd
from scipy.spatial import KDTree

def calculate_fcc_taylor_factor(R):
    """
    Calculates the Taylor factor M for an FCC crystal orientation R under uniaxial loading along X.
    Uses the 12 FCC {111}<110> slip systems.
    """
    # 4 {111} plane normals
    normals = np.array([
        [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1]
    ]) / np.sqrt(3)

    # 3 <110> directions per plane -> 12 slip systems total
    directions = np.array([
        [ 1, -1,  0], [ 1,  0, -1], [ 0,  1, -1],   # plane 0: (1,1,1)
        [ 1, -1,  0], [ 1,  0,  1], [ 0,  1,  1],   # plane 1: (1,1,-1)
        [ 1,  1,  0], [ 1,  0, -1], [ 0,  1,  1],   # plane 2: (1,-1,1)
        [ 1,  1,  0], [ 1,  0,  1], [ 0,  1, -1]    # plane 3: (1,-1,-1)
    ]) / np.sqrt(2)

    # Loading direction in sample reference frame: d = [1, 0, 0]
    d_sample = np.array([1.0, 0.0, 0.0])
    
    # Rotate loading direction into crystal reference frame
    d_crystal = R.T @ d_sample

    schmid_factors = []
    for plane_idx in range(4):
        n = normals[plane_idx]
        for slip_idx in range(3):
            b = directions[plane_idx * 3 + slip_idx]
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
    """
    Reads the LAST frame of a LAMMPS custom dump and returns (box_len, df_atoms).

    Expects the dump to include image flags: `id type x y z ix iy iz`.
    Returned positions are UNWRAPPED (x + ix * Lx, etc.) so that grains which
    straddle a periodic boundary get meaningful per-grain position statistics.
    Box lengths are taken from the final frame's box bounds.

    Falls back gracefully (no unwrapping, wrapped coords kept) if the dump has
    no image-flag columns (e.g. legacy trajectories).
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    timestep_indices = [i for i, line in enumerate(lines) if "ITEM: TIMESTEP" in line]
    last_idx = timestep_indices[-1]

    # ITEM: BOX BOUNDS is the 5th line after ITEM: TIMESTEP (0 1 2 3 4)
    box_idx = last_idx + 5
    box_x = [float(x) for x in lines[box_idx].split()]
    box_y = [float(x) for x in lines[box_idx+1].split()]
    box_z = [float(x) for x in lines[box_idx+2].split()]
    box_len = [box_x[1] - box_x[0], box_y[1] - box_y[0], box_z[1] - box_z[0]]

    # Frame layout from ITEM: TIMESTEP at +0: timestep value +1, ITEM: NUMBER
    # OF ATOMS +2, count +3, ITEM: BOX BOUNDS header +4, three bounds lines
    # +5..+7, ITEM: ATOMS header +8, first atom +9.
    header_line = lines[last_idx + 8]
    assert header_line.startswith("ITEM: ATOMS"), (
        f"Unexpected dump layout: expected ATOMS header, got: {header_line!r}"
    )
    columns = header_line.split()[2:]  # strip "ITEM:", "ATOMS"
    col_idx = {name: k for k, name in enumerate(columns)}

    has_images = all(c in col_idx for c in ("ix", "iy", "iz"))

    atoms_idx = last_idx + 8
    atom_lines = lines[atoms_idx+1:]

    data = []
    for l in atom_lines:
        if "ITEM: TIMESTEP" in l:
            break
        parts = l.split()
        if len(parts) < len(columns):
            continue
        x, y, z = float(parts[col_idx["x"]]), float(parts[col_idx["y"]]), float(parts[col_idx["z"]])
        if has_images:
            ix, iy, iz = (int(parts[col_idx[c]]) for c in ("ix", "iy", "iz"))
            x += ix * box_len[0]
            y += iy * box_len[1]
            z += iz * box_len[2]
        data.append([int(parts[col_idx["id"]]), int(parts[col_idx["type"]]), x, y, z])

    df_atoms = pd.DataFrame(data, columns=["id", "type", "x", "y", "z"]).sort_values("id").reset_index(drop=True)
    return box_len, df_atoms

def _cubic_symmetry_ops():
    """Returns the 24 proper rotation matrices of the cubic (O) symmetry group."""
    return np.array([
        [[ 1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]],  # identity
        [[ 0,-1, 0],[ 1, 0, 0],[ 0, 0, 1]],  # 90° about [001]
        [[-1, 0, 0],[ 0,-1, 0],[ 0, 0, 1]],  # 180° about [001]
        [[ 0, 1, 0],[-1, 0, 0],[ 0, 0, 1]],  # 270° about [001]
        [[ 0, 0, 1],[ 0, 1, 0],[-1, 0, 0]],  # 90° about [010]
        [[-1, 0, 0],[ 0, 1, 0],[ 0, 0,-1]],  # 180° about [010]
        [[ 0, 0,-1],[ 0, 1, 0],[ 1, 0, 0]],  # 270° about [010]
        [[ 1, 0, 0],[ 0, 0,-1],[ 0, 1, 0]],  # 90° about [100]
        [[ 1, 0, 0],[ 0,-1, 0],[ 0, 0,-1]],  # 180° about [100]
        [[ 1, 0, 0],[ 0, 0, 1],[ 0,-1, 0]],  # 270° about [100]
        [[ 0, 0, 1],[ 1, 0, 0],[ 0, 1, 0]],  # 120° about [111]
        [[ 0, 1, 0],[ 0, 0, 1],[ 1, 0, 0]],  # 240° about [111]
        [[ 0, 0,-1],[-1, 0, 0],[ 0, 1, 0]],  # 120° about [-111]
        [[ 0,-1, 0],[ 0, 0, 1],[-1, 0, 0]],  # 240° about [-111]
        [[ 0, 0, 1],[-1, 0, 0],[ 0,-1, 0]],  # 120° about [1-11]
        [[ 0,-1, 0],[ 0, 0,-1],[ 1, 0, 0]],  # 240° about [1-11]
        [[ 0, 0,-1],[ 1, 0, 0],[ 0,-1, 0]],  # 120° about [11-1]
        [[ 0, 1, 0],[ 0, 0,-1],[-1, 0, 0]],  # 240° about [11-1]
        [[ 0, 1, 0],[ 1, 0, 0],[ 0, 0,-1]],  # 180° about [110]
        [[ 0,-1, 0],[-1, 0, 0],[ 0, 0,-1]],  # 180° about [1-10]
        [[ 0, 0, 1],[ 0,-1, 0],[ 1, 0, 0]],  # 180° about [101]
        [[ 0, 0,-1],[ 0,-1, 0],[-1, 0, 0]],  # 180° about [10-1]
        [[-1, 0, 0],[ 0, 0, 1],[ 0, 1, 0]],  # 180° about [011]
        [[-1, 0, 0],[ 0, 0,-1],[ 0,-1, 0]],  # 180° about [01-1]
    ], dtype=float)

_CUBIC_SYMS = _cubic_symmetry_ops()

def calculate_misorientation(R1, R2):
    """
    Computes the disorientation angle between two cubic crystal orientations.
    Applies all 24 cubic symmetry operations to find the minimum misorientation
    angle (disorientation), bounded by 62.8° for cubic crystals.
    """
    R1, R2 = np.array(R1), np.array(R2)
    R_delta = R1 @ R2.T
    min_angle = 180.0
    for S in _CUBIC_SYMS:
        R_equiv = S @ R_delta
        trace = np.trace(R_equiv)
        val = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(val)))
        if angle_deg < min_angle:
            min_angle = angle_deg
    return min_angle


def compute_voronoi_adjacency(seeds, box_len, distance_threshold_factor=1.05):
    """
    Computes grain adjacency from periodic Voronoi tessellation.

    Two grains are adjacent iff they share a Voronoi face, i.e. there exist
    points closer to both seeds than to any other seed (a bisector ridge
    segment of the two-seed KDTree, extended over 27 periodic images).

    Parameters
    ----------
    seeds : (N, 3) array of grain seed positions in [0, L)^3
    box_len : float, periodic box edge length (cubic box)
    distance_threshold_factor : float
        Two grains are also considered adjacent if their (periodic-image)
        seed separation is below this factor times the characteristic seed
        spacing d0 = (V / N)^(1/3). Provides robustness against numerically
        degenerate (sliver) Voronoi cells whose ridge detection fails.

    Returns
    -------
    list of (gid_a, gid_b) undirected adjacency pairs with 1-based grain ids
    """
    seeds = np.asarray(seeds, dtype=float)
    n_grains = len(seeds)
    d0 = (box_len ** 3 / n_grains) ** (1.0 / 3.0)  # characteristic seed spacing

    # 27 periodic images of every seed
    ext_seeds, ext_ids = [], []
    for sx in (-box_len, 0.0, box_len):
        for sy in (-box_len, 0.0, box_len):
            for sz in (-box_len, 0.0, box_len):
                shift = np.array([sx, sy, sz])
                ext_seeds.append(seeds + shift)
                ext_ids.append(np.arange(1, n_grains + 1))
    ext_seeds = np.vstack(ext_seeds)
    ext_ids = np.concatenate(ext_ids)
    tree = KDTree(ext_seeds)

    adj_pairs = set()
    # Candidate pairs: all seeds (any periodic image) within 2*d0. In a
    # Poisson-Voronoi tessellation essentially every face-sharing neighbor
    # pair lies within this radius, and the expected candidate count is
    # (4/3)*pi*2^3 ~ 34 seeds, keeping the ridge verification cheap.
    candidates = tree.query_ball_point(seeds, r=2.0 * d0)
    for g_idx, hits in enumerate(candidates):
        g_id = g_idx + 1
        seed = seeds[g_idx]
        for j in hits:
            other_gid = int(ext_ids[j])
            if other_gid == g_id:
                continue
            pair = (min(g_id, other_gid), max(g_id, other_gid))
            if pair in adj_pairs:
                continue
            # Verify shared Voronoi face: points along the seed-seed
            # bisector that are closer to both seeds than to any other seed.
            other = ext_seeds[j]
            if _shares_voronoi_face(seed, other, tree):
                adj_pairs.add(pair)

    # Robustness: merge in near-neighbor pairs (guards against sliver cells
    # whose ridge verification can numerically fail)
    near = tree.query_ball_point(seeds, r=distance_threshold_factor * d0)
    for g_idx, hits in enumerate(near):
        g_id = g_idx + 1
        for j in hits:
            other_gid = int(ext_ids[j])
            if other_gid == g_id:
                continue
            adj_pairs.add((min(g_id, other_gid), max(g_id, other_gid)))

    return sorted(adj_pairs)


def _shares_voronoi_face(seed_a, seed_b, tree, n_radii=4, n_dirs=6):
    """
    True iff a and b share a Voronoi face: exists a point ON the perpendicular
    bisector plane of a-b that is closer to both a and b than to any other
    seed (strictly closer than the third-nearest seed, so that measure-zero
    edge/vertex ties — e.g. the 8-fold-degenerate vertices of a cubic seed
    lattice — are rejected).

    On the bisector plane d_a == d_b by definition, so the condition reduces
    to: both seeds are the (tied) nearest and every other seed is strictly
    farther. Sample points are laid out on concentric in-plane rings around
    the pair midpoint to also catch face lobes of near-degenerate sliver
    cells whose face does not contain the midpoint itself.
    """
    seed_a = np.asarray(seed_a, dtype=float)
    seed_b = np.asarray(seed_b, dtype=float)
    ab = seed_b - seed_a
    half_d = 0.5 * np.linalg.norm(ab)
    n_hat = ab / np.linalg.norm(ab)

    # Orthonormal basis of the bisector plane
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, n_hat)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n_hat, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_hat, e1)

    mid = 0.5 * (seed_a + seed_b)
    for rho in np.linspace(0.0, 0.9, n_radii) * half_d:
        for k in range(n_dirs):
            phi = np.pi * k / n_dirs
            u = np.cos(phi) * e1 + np.sin(phi) * e2
            for sgn in (1.0, -1.0):
                pt = mid + sgn * rho * u
                dists, _ = tree.query(pt, k=3)
                d_a = np.linalg.norm(pt - seed_a)
                if d_a <= dists[0] + 1e-9 and d_a < dists[2] - 1e-9:
                    return True
    return False


if __name__ == "__main__":
    # Helper module — imported by generate_multisample_dataset.py
    # Run generate_multisample_dataset.py for the full pipeline.
    print("This module provides helper functions for microstructure analysis.")
    print("Functions: calculate_fcc_taylor_factor, calculate_misorientation,")
    print("           calculate_read_shockley_gb_energy, parse_lammpstrj,")
    print("           compute_voronoi_adjacency")
