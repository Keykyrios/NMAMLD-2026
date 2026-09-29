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


if __name__ == "__main__":
    # Helper module — imported by generate_multisample_dataset.py
    # Run generate_multisample_dataset.py for the full pipeline.
    print("This module provides helper functions for microstructure analysis.")
    print("Functions: calculate_fcc_taylor_factor, calculate_misorientation,")
    print("           calculate_read_shockley_gb_energy, parse_lammpstrj")
