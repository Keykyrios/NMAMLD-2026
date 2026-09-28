import subprocess
import os
import time
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

lmp_path = r"C:\Users\mitra\AppData\Local\LAMMPS 64-bit 4Jul2026\bin\lmp.exe"
in_script = "phase4_tensile.in"
log_file = "phase4.log"
cwd = "simulations"

print(f"--- Running Phase 4: Uniaxial Tensile Loading (Polycrystalline Al) ---")
start_time = time.time()

# Run with OpenMP threads enabled (-pk omp 8 -sf omp)
cmd = [lmp_path, "-pk", "omp", "8", "-sf", "omp", "-in", in_script, "-log", log_file]

result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
wall_clock = time.time() - start_time

print(f"Phase 4 execution finished in {wall_clock:.2f} seconds.")
if result.returncode != 0:
    print("[ERROR] Phase 4 execution failed!")
    print(result.stderr)
    exit(1)

# Parse thermo data from phase4.log
log_path = os.path.join(cwd, log_file)
with open(log_path, "r") as f:
    lines = f.readlines()

thermo_data = []
header = None
in_deform = False

for line in lines:
    if "=== STAGE 4: Uniaxial Tensile Deformation" in line:
        in_deform = True
        continue
    if in_deform and "Step" in line and "v_strain" in line:
        header = line.split()
        continue
    if in_deform and header:
        parts = line.split()
        if len(parts) == len(header) and parts[0].isdigit():
            thermo_data.append([float(x) for x in parts])
        elif "Loop time" in line:
            break

df = pd.DataFrame(thermo_data, columns=header)
print(f"Extracted {len(df)} tensile loading data points.")
print(df.head(5))

# Export CSV for reproducibility
csv_path = os.path.join("data", "stress_strain_data.csv")
df.to_csv(csv_path, index=False)
print(f"Saved stress-strain data to {csv_path}.")

# Plot Publication-Quality Stress-Strain Response
os.makedirs("plots", exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 1. Stress-Strain Curve
ax1.plot(df["v_strain"] * 100.0, df["v_stress_x"], color="#c0392b", lw=2.5, label=r"Axial Tensile Stress $\sigma_{xx}$")
ax1.plot(df["v_strain"] * 100.0, df["v_stress_y"], color="#2980b9", lw=1.5, linestyle="--", label=r"Transverse Stress $\sigma_{yy}$")
ax1.plot(df["v_strain"] * 100.0, df["v_stress_z"], color="#27ae60", lw=1.5, linestyle="--", label=r"Transverse Stress $\sigma_{zz}$")
ax1.set_xlabel("Engineering Strain [%]", fontsize=12, fontweight="bold")
ax1.set_ylabel("Stress [GPa]", fontsize=12, fontweight="bold")
ax1.set_title("MD Uniaxial Stress–Strain Curve (Polycrystalline Al)", fontsize=13, fontweight="bold")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# 2. Potential Energy vs Strain
ax2.plot(df["v_strain"] * 100.0, df["PotEng"] / 17682.0, color="#8e44ad", lw=2.5, label="PE / Atom")
ax2.set_xlabel("Engineering Strain [%]", fontsize=12, fontweight="bold")
ax2.set_ylabel("Potential Energy [eV/atom]", fontsize=12, fontweight="bold")
ax2.set_title("Potential Energy Evolution during Tension", fontsize=13, fontweight="bold")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join("plots", "phase4_stress_strain.png")
plt.savefig(fig_path, dpi=300)
print(f"Stress-strain plot saved to {fig_path}.")

# Print Peak Stress & Yield Point
max_stress_idx = df["v_stress_x"].idxmax()
peak_strain = df["v_strain"].iloc[max_stress_idx] * 100.0
peak_stress = df["v_stress_x"].iloc[max_stress_idx]

print("\n==========================================")
print("       PHASE 4 TENSILE SUMMARY            ")
print("==========================================")
print(f"Peak Tensile Stress (UTS) : {peak_stress:.3f} GPa")
print(f"Strain at Peak Stress     : {peak_strain:.2f} %")
