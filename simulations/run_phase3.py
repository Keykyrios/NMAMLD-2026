import os
import re
import matplotlib.pyplot as plt
import pandas as pd

log_path = os.path.join("simulations", "phase3.log")
with open(log_path, "r") as f:
    lines = f.readlines()

# Locate Stage 2 NPT equilibration thermo table
thermo_data = []
header = None
in_stage2 = False

for line in lines:
    if "=== STAGE 2: NPT Thermal Equilibration" in line:
        in_stage2 = True
        continue
    if in_stage2 and "Step" in line and "Temp" in line:
        header = line.split()
        continue
    if in_stage2 and header:
        parts = line.split()
        if len(parts) == len(header) and parts[0].isdigit():
            thermo_data.append([float(x) for x in parts])
        elif "Loop time" in line:
            break

df = pd.DataFrame(thermo_data, columns=header)
print(f"Extracted {len(df)} NPT equilibration thermo data points:")
print(df)

# Save summary metrics
num_atoms = 17682.0
final_temp = df["Temp"].iloc[-1]
final_pe_atom = df["PotEng"].iloc[-1] / num_atoms
final_vol = df["Volume"].iloc[-1]

print("\n==========================================")
print("      PHASE 3 EQUILIBRATION SUMMARY       ")
print("==========================================")
print(f"Final Temperature : {final_temp:.2f} K")
print(f"Final PE / Atom   : {final_pe_atom:.4f} eV/atom")
print(f"Final Volume      : {final_vol:.2f} Å³")

# Generate publication-quality plots
os.makedirs("plots", exist_ok=True)
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# 1. Temperature
axes[0].plot(df["Step"] * 0.001, df["Temp"], color="#e74c3c", lw=2, label="Temperature (K)")
axes[0].axhline(300, color="black", linestyle="--", alpha=0.7, label="Target (300 K)")
axes[0].set_ylabel("Temperature [K]", fontsize=12)
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)
axes[0].set_title("Phase 3: Polycrystalline Al NPT Thermal Equilibration (300 K)", fontsize=14, fontweight="bold")

# 2. Potential Energy per atom
axes[1].plot(df["Step"] * 0.001, df["PotEng"] / num_atoms, color="#2980b9", lw=2, label="PE / atom (eV)")
axes[1].set_ylabel("PE / Atom [eV]", fontsize=12)
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

# 3. Box Volume
axes[2].plot(df["Step"] * 0.001, df["Volume"], color="#27ae60", lw=2, label="Volume (Å³)")
axes[2].set_ylabel("Volume [Å³]", fontsize=12)
axes[2].set_xlabel("Equilibration Time [ps]", fontsize=12)
axes[2].legend(loc="upper right")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join("plots", "phase3_equilibration.png")
plt.savefig(fig_path, dpi=300)
print(f"Equilibration plot saved to {fig_path}.")
