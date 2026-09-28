import subprocess
import os
import time
import re

lmp_path = r"C:\Users\mitra\AppData\Local\LAMMPS 64-bit 4Jul2026\bin\lmp.exe"
in_script = "phase1_single_crystal.in"
log_file = "phase1.log"
cwd = "simulations"

print(f"Executing Phase 1 benchmark using LAMMPS binary at:\n  {lmp_path}")
print(f"Running script: {in_script}")

start_time = time.time()
cmd = [lmp_path, "-in", in_script, "-log", log_file]

result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
wall_clock = time.time() - start_time

print(f"\n--- Simulation Output ---")
print(result.stdout[-1500:])

if result.returncode != 0:
    print(f"\n[ERROR] LAMMPS run failed with return code {result.returncode}!")
    print(result.stderr)
    exit(1)

# Parse log file for quantitative results
log_path = os.path.join(cwd, log_file)
with open(log_path, 'r') as f:
    log_content = f.read()

# Extract performance / timesteps per second
perf_match = re.search(r"Performance:\s+([\d\.]+)\s+ns/day,\s+([\d\.]+)\s+hours/ns,\s+([\d\.]+)\s+timesteps/s", log_content)
if perf_match:
    ns_per_day, hours_per_ns, timesteps_per_sec = perf_match.groups()
    print("\n==========================================")
    print("       PHASE 1 BENCHMARK RESULTS          ")
    print("==========================================")
    print(f"Wall-clock Runtime : {wall_clock:.2f} seconds")
    print(f"Performance        : {timesteps_per_sec} timesteps/sec ({ns_per_day} ns/day)")
else:
    print(f"\nSimulation completed in {wall_clock:.2f}s. Check log file for details.")
