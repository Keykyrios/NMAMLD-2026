# Multi-Sample Dataset Methodology & Provenance

**Date:** September 23, 2026  
**Dataset Size:** N = 10 Independent Atomistic Polycrystalline Specimens  
**Total Microstructure Graphs:** 10  
**Total Nodes (Grains):** 132  
**Total Edges (Boundaries):** 763  

---

## 1. Simulation & Generation Protocol

- **Material:** Pure Aluminum (Al), $A = 26.981539 \text{ g/mol}$
- **Potential:** Zhou et al. EAM Potential (`Al_zhou.eam.alloy`)
- **Box Dimensions:** $80.0 \times 80.0 \times 80.0 \text{ Å}^3$ ($V_0 = 512,000 \text{ Å}^3$)
- **Grain Count Range:** 10 to 16 Voronoi grains per sample
- **Grain Orientations:** Randomly sampled Bunge Euler angles $(\phi_1 \in [0, 360^\circ], \theta \in [0, 180^\circ], \phi_2 \in [0, 360^\circ])$
- **Loading Protocol:** Uniaxial tensile strain along X axis at $\dot{\varepsilon} = 0.01 \text{ ps}^{-1}$ ($10^{10} \text{ s}^{-1}$) under NPT transverse relaxation ($P_y = 0, P_z = 0$) up to 10% engineering strain.

---

## 2. Sample Manifest Summary

```csv
sample_id,seed,num_grains,atom_count,wall_clock_s,potential,strain_rate,max_strain
1,101,12,18618,829.91,Al_zhou.eam.alloy,0.01 ps^-1,0.10
2,102,14,14851,695.39,Al_zhou.eam.alloy,0.01 ps^-1,0.10
3,103,10,17965,912.9,Al_zhou.eam.alloy,0.01 ps^-1,0.10
4,104,15,16194,985.92,Al_zhou.eam.alloy,0.01 ps^-1,0.10
5,105,11,15719,621.25,Al_zhou.eam.alloy,0.01 ps^-1,0.10
6,106,13,16206,247.34,Al_zhou.eam.alloy,0.01 ps^-1,0.10
7,107,16,16853,249.02,Al_zhou.eam.alloy,0.01 ps^-1,0.10
8,108,12,16845,252.11,Al_zhou.eam.alloy,0.01 ps^-1,0.10
9,109,14,16536,602.69,Al_zhou.eam.alloy,0.01 ps^-1,0.10
10,110,15,17364,286.41,Al_zhou.eam.alloy,0.01 ps^-1,0.10

```
