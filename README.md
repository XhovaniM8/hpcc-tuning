# HPCC++ Parameter Tuning for AI Datacenter Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ECE-GY 6383: High-Speed Networks Term Project**  
**Author:** Xhovani Mali (xxm202@nyu.edu)  
**NYU Tandon School of Engineering**  
**December 2025**

---

## Overview

This repository contains the parameter tuning framework and validation tests for **HPCC++** (High Precision Congestion Control), a next-generation congestion control protocol for AI datacenter networks.

### Key Contributions

1. **Mathematical Proof of Optimal Parameters** - Rigorous derivation of optimal α, β values for 100-800 Gbps links
2. **Tuned vs Untuned Comparison** - Empirical validation showing significant performance improvements
3. **Multi-Speed Scaling** - Parameter scaling rules for 100G, 400G, and 800G links
4. **C++ & Python Implementations** - Ready-to-use code for simulation and integration

---

## Optimal Parameters (Proven)

For 100 Gbps datacenter links with ~10 µs RTT:

| Parameter | Value | Description |
|-----------|-------|-------------|
| α | 0.85 | Responsiveness to utilization error |
| β | 0.50 | Damping coefficient |
| η | 0.95 | Target utilization |
| T_s | 1 µs | Feedback sampling interval |
| W_AI | 1000 bytes | Additive increase |

### Scaling for Higher Link Speeds

| Link Speed | α | β | T_s | Notes |
|------------|-----|-----|------|-------|
| 100 Gbps | 0.85 | 0.50 | 1.0 µs | Per-packet INT |
| 400 Gbps | 0.70 | 0.42 | 0.25 µs | 4x faster feedback |
| 800 Gbps | 0.60 | 0.36 | 0.125 µs | 8x faster feedback |

---

## Repository Structure

```
hpcc-tuning-repo/
├── README.md                 # This file
├── LICENSE
├── requirements.txt
│
├── src/
│   ├── hpcc_plus_plus.h      # C++ implementation
│   ├── hpcc_refined_tuning.py    # Python simulation framework
│   └── final_proof.py        # Mathematical optimality proof
│
├── tests/
│   ├── test_tuned_vs_untuned.py  # Main comparison test
│   └── test_stability.py     # Stability constraint verification
│
├── docs/
│   ├── HPCC_TUNING_GUIDE.md  # Comprehensive tuning guide
│   ├── PROVEN_OPTIMAL_PARAMETERS.md  # Proof summary
│   └── term_project_paper.tex    # LaTeX source
│
├── figures/
│   ├── final_optimality_proof.png
│   ├── hpcc_parameter_space.png
│   ├── hpcc_cheat_sheet.png
│   └── tuned_vs_untuned_comparison.png
│
└── results/
    ├── comparison_100g.csv
    └── comparison_800g.csv
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/xhovanim/hpcc-tuning.git
cd hpcc-tuning
```

### 2. Install Dependencies

```bash
pip install numpy matplotlib pandas
```

### 3. Run the Comparison Test

```bash
python tests/test_tuned_vs_untuned.py
```

This will:
- Test HPCC++ with baseline (untuned) vs optimal (tuned) parameters
- Compare performance at 100 Gbps and 800 Gbps
- Generate comparison figures and CSV results

### 4. View Results

Results are saved to:
- `results/comparison_100g.csv` - 100 Gbps metrics
- `results/comparison_800g.csv` - 800 Gbps metrics
- `figures/tuned_vs_untuned_comparison.png` - Visualization

---

## Key Results

### Tuned vs Untuned Performance (100 Gbps)

| Metric | Untuned | Tuned | Improvement |
|--------|---------|-------|-------------|
| Utilization | 89.2% | 94.5% | +5.3% |
| Avg Queue | 45.2 KB | 9.8 KB | -78% |
| Rate Jitter | 8.5 Gbps | 2.0 Gbps | -76% |

### Tuned vs Untuned Performance (800 Gbps)

| Metric | Untuned | Tuned | Improvement |
|--------|---------|-------|-------------|
| Utilization | 87.1% | 94.2% | +7.1% |
| Avg Queue | 180 KB | 38 KB | -79% |
| Rate Jitter | 52 Gbps | 12 Gbps | -77% |

---

## Mathematical Foundation

### HPCC++ Control Law

```
W_i(t+1) = W_i(t) × [1 - α(U_j - η) - β·dU_j/dt] + W_AI
```

Where:
- `U_j(t) = qLen_j/(B_j·T) + txRate_j/B_j` — Normalized utilization
- `α` — Responsiveness parameter
- `β` — Damping coefficient  
- `η` — Target utilization
- `dU_j/dt` — Derivative term for predictive control

### Stability Constraints

1. **Loop Gain Bound:** `α × C × T_s < BDP`
2. **Critical Damping:** `β ≥ α/2`

### Damping Ratio

```
ζ = β / (α/2)
```

- ζ < 0.7: Underdamped (oscillatory)
- 0.7 ≤ ζ ≤ 1.2: **Optimal range**
- ζ > 1.2: Overdamped (slow)

---

## Integration with htsim/ns-3

### C++ Usage

```cpp
#include "hpcc_plus_plus.h"

// Configure optimal parameters
hpcc::HPCCParams optimal;
optimal.alpha = 0.85;
optimal.beta = 0.50;
optimal.eta = 0.95;
optimal.T_s = 1e-6;
optimal.W_AI = 1000;

// Create flow
hpcc::HPCCFlow flow(optimal, 100e9, 10e-6);

// On receiving telemetry
hpcc::LinkTelemetry telemetry = {...};
flow.updateRate(telemetry, true, &prev_telemetry);
double rate = flow.getRate();
```

### Python Usage

```python
from hpcc_refined_tuning import HPCCParams, SimpleHPCCSimulation

# Configure optimal parameters
params = HPCCParams(alpha=0.85, beta=0.50, eta=0.95, T_s=1e-6)

# Run simulation
sim = SimpleHPCCSimulation(params, capacity=100e9, rtt=10e-6, duration=0.01)
sim.run()
metrics = sim.metrics()

print(f"Utilization: {metrics['avg_util']:.3f}")
print(f"Queue: {metrics['avg_queue_kb']:.2f} KB")
```

---

## Related Course Materials

This work builds on concepts from ECE-GY 6383:
- **Lecture 8:** Datacenter congestion control (DCTCP, TIMELY)
- **Lecture 10:** In-band Network Telemetry (INT)
- **Lab 5:** htsim network simulator
- **Guest Lecture:** Ultra Ethernet Consortium (UEC) specification

---

## References

1. Y. Li et al., "HPCC: High Precision Congestion Control," *ACM SIGCOMM*, 2019.
2. Y. Li et al., "HPCC++: Enhanced High Precision Congestion Control," *ACM CoNEXT*, 2021.
3. F. Bonato et al., "FASTFLOW: A Dual Congestion Signal Scheme for Distributed Machine Learning," *ACM SIGCOMM*, 2024.
4. M. Handley et al., "Re-architecting Datacenter Networks and Stacks for Low Latency and High Performance," *ACM SIGCOMM*, 2017.
5. Ultra Ethernet Consortium, "Congestion Control Specification v1.0," 2024.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

**Xhovani Mali**  
MS Computer Engineering, NYU Tandon (Dec 2025)  
Email: xxm202@nyu.edu  
GitHub: [@xhovanim](https://github.com/xhovanim)
