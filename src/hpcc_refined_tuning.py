#!/usr/bin/env python3
"""
HPCC++ Parameter Tuning - Refined for High-Speed Networks
Proper time scaling for 100 Gbps+ datacenter links

Author: Xhovani Mali
ECE-GY 6383: High-Speed Networks
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass

@dataclass
class HPCCParams:
    """HPCC++ control parameters with proper scaling"""
    alpha: float = 0.1
    beta: float = 0.02
    eta: float = 0.95
    W_AI: float = 1000
    T_s: float = 1e-5  # 10 microseconds - appropriate for 100G links
    
    def check_stability(self, capacity: float) -> tuple:
        """Check stability constraints"""
        # For high-speed links, use normalized alpha
        # Constraint: alpha * (C * T_s) / W < 1
        # Where W is typical window size ≈ BDP
        
        # Simpler constraint: alpha * C * T_s should be << typical window
        constraint_val = self.alpha * capacity * self.T_s
        bdp_typical = capacity * 10e-6  # BDP for 10us RTT
        
        # Normalized constraint
        normalized_alpha = constraint_val / bdp_typical
        
        if normalized_alpha >= 1.0:
            return False, f"α too large: normalized = {normalized_alpha:.3f}"
        
        if self.beta < self.alpha / 2:
            return False, f"β = {self.beta} < α/2 = {self.alpha/2}"
        
        return True, "Stable"

class SimpleHPCCSimulation:
    """Simplified HPCC++ simulation for parameter tuning"""
    
    def __init__(self, params: HPCCParams, capacity: float, rtt: float, duration: float):
        self.params = params
        self.capacity = capacity
        self.rtt = rtt
        self.duration = duration
        
        # Initialize
        self.window = capacity * rtt  # BDP
        self.rate = capacity
        
        # State
        self.queue = 0
        self.time = 0
        self.prev_util = 0
        
        # History
        self.rate_hist = []
        self.queue_hist = []
        self.util_hist = []
        self.window_hist = []
        self.time_hist = []
        
    def step(self, dt: float):
        """Single simulation step"""
        # Incoming bytes this step
        incoming = self.rate * dt / 8  # Convert bps to bytes
        
        # Service bytes
        service = self.capacity * dt / 8
        
        # Update queue
        self.queue += incoming
        served = min(self.queue, service)
        self.queue -= served
        
        # Compute utilization U_j = qLen/(B*T) + txRate/B
        q_ratio = self.queue / (self.capacity * self.rtt / 8)  # Normalize by BDP
        rate_ratio = self.rate / self.capacity
        U_j = q_ratio + rate_ratio
        
        # Derivative term
        dU_dt = (U_j - self.prev_util) / dt if dt > 0 else 0
        
        # HPCC++ update
        error = U_j - self.params.eta
        damping = self.params.beta * dU_dt
        mult = 1 - self.params.alpha * error - damping
        mult = np.clip(mult, 0.5, 1.5)
        
        # Update window
        new_window = self.window * mult + self.params.W_AI
        self.window = np.clip(new_window, 0.1 * self.capacity * self.rtt, 
                             2.0 * self.capacity * self.rtt)
        self.rate = self.window / self.rtt
        
        # Record
        self.rate_hist.append(self.rate)
        self.queue_hist.append(self.queue)
        self.util_hist.append(U_j)
        self.window_hist.append(self.window)
        self.time_hist.append(self.time)
        
        self.prev_util = U_j
        self.time += dt
        
    def run(self):
        """Run full simulation"""
        dt = self.params.T_s
        while self.time < self.duration:
            self.step(dt)
            
    def metrics(self):
        """Compute metrics"""
        return {
            'avg_rate_gbps': np.mean(self.rate_hist) / 1e9,
            'rate_std_gbps': np.std(self.rate_hist) / 1e9,
            'avg_util': np.mean(self.util_hist),
            'util_std': np.std(self.util_hist),
            'avg_queue_kb': np.mean(self.queue_hist) / 1024,
            'max_queue_kb': np.max(self.queue_hist) / 1024,
            'util_error': abs(np.mean(self.util_hist) - self.params.eta)
        }

def objective(metrics, w_queue=1.0, w_util=2.0, w_stability=0.5):
    """Cost function - lower is better"""
    queue_cost = w_queue * metrics['avg_queue_kb'] / 100  # Normalize to ~100KB
    util_cost = w_util * metrics['util_error']
    stability_cost = w_stability * metrics['rate_std_gbps']
    return queue_cost + util_cost + stability_cost

def tune_parameters():
    """Main parameter tuning workflow"""
    print("="*70)
    print("HPCC++ Parameter Tuning for AI Datacenter Networks")
    print("Refined for High-Speed Links (100 Gbps)")
    print("="*70)
    
    # Configuration
    capacity = 100e9  # 100 Gbps
    rtt = 10e-6       # 10 microseconds
    duration = 0.01   # 10 ms simulation
    
    print(f"\nConfiguration:")
    print(f"  Link capacity: {capacity/1e9:.0f} Gbps")
    print(f"  Base RTT: {rtt*1e6:.0f} µs")
    print(f"  BDP: {capacity * rtt / 8 / 1024:.1f} KB")
    
    # Test baseline
    print("\n" + "="*70)
    print("Baseline Test (α=0.1, β=0.02, η=0.95)")
    print("="*70)
    
    baseline = HPCCParams(alpha=0.1, beta=0.02, eta=0.95, T_s=1e-5)
    stable, msg = baseline.check_stability(capacity)
    print(f"Stability: {msg}")
    
    if stable:
        sim = SimpleHPCCSimulation(baseline, capacity, rtt, duration)
        sim.run()
        m = sim.metrics()
        
        print("\nPerformance:")
        print(f"  Avg rate: {m['avg_rate_gbps']:.2f} Gbps")
        print(f"  Avg utilization: {m['avg_util']:.3f}")
        print(f"  Avg queue: {m['avg_queue_kb']:.2f} KB")
        print(f"  Max queue: {m['max_queue_kb']:.2f} KB")
        print(f"  Rate stability (σ): {m['rate_std_gbps']:.3f} Gbps")
        print(f"  Cost: {objective(m):.4f}")
    
    # Parameter sweep
    print("\n" + "="*70)
    print("Parameter Sweep")
    print("="*70)
    
    alpha_vals = np.linspace(0.01, 0.5, 15)
    beta_vals = np.linspace(0.005, 0.25, 15)
    
    results = []
    best_cost = float('inf')
    best_params = None
    
    print(f"Testing {len(alpha_vals) * len(beta_vals)} combinations...")
    
    for alpha in alpha_vals:
        for beta in beta_vals:
            params = HPCCParams(alpha=alpha, beta=beta, eta=0.95, T_s=1e-5)
            stable, _ = params.check_stability(capacity)
            
            if not stable:
                continue
                
            try:
                sim = SimpleHPCCSimulation(params, capacity, rtt, 0.005)
                sim.run()
                m = sim.metrics()
                cost = objective(m)
                
                results.append({
                    'alpha': alpha,
                    'beta': beta,
                    'cost': cost,
                    **m
                })
                
                if cost < best_cost:
                    best_cost = cost
                    best_params = (alpha, beta)
                    
            except Exception as e:
                continue
    
    df = pd.DataFrame(results)
    
    print(f"\nTested {len(results)} stable parameter combinations")
    print(f"\nBest parameters:")
    print(f"  α = {best_params[0]:.3f}")
    print(f"  β = {best_params[1]:.3f}")
    print(f"  Cost = {best_cost:.4f}")
    
    # Test best parameters
    print("\n" + "="*70)
    print("Validating Best Parameters")
    print("="*70)
    
    best = HPCCParams(alpha=best_params[0], beta=best_params[1], eta=0.95, T_s=1e-5)
    sim_best = SimpleHPCCSimulation(best, capacity, rtt, 0.02)
    sim_best.run()
    m_best = sim_best.metrics()
    
    print("\nOptimized Performance:")
    print(f"  Avg rate: {m_best['avg_rate_gbps']:.2f} Gbps")
    print(f"  Avg utilization: {m_best['avg_util']:.3f}")
    print(f"  Avg queue: {m_best['avg_queue_kb']:.2f} KB")
    print(f"  Max queue: {m_best['max_queue_kb']:.2f} KB")
    print(f"  Rate stability: {m_best['rate_std_gbps']:.3f} Gbps")
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Baseline rate evolution
    ax = axes[0, 0]
    time_ms = np.array(sim.time_hist) * 1000
    rate_gbps = np.array(sim.rate_hist) / 1e9
    ax.plot(time_ms, rate_gbps, 'b-', linewidth=1.5)
    ax.axhline(capacity/1e9, color='r', linestyle='--', alpha=0.7, label='Capacity')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Rate (Gbps)')
    ax.set_title('Baseline: Rate Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Baseline utilization
    ax = axes[0, 1]
    ax.plot(time_ms, sim.util_hist, 'g-', linewidth=1.5)
    ax.axhline(baseline.eta, color='r', linestyle='--', alpha=0.7, label=f'Target η={baseline.eta}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Utilization U_j')
    ax.set_title('Baseline: Utilization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Baseline queue
    ax = axes[0, 2]
    queue_kb = np.array(sim.queue_hist) / 1024
    ax.plot(time_ms, queue_kb, 'purple', linewidth=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Queue (KB)')
    ax.set_title('Baseline: Queue Length')
    ax.grid(True, alpha=0.3)
    
    # 4. Optimized rate
    ax = axes[1, 0]
    time_ms_best = np.array(sim_best.time_hist) * 1000
    rate_gbps_best = np.array(sim_best.rate_hist) / 1e9
    ax.plot(time_ms_best, rate_gbps_best, 'b-', linewidth=1.5)
    ax.axhline(capacity/1e9, color='r', linestyle='--', alpha=0.7, label='Capacity')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Rate (Gbps)')
    ax.set_title(f'Optimized (α={best_params[0]:.3f}, β={best_params[1]:.3f}): Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Parameter space heatmap
    ax = axes[1, 1]
    pivot = df.pivot_table(values='cost', index='beta', columns='alpha')
    im = ax.contourf(pivot.columns, pivot.index, pivot.values, levels=20, cmap='viridis')
    ax.plot(best_params[0], best_params[1], 'r*', markersize=15, label='Optimal')
    ax.set_xlabel('α (responsiveness)')
    ax.set_ylabel('β (damping)')
    ax.set_title('Parameter Space: Cost Function')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Cost')
    
    # 6. Optimized queue
    ax = axes[1, 2]
    queue_kb_best = np.array(sim_best.queue_hist) / 1024
    ax.plot(time_ms_best, queue_kb_best, 'purple', linewidth=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Queue (KB)')
    ax.set_title('Optimized: Queue Length')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('HPCC++ Parameter Tuning Results', fontsize=16)
    plt.tight_layout()
    
    # Save
    df.to_csv('/mnt/user-data/outputs/hpcc_tuning_results.csv', index=False)
    plt.savefig('/mnt/user-data/outputs/hpcc_tuning_analysis.png', dpi=300, bbox_inches='tight')
    
    print("\n" + "="*70)
    print("Results saved:")
    print("  - hpcc_tuning_results.csv")
    print("  - hpcc_tuning_analysis.png")
    print("="*70)
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR YOUR PROPOSAL")
    print("="*70)
    print(f"""
For 100 Gbps AI datacenter links with ~10µs RTT:

Optimal HPCC++ Parameters:
  • α (responsiveness) = {best_params[0]:.3f}
  • β (damping) = {best_params[1]:.3f}
  • η (target utilization) = 0.95
  • T_s (feedback interval) = 10 µs
  • W_AI (additive increase) = 1000 bytes

Key Insights:
  1. Feedback interval T_s must scale with link speed
     - For 100G: use 10 µs intervals
     - For 400G: use 2.5 µs intervals
  
  2. Stability constraint: α * C * T_s < W (window size)
     - This ensures loop gain < 1
  
  3. Damping ratio: β ≥ α/2 prevents oscillation
     - Higher β = more stable, slower response
     - Lower β = faster response, possible overshoot
  
  4. For AI workloads (incast-heavy):
     - Use slightly higher β for stability
     - Keep η at 0.95 for high throughput
     - Monitor queue buildup carefully

Comparison with NDP:
  - HPCC++: Explicit feedback, faster convergence
  - NDP: Receiver-driven, natural load balancing
  - Both achieve near-zero queuing
  - HPCC++ has lower latency variance under incast
""")

if __name__ == "__main__":
    tune_parameters()
    plt.show()
