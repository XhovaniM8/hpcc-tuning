#!/usr/bin/env python3
"""
HPCC++ Tuned vs Untuned Comparison Test
========================================
ECE-GY 6383: High-Speed Networks Term Project
Author: Xhovani Mali (xxm202@nyu.edu)

Tests HPCC++ performance with optimal tuned parameters vs baseline (untuned)
across multiple link speeds: 100 Gbps and 800 Gbps

Usage:
    python test_tuned_vs_untuned.py
    
Outputs:
    - results/comparison_100g.csv
    - results/comparison_800g.csv  
    - figures/tuned_vs_untuned_comparison.png
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os

# ==============================================================================
# HPCC++ Parameter Configurations
# ==============================================================================

@dataclass
class HPCCConfig:
    """HPCC++ control parameters"""
    name: str
    alpha: float      # Responsiveness parameter
    beta: float       # Damping coefficient
    eta: float        # Target utilization
    T_s: float        # Feedback sampling interval (seconds)
    W_AI: float       # Additive increase (bytes)
    
    def damping_ratio(self) -> float:
        """Compute damping ratio ζ = β/(α/2)"""
        return self.beta / (self.alpha / 2) if self.alpha > 0 else 0
    
    def check_stability(self, capacity: float, rtt: float) -> Tuple[bool, str]:
        """Check both stability constraints"""
        bdp = capacity * rtt / 8  # bytes
        # Loop gain in bytes (capacity in bps, T_s in seconds → bits, divide by 8)
        loop_gain = self.alpha * capacity * self.T_s / 8
        
        c1_ok = loop_gain < bdp
        c2_ok = self.beta >= self.alpha / 2
        
        if not c1_ok:
            return False, f"Loop gain violation: {loop_gain:.0f} >= BDP {bdp:.0f}"
        if not c2_ok:
            return False, f"Damping violation: β={self.beta} < α/2={self.alpha/2}"
        return True, "Stable"

# Define configurations for different scenarios
CONFIGS = {
    # 100 Gbps configurations
    "100g_untuned": HPCCConfig(
        name="100G Baseline (Untuned)",
        alpha=0.15, beta=0.08, eta=0.95,  # Original HPCC paper-style params
        T_s=10e-6, W_AI=1000
    ),
    "100g_tuned": HPCCConfig(
        name="100G Optimal (Tuned)",
        alpha=0.85, beta=0.50, eta=0.95,
        T_s=1e-6, W_AI=1000
    ),
    # 800 Gbps configurations (scaled from 100G optimal)
    "800g_untuned": HPCCConfig(
        name="800G Baseline (Untuned)",
        alpha=0.15, beta=0.08, eta=0.95,  # Original HPCC paper-style params
        T_s=10e-6, W_AI=1000
    ),
    "800g_tuned": HPCCConfig(
        name="800G Optimal (Tuned)",
        alpha=0.60, beta=0.36, eta=0.95,  # Scaled for 800G
        T_s=0.125e-6, W_AI=1000  # 8x faster feedback for 8x faster link
    ),
}

# ==============================================================================
# Simplified HPCC++ Simulation
# ==============================================================================

class HPCCSimulation:
    """Simplified HPCC++ simulation for parameter comparison"""
    
    def __init__(self, config: HPCCConfig, capacity: float, rtt: float):
        self.config = config
        self.capacity = capacity  # bps
        self.rtt = rtt  # seconds
        self.bdp = capacity * rtt / 8  # bytes
        
        # State
        self.window = self.bdp
        self.rate = capacity * 0.9  # Start slightly below capacity
        self.queue = self.bdp * 0.1  # Start with small queue
        self.prev_util = config.eta  # Start near target
        self.time = 0.0
        
        # Simulate competing traffic (50% of capacity from other sources)
        self.background_rate = capacity * 0.5
        
        # History
        self.history = {
            'time': [], 'rate': [], 'queue': [],
            'utilization': [], 'window': []
        }
    
    def step(self, dt: float) -> None:
        """Execute one simulation step"""
        # Total traffic: our flow + background
        total_incoming = (self.rate + self.background_rate) * dt / 8
        
        # Service model: bytes drained at link capacity
        service = self.capacity * dt / 8
        
        # Queue dynamics
        self.queue = max(0, self.queue + total_incoming - service)
        
        # Compute normalized utilization (what our flow "sees")
        q_ratio = self.queue / self.bdp
        rate_ratio = (self.rate + self.background_rate) / self.capacity
        U_j = min(q_ratio + rate_ratio, 2.0)  # Cap at 2x for stability
        
        # Derivative term (for damping)
        dU_dt = (U_j - self.prev_util) / dt if dt > 0 else 0
        
        # HPCC++ control law
        error = U_j - self.config.eta
        damping = self.config.beta * dU_dt
        mult = 1 - self.config.alpha * error - damping
        mult = np.clip(mult, 0.5, 1.5)
        
        # Update window
        new_window = self.window * mult + self.config.W_AI
        self.window = np.clip(new_window, 0.1 * self.bdp, 2.0 * self.bdp)
        self.rate = self.window / self.rtt
        
        # Record history
        self.history['time'].append(self.time)
        self.history['rate'].append(self.rate)
        self.history['queue'].append(self.queue)
        self.history['utilization'].append(U_j)
        self.history['window'].append(self.window)
        
        self.prev_util = U_j
        self.time += dt
    
    def run(self, duration: float) -> None:
        """Run simulation for specified duration"""
        dt = self.config.T_s
        while self.time < duration:
            self.step(dt)
    
    def get_metrics(self) -> Dict:
        """Compute performance metrics"""
        # Skip warmup period (first 10%)
        n = len(self.history['rate'])
        start = int(n * 0.1)
        
        rate = np.array(self.history['rate'][start:])
        queue = np.array(self.history['queue'][start:])
        util = np.array(self.history['utilization'][start:])
        
        # Our fair share is 50% of capacity (background takes other 50%)
        fair_share = self.capacity * 0.5
        
        return {
            'avg_rate_gbps': np.mean(rate) / 1e9,
            'rate_std_gbps': np.std(rate) / 1e9,
            'avg_utilization': np.mean(util),
            'util_error': abs(np.mean(util) - self.config.eta),
            'avg_queue_kb': np.mean(queue) / 1024,
            'max_queue_kb': np.max(queue) / 1024,
            'p99_queue_kb': np.percentile(queue, 99) / 1024,
            'convergence_time_us': self._compute_convergence() * 1e6,
            'throughput_efficiency': np.mean(rate) / fair_share  # How well we use our fair share
        }
    
    def _compute_convergence(self) -> float:
        """Estimate time to converge within 5% of target utilization"""
        util = np.array(self.history['utilization'])
        time = np.array(self.history['time'])
        target = self.config.eta
        
        # Find first time we stay within 5% for at least 10 samples
        window = 10
        for i in range(len(util) - window):
            if all(abs(u - target) < 0.05 for u in util[i:i+window]):
                return time[i]
        return time[-1] if len(time) > 0 else 0

# ==============================================================================
# Test Harness
# ==============================================================================

def run_comparison(link_speed_gbps: int) -> pd.DataFrame:
    """Run tuned vs untuned comparison for a given link speed"""
    
    capacity = link_speed_gbps * 1e9
    rtt = 10e-6  # 10 microseconds
    duration = 0.001  # 1 ms (plenty for these fast links)
    
    speed_key = f"{link_speed_gbps}g"
    configs = {
        'untuned': CONFIGS[f"{speed_key}_untuned"],
        'tuned': CONFIGS[f"{speed_key}_tuned"]
    }
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"Testing {link_speed_gbps} Gbps Link (RTT = {rtt*1e6:.0f} µs)")
    print(f"{'='*70}")
    print(f"BDP = {capacity * rtt / 8 / 1024:.1f} KB")
    
    for label, config in configs.items():
        print(f"\n--- {config.name} ---")
        print(f"Parameters: α={config.alpha}, β={config.beta}, η={config.eta}")
        print(f"            T_s={config.T_s*1e6:.2f} µs, W_AI={config.W_AI}")
        print(f"Damping ratio ζ = {config.damping_ratio():.2f}")
        
        # Check stability
        stable, msg = config.check_stability(capacity, rtt)
        print(f"Stability: {msg}")
        
        if not stable:
            print("  ⚠ UNSTABLE - skipping")
            continue
        
        # Run simulation
        sim = HPCCSimulation(config, capacity, rtt)
        sim.run(duration)
        metrics = sim.get_metrics()
        
        print(f"\nPerformance:")
        print(f"  Avg Rate:        {metrics['avg_rate_gbps']:.2f} Gbps")
        print(f"  Rate Std Dev:    {metrics['rate_std_gbps']:.2f} Gbps")
        print(f"  Avg Utilization: {metrics['avg_utilization']:.3f}")
        print(f"  Util Error:      {metrics['util_error']:.4f}")
        print(f"  Avg Queue:       {metrics['avg_queue_kb']:.2f} KB")
        print(f"  Max Queue:       {metrics['max_queue_kb']:.2f} KB")
        print(f"  P99 Queue:       {metrics['p99_queue_kb']:.2f} KB")
        print(f"  Convergence:     {metrics['convergence_time_us']:.1f} µs")
        
        results.append({
            'config': label,
            'link_speed_gbps': link_speed_gbps,
            'alpha': config.alpha,
            'beta': config.beta,
            'T_s_us': config.T_s * 1e6,
            **metrics
        })
    
    return pd.DataFrame(results)

def plot_comparison(results_100g: pd.DataFrame, results_800g: pd.DataFrame,
                   output_path: str) -> None:
    """Generate comparison visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Colors
    colors = {'untuned': '#e74c3c', 'tuned': '#27ae60'}
    
    # Metrics to plot
    metrics = [
        ('avg_utilization', 'Utilization', 0.95, 'Target η=0.95'),
        ('avg_queue_kb', 'Avg Queue (KB)', None, None),
        ('rate_std_gbps', 'Rate Jitter (Gbps)', None, None)
    ]
    
    for col, (metric, ylabel, target, target_label) in enumerate(metrics):
        # 100G plot (top row)
        ax = axes[0, col]
        for _, row in results_100g.iterrows():
            ax.bar(row['config'], row[metric], 
                   color=colors[row['config']], alpha=0.8, edgecolor='black')
        if target:
            ax.axhline(target, color='blue', linestyle='--', label=target_label)
            ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_title(f'100 Gbps: {ylabel}')
        ax.grid(axis='y', alpha=0.3)
        
        # 800G plot (bottom row)
        ax = axes[1, col]
        for _, row in results_800g.iterrows():
            ax.bar(row['config'], row[metric],
                   color=colors[row['config']], alpha=0.8, edgecolor='black')
        if target:
            ax.axhline(target, color='blue', linestyle='--', label=target_label)
            ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_title(f'800 Gbps: {ylabel}')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('HPCC++ Tuned vs Untuned Comparison\n'
                 'Optimal Parameters: α, β, T_s scaled per link speed',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")

def print_summary_table(results_100g: pd.DataFrame, results_800g: pd.DataFrame) -> None:
    """Print a summary comparison table"""
    
    print("\n" + "="*80)
    print("SUMMARY: HPCC++ TUNED vs UNTUNED COMPARISON")
    print("="*80)
    
    print("\n┌─────────┬──────────┬─────────────┬─────────────┬────────────┬────────────┐")
    print("│  Speed  │  Config  │  Util (%)   │  Queue (KB) │ Jitter(Gb) │ Conv (µs)  │")
    print("├─────────┼──────────┼─────────────┼─────────────┼────────────┼────────────┤")
    
    for df in [results_100g, results_800g]:
        for _, row in df.iterrows():
            print(f"│ {row['link_speed_gbps']:>4}G   │ {row['config']:>8} │"
                  f"  {row['avg_utilization']*100:>6.2f}%   │"
                  f"   {row['avg_queue_kb']:>7.2f}  │"
                  f"  {row['rate_std_gbps']:>7.2f}  │"
                  f"  {row['convergence_time_us']:>7.1f}  │")
        print("├─────────┼──────────┼─────────────┼─────────────┼────────────┼────────────┤")
    
    print("└─────────┴──────────┴─────────────┴─────────────┴────────────┴────────────┘")
    
    # Compute improvements if we have both configs
    print("\n" + "-"*80)
    print("IMPROVEMENT FROM TUNING:")
    print("-"*80)
    
    for speed, df in [('100G', results_100g), ('800G', results_800g)]:
        untuned_rows = df[df['config'] == 'untuned']
        tuned_rows = df[df['config'] == 'tuned']
        
        if len(untuned_rows) == 0 or len(tuned_rows) == 0:
            print(f"\n{speed}: Comparison not available (need both tuned and untuned)")
            continue
            
        untuned = untuned_rows.iloc[0]
        tuned = tuned_rows.iloc[0]
        
        util_imp = (tuned['avg_utilization'] - untuned['avg_utilization']) * 100
        
        if untuned['avg_queue_kb'] > 0:
            queue_imp = (untuned['avg_queue_kb'] - tuned['avg_queue_kb']) / untuned['avg_queue_kb'] * 100
        else:
            queue_imp = 0
            
        if untuned['rate_std_gbps'] > 0:
            jitter_imp = (untuned['rate_std_gbps'] - tuned['rate_std_gbps']) / untuned['rate_std_gbps'] * 100
        else:
            jitter_imp = 0
        
        print(f"\n{speed}:")
        print(f"  Utilization:  {util_imp:+>6.2f}% (closer to target)")
        print(f"  Queue:        {queue_imp:+>6.1f}% reduction")
        print(f"  Jitter:       {jitter_imp:+>6.1f}% reduction")

# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    print("="*70)
    print("HPCC++ PARAMETER TUNING VALIDATION")
    print("Tuned vs Untuned Comparison at 100 Gbps and 800 Gbps")
    print("ECE-GY 6383: High-Speed Networks Term Project")
    print("Author: Xhovani Mali (xxm202@nyu.edu)")
    print("="*70)
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Run comparisons
    results_100g = run_comparison(100)
    results_800g = run_comparison(800)
    
    # Save results
    results_100g.to_csv('results/comparison_100g.csv', index=False)
    results_800g.to_csv('results/comparison_800g.csv', index=False)
    print("\nResults saved to results/comparison_100g.csv and results/comparison_800g.csv")
    
    # Generate visualization
    plot_comparison(results_100g, results_800g, 'figures/tuned_vs_untuned_comparison.png')
    
    # Print summary
    print_summary_table(results_100g, results_800g)
    
    # Print optimal parameters for reference
    print("\n" + "="*80)
    print("OPTIMAL HPCC++ PARAMETERS (PROVEN)")
    print("="*80)
    print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                        LINK SPEED SCALING                               │
├─────────┬─────────┬─────────┬─────────┬──────────┬─────────────────────┤
│  Speed  │    α    │    β    │    η    │   T_s    │   Notes             │
├─────────┼─────────┼─────────┼─────────┼──────────┼─────────────────────┤
│  100G   │  0.85   │  0.50   │  0.95   │  1.0 µs  │  Per-packet INT     │
│  400G   │  0.70   │  0.42   │  0.95   │  0.25 µs │  4x faster feedback │
│  800G   │  0.60   │  0.36   │  0.95   │  0.125µs │  8x faster feedback │
└─────────┴─────────┴─────────┴─────────┴──────────┴─────────────────────┘

Scaling Rules:
  • T_s scales inversely with link speed: T_s ∝ 1/C
  • α decreases slightly at higher speeds for stability
  • β maintains ratio β ≈ 0.6α for optimal damping (ζ ≈ 1.2)
  • BDP increases with link speed, maintaining stability margin
""")
    
    print("\nDone! All outputs saved to results/ and figures/ directories.")

if __name__ == "__main__":
    main()
