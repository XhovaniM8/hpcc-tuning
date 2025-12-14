#!/usr/bin/env python3
"""
HPCC++ Control-Theoretic Parameter Tuning Experiments
=====================================================
ECE-GY 6383: High-Speed Networks Term Project
Author: Xhovani Mali (xxm202@nyu.edu)

This script implements systematic parameter tuning following control theory:
1. Step response analysis (rise time, overshoot, settling time)
2. Stability boundary sweep (find critical α)
3. Parameter coupling demonstration
4. Multi-speed validation (100G, 400G, 800G)

Based on methodology from HPCC++ and UET congestion control literature.
"""

import subprocess
import re
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SIM_DIR = PROJECT_ROOT / "simulation"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = RESULTS_DIR / "data"

# Ensure directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class HPCCConfig:
    """HPCC++ configuration parameters"""
    name: str
    alpha: float = 0.15      # Proportional gain
    beta: float = 0.08       # Derivative gain (damping)
    T_us: float = 12.0       # Baseline RTT (µs)
    T_s_us: float = 10.0     # Sampling interval (µs)
    eta: float = 0.95        # Target utilization
    linkspeed_mbps: int = 100000  # Link speed (Mbps)
    
    def cli_args(self) -> str:
        return (f"-hpcc_eta {self.eta} -hpcc_T {self.T_us} "
                f"-hpcc_alpha {self.alpha} -hpcc_beta {self.beta} "
                f"-hpcc_Ts {self.T_s_us}")
    
    def stability_margin(self) -> float:
        """Calculate stability margin: how far from instability"""
        C_bps = self.linkspeed_mbps * 1e6
        C_Bps = C_bps / 8
        T_s_sec = self.T_s_us * 1e-6
        T_sec = self.T_us * 1e-6
        BDP = C_Bps * T_sec
        loop_gain = self.alpha * C_Bps * T_s_sec
        return (BDP - loop_gain) / BDP  # >0 means stable
    
    def damping_ratio(self) -> float:
        """Calculate damping ratio ζ = β / (α/2)"""
        return self.beta / (self.alpha / 2) if self.alpha > 0 else float('inf')


@dataclass
class ExperimentResult:
    """Results from a single simulation run"""
    config: HPCCConfig
    flows: int
    completed: int
    fcts_us: List[float]  # Flow completion times
    retx_packets: int
    nacks: int
    total_bytes: int
    
    @property
    def avg_fct(self) -> float:
        return np.mean(self.fcts_us) if self.fcts_us else 0
    
    @property
    def p99_fct(self) -> float:
        return np.percentile(self.fcts_us, 99) if self.fcts_us else 0
    
    @property
    def min_fct(self) -> float:
        return min(self.fcts_us) if self.fcts_us else 0
    
    @property
    def max_fct(self) -> float:
        return max(self.fcts_us) if self.fcts_us else 0
    
    @property
    def fct_std(self) -> float:
        return np.std(self.fcts_us) if len(self.fcts_us) > 1 else 0
    
    @property
    def completion_rate(self) -> float:
        return self.completed / self.flows if self.flows > 0 else 0


def parse_simulation_output(output: str, config: HPCCConfig, expected_flows: int) -> ExperimentResult:
    """Parse hpcc_tuned stdout to extract metrics"""
    fcts = []
    
    # Parse flow completion times: "Flow HPCC_X_Y finished at Z.ZZZ total bytes N"
    for match in re.finditer(r'Flow HPCC_\d+_\d+ finished at ([\d.]+) total bytes (\d+)', output):
        fct_us = float(match.group(1))
        fcts.append(fct_us)
    
    # Parse summary statistics
    retx_match = re.search(r'Retx packets:\s+(\d+)', output)
    retx = int(retx_match.group(1)) if retx_match else 0
    
    bytes_match = re.search(r'Total bytes:\s+(\d+)', output)
    total_bytes = int(bytes_match.group(1)) if bytes_match else 0
    
    flows_match = re.search(r'Flows:\s+(\d+)', output)
    flows = int(flows_match.group(1)) if flows_match else expected_flows
    
    # Count NACKs from "go back n" events
    nacks = output.count('go back n')
    
    return ExperimentResult(
        config=config,
        flows=expected_flows,
        completed=len(fcts),
        fcts_us=fcts,
        retx_packets=retx,
        nacks=nacks,
        total_bytes=total_bytes
    )


def run_experiment(config: HPCCConfig, traffic_file: str, 
                   nodes: int = 128, end_time_us: int = 10000,
                   paths: int = 4) -> Optional[ExperimentResult]:
    """Run a single simulation experiment"""
    
    # Count expected flows from traffic file
    traffic_path = SIM_DIR / "traffic" / traffic_file
    if not traffic_path.exists():
        print(f"Traffic file not found: {traffic_path}")
        return None
    
    with open(traffic_path) as f:
        content = f.read()
        conn_match = re.search(r'Connections (\d+)', content)
        expected_flows = int(conn_match.group(1)) if conn_match else 0
    
    cmd = [
        str(SIM_DIR / "hpcc_tuned"),
        "-strat", "ecmp_host",
        "-nodes", str(nodes),
        "-tm", str(traffic_path),
        "-end", str(end_time_us),
        "-linkspeed", str(config.linkspeed_mbps),
        "-paths", str(paths),
        "-o", "/dev/null"  # Don't write log file
    ] + config.cli_args().split()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return parse_simulation_output(result.stdout, config, expected_flows)
    except subprocess.TimeoutExpired:
        print(f"Timeout running {config.name}")
        return None
    except Exception as e:
        print(f"Error running {config.name}: {e}")
        return None


def create_traffic_file(name: str, nodes: int, senders: int, 
                        receiver: int, flow_size: int = 1000000):
    """Generate a traffic matrix file"""
    traffic_dir = SIM_DIR / "traffic"
    traffic_dir.mkdir(exist_ok=True)
    
    path = traffic_dir / name
    with open(path, 'w') as f:
        f.write(f"Nodes {nodes}\n")
        f.write(f"Connections {senders}\n")
        for i in range(senders):
            f.write(f"{i}->{receiver} start 0 size {flow_size}\n")
    
    return name


# =============================================================================
# EXPERIMENT 1: Stability Boundary Sweep
# =============================================================================
def experiment_stability_sweep():
    """
    Sweep α from conservative to aggressive to find instability boundary.
    Keep β = α/2 (critical damping) and observe where retx/NACKs spike.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Stability Boundary Sweep")
    print("="*70)
    print("Goal: Find critical α where system becomes unstable")
    print("Method: Sweep α from 0.05 to 0.30, β = α/2 (critical damping)")
    
    # Create traffic
    traffic = create_traffic_file("incast_16to1.txt", 128, 16, 64)
    
    alpha_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    results = []
    
    for alpha in alpha_values:
        beta = alpha / 2  # Critical damping ζ = 1
        config = HPCCConfig(
            name=f"alpha_{alpha:.2f}",
            alpha=alpha,
            beta=beta,
            T_us=10.0,
            T_s_us=1.0,  # Per-packet feedback
            linkspeed_mbps=100000
        )
        
        print(f"\nRunning α={alpha:.2f}, β={beta:.2f}...")
        print(f"  Stability margin: {config.stability_margin():.3f}")
        print(f"  Damping ratio ζ: {config.damping_ratio():.2f}")
        
        result = run_experiment(config, traffic, end_time_us=5000)
        if result:
            results.append({
                'alpha': alpha,
                'beta': beta,
                'stability_margin': config.stability_margin(),
                'damping_ratio': config.damping_ratio(),
                'avg_fct': result.avg_fct,
                'max_fct': result.max_fct,
                'fct_std': result.fct_std,
                'retx': result.retx_packets,
                'nacks': result.nacks,
                'completion_rate': result.completion_rate
            })
            print(f"  Avg FCT: {result.avg_fct:.1f} µs, Retx: {result.retx_packets}, NACKs: {result.nacks}")
    
    df = pd.DataFrame(results)
    df.to_csv(DATA_DIR / "stability_sweep.csv", index=False)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    ax.plot(df['alpha'], df['avg_fct'], 'b-o', linewidth=2, markersize=8)
    ax.fill_between(df['alpha'], 
                    df['avg_fct'] - df['fct_std'], 
                    df['avg_fct'] + df['fct_std'], alpha=0.3)
    ax.set_xlabel('α (Proportional Gain)', fontsize=12)
    ax.set_ylabel('Average FCT (µs)', fontsize=12)
    ax.set_title('Flow Completion Time vs α', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.semilogy(df['alpha'], df['retx'] + 1, 'r-o', linewidth=2, markersize=8)
    ax.set_xlabel('α (Proportional Gain)', fontsize=12)
    ax.set_ylabel('Retransmissions (log scale)', fontsize=12)
    ax.set_title('Retransmissions vs α', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(df['alpha'], df['stability_margin'], 'g-o', linewidth=2, markersize=8)
    ax.axhline(0, color='r', linestyle='--', linewidth=2, label='Instability threshold')
    ax.set_xlabel('α (Proportional Gain)', fontsize=12)
    ax.set_ylabel('Stability Margin', fontsize=12)
    ax.set_title('Stability Margin vs α', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(df['alpha'], df['completion_rate'] * 100, 'm-o', linewidth=2, markersize=8)
    ax.set_xlabel('α (Proportional Gain)', fontsize=12)
    ax.set_ylabel('Completion Rate (%)', fontsize=12)
    ax.set_title('Flow Completion Rate vs α', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Experiment 1: Stability Boundary Analysis\n(β = α/2, T_s = 1µs, 100 Gbps)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp1_stability_sweep.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: {FIGURES_DIR / 'exp1_stability_sweep.png'}")
    
    return df


# =============================================================================
# EXPERIMENT 2: Damping Ratio Effect
# =============================================================================
def experiment_damping_ratio():
    """
    Fix α, vary β to show damping effect.
    ζ < 0.7: underdamped (oscillation)
    ζ ≈ 1.0: critically damped (optimal)
    ζ > 1.2: overdamped (slow)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Damping Ratio Effect")
    print("="*70)
    print("Goal: Show effect of damping on system response")
    print("Method: Fix α=0.10, vary β to get ζ ∈ [0.4, 2.0]")
    
    traffic = create_traffic_file("incast_16to1.txt", 128, 16, 64)
    
    alpha = 0.10
    # β = ζ * (α/2), so β values for different ζ
    zeta_values = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    results = []
    
    for zeta in zeta_values:
        beta = zeta * (alpha / 2)
        config = HPCCConfig(
            name=f"zeta_{zeta:.1f}",
            alpha=alpha,
            beta=beta,
            T_us=10.0,
            T_s_us=1.0,
            linkspeed_mbps=100000
        )
        
        print(f"\nRunning ζ={zeta:.1f} (α={alpha}, β={beta:.3f})...")
        
        result = run_experiment(config, traffic, end_time_us=5000)
        if result:
            results.append({
                'zeta': zeta,
                'alpha': alpha,
                'beta': beta,
                'avg_fct': result.avg_fct,
                'max_fct': result.max_fct,
                'fct_std': result.fct_std,
                'retx': result.retx_packets,
                'nacks': result.nacks
            })
            print(f"  FCT: {result.avg_fct:.1f} ± {result.fct_std:.1f} µs, Retx: {result.retx_packets}")
    
    df = pd.DataFrame(results)
    df.to_csv(DATA_DIR / "damping_ratio.csv", index=False)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax = axes[0]
    ax.plot(df['zeta'], df['avg_fct'], 'b-o', linewidth=2, markersize=8)
    ax.axvspan(0.7, 1.2, alpha=0.2, color='green', label='Optimal range')
    ax.set_xlabel('Damping Ratio ζ', fontsize=12)
    ax.set_ylabel('Average FCT (µs)', fontsize=12)
    ax.set_title('FCT vs Damping', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(df['zeta'], df['fct_std'], 'r-o', linewidth=2, markersize=8)
    ax.axvspan(0.7, 1.2, alpha=0.2, color='green')
    ax.set_xlabel('Damping Ratio ζ', fontsize=12)
    ax.set_ylabel('FCT Std Dev (µs)', fontsize=12)
    ax.set_title('FCT Variance vs Damping', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.semilogy(df['zeta'], df['retx'] + 1, 'purple', marker='o', linewidth=2, markersize=8)
    ax.axvspan(0.7, 1.2, alpha=0.2, color='green')
    ax.set_xlabel('Damping Ratio ζ', fontsize=12)
    ax.set_ylabel('Retransmissions (log)', fontsize=12)
    ax.set_title('Retx vs Damping', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Experiment 2: Damping Ratio Effect (α=0.10 fixed)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp2_damping_ratio.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: {FIGURES_DIR / 'exp2_damping_ratio.png'}")
    
    return df


# =============================================================================
# EXPERIMENT 3: Sampling Interval (T_s) Effect
# =============================================================================
def experiment_sampling_interval():
    """
    Show effect of feedback frequency T_s.
    Faster feedback (smaller T_s) allows higher α while maintaining stability.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Sampling Interval Effect")
    print("="*70)
    print("Goal: Show how T_s affects stability and performance")
    print("Method: Fix α=0.10, β=0.05, vary T_s from 0.5µs to 20µs")
    
    traffic = create_traffic_file("incast_16to1.txt", 128, 16, 64)
    
    Ts_values = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    results = []
    
    for Ts in Ts_values:
        config = HPCCConfig(
            name=f"Ts_{Ts:.1f}",
            alpha=0.10,
            beta=0.05,
            T_us=10.0,
            T_s_us=Ts,
            linkspeed_mbps=100000
        )
        
        print(f"\nRunning T_s={Ts:.1f}µs...")
        print(f"  Stability margin: {config.stability_margin():.3f}")
        
        result = run_experiment(config, traffic, end_time_us=5000)
        if result:
            results.append({
                'T_s_us': Ts,
                'stability_margin': config.stability_margin(),
                'avg_fct': result.avg_fct,
                'max_fct': result.max_fct,
                'fct_std': result.fct_std,
                'retx': result.retx_packets
            })
            print(f"  FCT: {result.avg_fct:.1f} µs, Retx: {result.retx_packets}")
    
    df = pd.DataFrame(results)
    df.to_csv(DATA_DIR / "sampling_interval.csv", index=False)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.plot(df['T_s_us'], df['avg_fct'], 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Sampling Interval T_s (µs)', fontsize=12)
    ax.set_ylabel('Average FCT (µs)', fontsize=12)
    ax.set_title('FCT vs Feedback Frequency', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(df['T_s_us'], df['stability_margin'], 'g-o', linewidth=2, markersize=8)
    ax.axhline(0, color='r', linestyle='--', linewidth=2, label='Instability')
    ax.set_xlabel('Sampling Interval T_s (µs)', fontsize=12)
    ax.set_ylabel('Stability Margin', fontsize=12)
    ax.set_title('Stability vs Feedback Frequency', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Experiment 3: Sampling Interval Effect\n(α=0.10, β=0.05 fixed)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp3_sampling_interval.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: {FIGURES_DIR / 'exp3_sampling_interval.png'}")
    
    return df


# =============================================================================
# EXPERIMENT 4: Multi-Speed Scaling (100G, 400G, 800G)
# =============================================================================
def experiment_multi_speed():
    """
    Test if optimal parameters scale across link speeds.
    Key insight: As C increases, stability constraint tightens.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Multi-Speed Scaling")
    print("="*70)
    print("Goal: Verify parameter scaling across 100G, 400G, 800G")
    print("Method: Test fixed params and scaled params at each speed")
    
    traffic = create_traffic_file("incast_16to1.txt", 128, 16, 64)
    
    speeds = [100000, 400000, 800000]  # Mbps
    results = []
    
    for speed in speeds:
        speed_gbps = speed / 1000
        
        # Fixed parameters (may break at high speed)
        config_fixed = HPCCConfig(
            name=f"fixed_{speed_gbps}G",
            alpha=0.10,
            beta=0.05,
            T_us=10.0,
            T_s_us=1.0,
            linkspeed_mbps=speed
        )
        
        # Scaled parameters: reduce α proportionally to maintain stability margin
        # At 100G: α=0.10, at 400G: α=0.025, at 800G: α=0.0125
        scale_factor = 100000 / speed
        config_scaled = HPCCConfig(
            name=f"scaled_{speed_gbps}G",
            alpha=0.10 * scale_factor,
            beta=0.05 * scale_factor,
            T_us=10.0,
            T_s_us=1.0,
            linkspeed_mbps=speed
        )
        
        print(f"\n=== {speed_gbps} Gbps ===")
        
        # Fixed
        print(f"Fixed (α=0.10): stability margin = {config_fixed.stability_margin():.3f}")
        result_fixed = run_experiment(config_fixed, traffic, end_time_us=5000)
        if result_fixed:
            results.append({
                'speed_gbps': speed_gbps,
                'config': 'fixed',
                'alpha': config_fixed.alpha,
                'beta': config_fixed.beta,
                'stability_margin': config_fixed.stability_margin(),
                'avg_fct': result_fixed.avg_fct,
                'retx': result_fixed.retx_packets,
                'completion_rate': result_fixed.completion_rate
            })
            print(f"  FCT: {result_fixed.avg_fct:.1f} µs, Retx: {result_fixed.retx_packets}")
        
        # Scaled
        print(f"Scaled (α={config_scaled.alpha:.3f}): stability margin = {config_scaled.stability_margin():.3f}")
        result_scaled = run_experiment(config_scaled, traffic, end_time_us=5000)
        if result_scaled:
            results.append({
                'speed_gbps': speed_gbps,
                'config': 'scaled',
                'alpha': config_scaled.alpha,
                'beta': config_scaled.beta,
                'stability_margin': config_scaled.stability_margin(),
                'avg_fct': result_scaled.avg_fct,
                'retx': result_scaled.retx_packets,
                'completion_rate': result_scaled.completion_rate
            })
            print(f"  FCT: {result_scaled.avg_fct:.1f} µs, Retx: {result_scaled.retx_packets}")
    
    df = pd.DataFrame(results)
    df.to_csv(DATA_DIR / "multi_speed.csv", index=False)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    df_fixed = df[df['config'] == 'fixed']
    df_scaled = df[df['config'] == 'scaled']
    
    ax = axes[0]
    ax.bar(np.arange(len(df_fixed)) - 0.2, df_fixed['avg_fct'], 0.4, label='Fixed α=0.10', color='blue')
    ax.bar(np.arange(len(df_scaled)) + 0.2, df_scaled['avg_fct'], 0.4, label='Scaled α', color='green')
    ax.set_xticks(range(len(df_fixed)))
    ax.set_xticklabels([f"{s}G" for s in df_fixed['speed_gbps']])
    ax.set_xlabel('Link Speed', fontsize=12)
    ax.set_ylabel('Average FCT (µs)', fontsize=12)
    ax.set_title('FCT vs Speed', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1]
    ax.bar(np.arange(len(df_fixed)) - 0.2, df_fixed['stability_margin'], 0.4, label='Fixed', color='blue')
    ax.bar(np.arange(len(df_scaled)) + 0.2, df_scaled['stability_margin'], 0.4, label='Scaled', color='green')
    ax.axhline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xticks(range(len(df_fixed)))
    ax.set_xticklabels([f"{s}G" for s in df_fixed['speed_gbps']])
    ax.set_xlabel('Link Speed', fontsize=12)
    ax.set_ylabel('Stability Margin', fontsize=12)
    ax.set_title('Stability vs Speed', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[2]
    ax.bar(np.arange(len(df_fixed)) - 0.2, df_fixed['retx'], 0.4, label='Fixed', color='blue')
    ax.bar(np.arange(len(df_scaled)) + 0.2, df_scaled['retx'], 0.4, label='Scaled', color='green')
    ax.set_xticks(range(len(df_fixed)))
    ax.set_xticklabels([f"{s}G" for s in df_fixed['speed_gbps']])
    ax.set_xlabel('Link Speed', fontsize=12)
    ax.set_ylabel('Retransmissions', fontsize=12)
    ax.set_title('Retx vs Speed', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Experiment 4: Multi-Speed Scaling\n(Fixed vs Scaled Parameters)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp4_multi_speed.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: {FIGURES_DIR / 'exp4_multi_speed.png'}")
    
    return df


# =============================================================================
# EXPERIMENT 5: Incast Degree Scaling
# =============================================================================
def experiment_incast_scaling():
    """
    Test how optimal parameters hold under increasing incast severity.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Incast Degree Scaling")
    print("="*70)
    print("Goal: Test parameter robustness under 8:1, 16:1, 32:1, 64:1 incast")
    
    incast_degrees = [8, 16, 32, 64]
    results = []
    
    # Optimal config
    config = HPCCConfig(
        name="optimal",
        alpha=0.10,
        beta=0.05,
        T_us=10.0,
        T_s_us=1.0,
        linkspeed_mbps=100000
    )
    
    for degree in incast_degrees:
        traffic = create_traffic_file(f"incast_{degree}to1.txt", 128, degree, 64)
        
        print(f"\nRunning {degree}:1 incast...")
        result = run_experiment(config, traffic, end_time_us=10000)
        
        if result:
            results.append({
                'incast_degree': degree,
                'avg_fct': result.avg_fct,
                'max_fct': result.max_fct,
                'fct_std': result.fct_std,
                'retx': result.retx_packets,
                'completion_rate': result.completion_rate
            })
            print(f"  FCT: {result.avg_fct:.1f} µs (max {result.max_fct:.1f}), "
                  f"Retx: {result.retx_packets}, Complete: {result.completion_rate*100:.1f}%")
    
    df = pd.DataFrame(results)
    df.to_csv(DATA_DIR / "incast_scaling.csv", index=False)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax = axes[0]
    ax.plot(df['incast_degree'], df['avg_fct'], 'b-o', linewidth=2, markersize=8, label='Avg')
    ax.plot(df['incast_degree'], df['max_fct'], 'r--s', linewidth=2, markersize=8, label='Max')
    ax.set_xlabel('Incast Degree (N:1)', fontsize=12)
    ax.set_ylabel('FCT (µs)', fontsize=12)
    ax.set_title('FCT vs Incast Severity', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.semilogy(df['incast_degree'], df['retx'] + 1, 'r-o', linewidth=2, markersize=8)
    ax.set_xlabel('Incast Degree (N:1)', fontsize=12)
    ax.set_ylabel('Retransmissions (log)', fontsize=12)
    ax.set_title('Retx vs Incast Severity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.plot(df['incast_degree'], df['completion_rate'] * 100, 'g-o', linewidth=2, markersize=8)
    ax.set_xlabel('Incast Degree (N:1)', fontsize=12)
    ax.set_ylabel('Completion Rate (%)', fontsize=12)
    ax.set_title('Completion vs Incast Severity', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Experiment 5: Incast Scaling (α=0.10, β=0.05, T_s=1µs)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp5_incast_scaling.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: {FIGURES_DIR / 'exp5_incast_scaling.png'}")
    
    return df


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*70)
    print("HPCC++ Control-Theoretic Parameter Tuning")
    print("="*70)
    
    # Check simulator exists
    if not (SIM_DIR / "hpcc_tuned").exists():
        print(f"ERROR: Simulator not found at {SIM_DIR / 'hpcc_tuned'}")
        print("Run 'make' in simulation/ directory first")
        sys.exit(1)
    
    # Run all experiments
    results = {}
    
    results['stability'] = experiment_stability_sweep()
    results['damping'] = experiment_damping_ratio()
    results['sampling'] = experiment_sampling_interval()
    results['speed'] = experiment_multi_speed()
    results['incast'] = experiment_incast_scaling()
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    print("\nAll results saved to:", DATA_DIR)
    print("All figures saved to:", FIGURES_DIR)
    
    print("\nKey findings to check:")
    print("1. Stability sweep: Does FCT/retx spike at high α?")
    print("2. Damping ratio: Is ζ ∈ [0.7, 1.2] optimal?")
    print("3. Sampling interval: Does smaller T_s improve stability?")
    print("4. Multi-speed: Do fixed params break at 400G/800G?")
    print("5. Incast scaling: Do optimal params hold under severe incast?")


if __name__ == "__main__":
    main()
