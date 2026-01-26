#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper script to run the attack resilience experiment.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import pandas as pd



from attack_exp.attack_resilience_experiment import AttackResilienceExperiment


def run_quick_experiment():
    """Run a small, fast experiment."""
    print("="*60)
    print("Attack Resilience (Quick)")
    print("="*60)
    
    experiment = AttackResilienceExperiment(
        num_clients=50,
        num_trusted_nodes=5,
        embedding_dim=32
    )
    
    print("Running...")
    
    all_results = pd.DataFrame()
    
    attack_types = ["noise_injection", "gradient_poisoning", "label_flipping"]
    
    for attack_type in attack_types:
        print(f"\n>>> Attack type: {attack_type}")
        results = experiment.run_experiment(
            num_runs=20,
            malicious_ratios=[0.2],
            use_trusted_nodes=True,
            fixed_attack_type=attack_type
        )
        all_results = pd.concat([all_results, results])
        
    experiment.results = all_results
    
    experiment.generate_resilience_report()
    
    return all_results

def run_comprehensive_experiment():
    """Run a larger experiment."""
    print("="*60)
    print("Attack Resilience (Full)")
    print("="*60)
    
    experiment = AttackResilienceExperiment(
        num_clients=100,
        num_trusted_nodes=10,
        embedding_dim=64
    )
    
    print("Running...")
    results = experiment.run_experiment(
        num_runs=50,
        malicious_ratios=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        use_trusted_nodes=True
    )
    
    experiment.generate_resilience_report()
    
    return results

def analyze_results(results):
    """Print basic analysis for the collected results."""
    print("\n" + "="*60)
    print("Results Analysis")
    print("="*60)
    
    print("1. By malicious ratio:")
    for ratio in sorted(results['malicious_ratio'].unique()):
        subset = results[results['malicious_ratio'] == ratio]
        success_rate = subset['attack_success'].mean()
        avg_damage = subset['server_damage'].mean()
        avg_recovery = subset['recovery_time'].mean()
        
        print(f"   Malicious ratio {ratio:.1%}:")
        print(f"     - Attack success rate: {success_rate:.2%}")
        print(f"     - Mean server damage: {avg_damage:.4f}")
        print(f"     - Mean recovery time: {avg_recovery:.2f}")
    
    print("\n2. By attack type:")
    attack_types = results['attack_type'].unique()
    for attack_type in attack_types:
        if attack_type != 'none':
            subset = results[results['attack_type'] == attack_type]
            success_rate = subset['attack_success'].mean()
            avg_damage = subset['server_damage'].mean()
            
            print(f"   {attack_type}:")
            print(f"     - Attack success rate: {success_rate:.2%}")
            print(f"     - Mean server damage: {avg_damage:.4f}")
    
    print("\n3. Key metrics:")
    total_attacks = len(results[results['attack_type'] != 'none'])
    successful_attacks = len(results[results['attack_success'] == True])
    detection_rate = (total_attacks - successful_attacks) / total_attacks if total_attacks > 0 else 0
    
    print(f"   - Total attacks: {total_attacks}")
    print(f"   - Detected attacks: {total_attacks - successful_attacks}")
    print(f"   - Detection rate: {detection_rate:.2%}")
    print(f"   - Protection rate: {(1 - results['server_damage'].mean()):.2%}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run attack resilience experiments')
    parser.add_argument('--mode', choices=['quick', 'full', 'simple'], default='quick',
                       help='Experiment mode: quick, full, or simple')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        results = run_quick_experiment()
    elif args.mode == 'full':
        results = run_comprehensive_experiment()
    else:
        exp = AttackResilienceExperiment(num_clients=80, num_trusted_nodes=8, embedding_dim=32)
        summary = exp.run_simple_comparison(malicious_ratio=0.3, num_runs=30)
        print("\nSimple comparison (parameter perturbation only):")
        print(f"  Malicious ratio: {summary['malicious_ratio']:.0%}")
        print(f"  With trusted nodes - server damage: {summary['server_damage_with_trusted_nodes']:.4f}")
        print(f"  Without trusted nodes - server damage: {summary['server_damage_without_trusted_nodes']:.4f}")
        results = exp.results
    
    analyze_results(results)
    
    print("\nDone. Results are saved to CSV.")
