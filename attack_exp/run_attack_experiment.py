#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行攻击鲁棒性实验的简化脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd



from attack_exp.attack_resilience_experiment import AttackResilienceExperiment


def run_quick_experiment():
    """运行快速实验"""
    print("="*60)
    print("可信节点攻击鲁棒性快速实验")
    print("="*60)
    
    # 创建实验实例
    experiment = AttackResilienceExperiment(
        num_clients=50,      # 减少客户端数量以加快实验
        num_trusted_nodes=5, # 减少可信节点数量
        embedding_dim=32     # 减少嵌入维度
    )
    
    # 运行实验
    print("正在运行实验...")
    
    all_results = pd.DataFrame()
    
    # 分别测试三种攻击类型
    attack_types = ["noise_injection", "gradient_poisoning", "label_flipping"]
    
    for attack_type in attack_types:
        print(f"\n>>> 测试攻击类型: {attack_type}")
        results = experiment.run_experiment(
            num_runs=20,  # 每个类型运行20次
            malicious_ratios=[0.2],  # 固定20%
            use_trusted_nodes=True,
            fixed_attack_type=attack_type
        )
        all_results = pd.concat([all_results, results])
        
    experiment.results = all_results
    
    # 生成报告
    experiment.generate_resilience_report()
    
    # 绘制结果
    
    return all_results

def run_comprehensive_experiment():
    """运行完整实验"""
    print("="*60)
    print("可信节点攻击鲁棒性完整实验")
    print("="*60)
    
    # 创建实验实例
    experiment = AttackResilienceExperiment(
        num_clients=100,
        num_trusted_nodes=10,
        embedding_dim=64
    )
    
    # 运行实验
    print("正在运行完整实验...")
    results = experiment.run_experiment(
        num_runs=50,
        malicious_ratios=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        use_trusted_nodes=True
    )
    
    # 生成报告
    experiment.generate_resilience_report()
    
    # 绘制结果
    # experiment.plot_resilience_analysis('comprehensive_attack_resilience.pdf')
    
    return results

def analyze_results(results):
    """分析实验结果"""
    print("\n" + "="*60)
    print("实验结果深度分析")
    print("="*60)
    
    # 按恶意比例分析
    print("1. 按恶意客户端比例分析:")
    for ratio in sorted(results['malicious_ratio'].unique()):
        subset = results[results['malicious_ratio'] == ratio]
        success_rate = subset['attack_success'].mean()
        avg_damage = subset['server_damage'].mean()
        avg_recovery = subset['recovery_time'].mean()
        
        print(f"   恶意比例 {ratio:.1%}:")
        print(f"     - 攻击成功率: {success_rate:.2%}")
        print(f"     - 平均服务器损伤: {avg_damage:.4f}")
        print(f"     - 平均恢复时间: {avg_recovery:.2f}")
    
    # 按攻击类型分析
    print("\n2. 按攻击类型分析:")
    attack_types = results['attack_type'].unique()
    for attack_type in attack_types:
        if attack_type != 'none':
            subset = results[results['attack_type'] == attack_type]
            success_rate = subset['attack_success'].mean()
            avg_damage = subset['server_damage'].mean()
            
            print(f"   {attack_type}:")
            print(f"     - 攻击成功率: {success_rate:.2%}")
            print(f"     - 平均服务器损伤: {avg_damage:.4f}")
    
    # 计算关键指标
    print("\n3. 关键性能指标:")
    total_attacks = len(results[results['attack_type'] != 'none'])
    successful_attacks = len(results[results['attack_success'] == True])
    detection_rate = (total_attacks - successful_attacks) / total_attacks if total_attacks > 0 else 0
    
    print(f"   - 总攻击次数: {total_attacks}")
    print(f"   - 成功检测次数: {total_attacks - successful_attacks}")
    print(f"   - 攻击检测率: {detection_rate:.2%}")
    print(f"   - 系统保护率: {(1 - results['server_damage'].mean()):.2%}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行攻击鲁棒性实验')
    parser.add_argument('--mode', choices=['quick', 'full', 'simple'], default='quick',
                       help='实验模式: quick (快速) 或 full (完整)')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        results = run_quick_experiment()
    elif args.mode == 'full':
        results = run_comprehensive_experiment()
    else:
        # 最简对比：展示有无可信节点时服务器损伤的差异
        exp = AttackResilienceExperiment(num_clients=80, num_trusted_nodes=8, embedding_dim=32)
        summary = exp.run_simple_comparison(malicious_ratio=0.3, num_runs=30)
        print("\n最简对比(仅参数扰动攻击):")
        print(f"  恶意比例: {summary['malicious_ratio']:.0%}")
        print(f"  有可信节点-服务器损伤: {summary['server_damage_with_trusted_nodes']:.4f}")
        print(f"  无可信节点-服务器损伤: {summary['server_damage_without_trusted_nodes']:.4f}")
        results = exp.results
    
    # 分析结果
    analyze_results(results)
    
    print(f"\n实验完成！结果已保存到CSV文件。")
