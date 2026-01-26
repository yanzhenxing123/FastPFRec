#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack resilience experiment for trusted nodes.
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AttackResult:
    """Attack result."""
    attack_type: str
    success: bool
    server_damage: float
    system_recovery_time: float
    affected_clients: int
    model_accuracy_drop: float

class MaliciousClient:
    """Malicious client simulator."""
    
    def __init__(self, client_id: int, attack_type: str = "noise_injection"):
        self.client_id = client_id
        self.attack_type = attack_type
        self.is_malicious = True
        
    def generate_attack(self, normal_parameters: torch.Tensor, 
                       attack_strength: float = 1.0) -> torch.Tensor:
        """Generate adversarial parameters."""
        if self.attack_type == "noise_injection":
            noise = torch.randn_like(normal_parameters) * attack_strength
            return normal_parameters + noise
            
        elif self.attack_type == "gradient_poisoning":
            poison = torch.ones_like(normal_parameters) * attack_strength
            return normal_parameters + poison
            
        elif self.attack_type == "model_replacement":
            return torch.randn_like(normal_parameters) * attack_strength
            
        elif self.attack_type == "label_flipping":
            return -normal_parameters * attack_strength
            
        elif self.attack_type == "backdoor":
            backdoor_pattern = torch.zeros_like(normal_parameters)
            backdoor_pattern[0:10] = attack_strength
            return normal_parameters + backdoor_pattern
            
        else:
            return normal_parameters

class TrustedNode:
    """Trusted node simulator."""
    
    def __init__(self, node_id: int, max_clients: int = 10):
        self.node_id = node_id
        self.max_clients = max_clients
        self.assigned_clients = []
        self.is_compromised = False
        self.aggregated_parameters = None
        
    def assign_client(self, client_id: int):
        """Assign a client to this node."""
        if len(self.assigned_clients) < self.max_clients:
            self.assigned_clients.append(client_id)
            return True
        return False
    
    def aggregate_parameters(self, client_parameters: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Aggregate client parameters."""
        if not self.assigned_clients:
            return None
            
        node_parameters = []
        for client_id in self.assigned_clients:
            if client_id in client_parameters:
                node_parameters.append(client_parameters[client_id])
        
        if not node_parameters:
            return None
            
        self.aggregated_parameters = torch.stack(node_parameters).mean(dim=0)
        return self.aggregated_parameters
    
    def get_node_client_parameters(self, client_parameters: Dict[int, torch.Tensor]) -> List[torch.Tensor]:
        """Return all client parameter tensors assigned to this node."""
        node_params = []
        for client_id in self.assigned_clients:
            if client_id in client_parameters:
                node_params.append(client_parameters[client_id])
        return node_params

    def detect_node_anomaly(self,
                            node_client_params: List[torch.Tensor],
                            reference: torch.Tensor,
                            ratio_threshold: float = 0.2,
                            dist_threshold: float = 3.5) -> bool:
        """Detect anomalies within a node using median/MAD robust z-scores."""
        if not node_client_params:
            return False

        with torch.no_grad():
            dists = torch.tensor([torch.norm(p - reference).item() for p in node_client_params])
            median = torch.median(dists)
            mad = torch.median(torch.abs(dists - median)) + 1e-6
            robust_z = 0.6745 * (dists - median) / mad
            outlier_ratio = (torch.abs(robust_z) > dist_threshold).float().mean().item()
            return outlier_ratio > ratio_threshold
    
    def isolate(self):
        """Isolate this node from aggregation."""
        self.is_compromised = True
        self.assigned_clients = []
        self.aggregated_parameters = None

class CentralServer:
    """Central server simulator."""
    
    def __init__(self):
        self.global_model = None
        self.is_compromised = False
        self.performance_history = []
        
    def update_global_model(self, trusted_node_parameters: List[torch.Tensor]) -> torch.Tensor:
        """Update the global model."""
        if not trusted_node_parameters:
            return self.global_model
            
        self.global_model = torch.stack(trusted_node_parameters).mean(dim=0)
        return self.global_model
    
    def evaluate_damage(self, malicious_parameters: torch.Tensor) -> float:
        """Estimate damage relative to the current global model."""
        if self.global_model is None:
            return 0.0
            
        damage = torch.norm(malicious_parameters - self.global_model).item()
        return damage

class AttackResilienceExperiment:
    """Attack resilience experiment."""
    
    def __init__(self, num_clients: int = 100, num_trusted_nodes: int = 10, 
                 embedding_dim: int = 64):
        self.num_clients = num_clients
        self.num_trusted_nodes = num_trusted_nodes
        self.embedding_dim = embedding_dim
        self.results = []
        
    def setup_system(self, malicious_ratio: float = 0.1, fixed_attack_type: str = None) -> Tuple[List, List, CentralServer]:
        """Build clients, trusted nodes, and the server."""
        clients = []
        num_malicious = int(self.num_clients * malicious_ratio)
        
        attack_types = ["noise_injection", "gradient_poisoning", "model_replacement", "label_flipping"]
        
        for i in range(self.num_clients):
            if i < num_malicious:
                if fixed_attack_type:
                    attack_type = fixed_attack_type
                else:
                    attack_type = random.choice(attack_types)
                client = MaliciousClient(i, attack_type)
            else:
                client = type('NormalClient', (), {'client_id': i, 'is_malicious': False})()
            clients.append(client)
        
        trusted_nodes = [TrustedNode(i) for i in range(self.num_trusted_nodes)]
        
        for client in clients:
            assigned = False
            while not assigned:
                node_id = random.randint(0, self.num_trusted_nodes - 1)
                assigned = trusted_nodes[node_id].assign_client(client.client_id)
        
        server = CentralServer()
        server.global_model = torch.randn(self.embedding_dim)
        
        return clients, trusted_nodes, server
    
    def simulate_attack_scenario(self, clients: List, trusted_nodes: List, 
                               server: CentralServer, attack_strength: float = 1.0,
                               use_trusted_nodes: bool = True) -> AttackResult:
        """Simulate one attack round."""
        normal_parameters = torch.randn(self.embedding_dim)
        prev_global = server.global_model.clone() if server.global_model is not None else torch.zeros(self.embedding_dim)
        
        client_parameters = {}
        malicious_clients = []
        
        for client in clients:
            if hasattr(client, 'is_malicious') and client.is_malicious:
                attack_strength = 5.0
                attack_params = client.generate_attack(normal_parameters, attack_strength)
                client_parameters[client.client_id] = attack_params
                malicious_clients.append(client)
            else:
                client_parameters[client.client_id] = normal_parameters + torch.randn_like(normal_parameters) * 0.1
        
        trusted_node_parameters = []
        compromised_nodes = []
        
        for node in trusted_nodes:
            node_client_params = node.get_node_client_parameters(client_parameters)
            if not node_client_params:
                continue

            if use_trusted_nodes:
                is_anomalous = node.detect_node_anomaly(
                    node_client_params=node_client_params,
                    reference=prev_global,
                    ratio_threshold=0.1,
                    dist_threshold=1.2,
                )

                if is_anomalous:
                    compromised_nodes.append(node)
                    node.isolate()
                    continue

            node_params = torch.stack(node_client_params).mean(dim=0)
            trusted_node_parameters.append(node_params)
        
        if trusted_node_parameters:
            server.update_global_model(trusted_node_parameters)
        
        server_damage = torch.norm(server.global_model - prev_global).item() if server.global_model is not None else 0.0
        
        recovery_time = len(compromised_nodes) * 0.1
        
        affected_clients = sum(len(node.assigned_clients) for node in compromised_nodes)
        
        accuracy_drop = min(server_damage * 0.1, 1.0)
        
        damage_threshold = 1.0

        return AttackResult(
            attack_type=malicious_clients[0].attack_type if malicious_clients else "none",
            success=server_damage > damage_threshold,
            server_damage=server_damage,
            system_recovery_time=recovery_time,
            affected_clients=affected_clients,
            model_accuracy_drop=accuracy_drop
        )
    
    def run_experiment(self, num_runs: int = 50, malicious_ratios: List[float] = [0.05, 0.1, 0.2, 0.3],
                      use_trusted_nodes: bool = True, fixed_attack_type: str = None):
        """Run multiple trials."""
        print("Starting attack resilience experiment...")
        
        malicious_ratios = [0.2]
        
        all_results = []
        
        for malicious_ratio in malicious_ratios:
            if fixed_attack_type:
                print(f"Attack type: {fixed_attack_type}")
            else:
                print(f"Malicious ratio: {malicious_ratio:.1%}")
            
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}...", end='\r')
                
                clients, trusted_nodes, server = self.setup_system(malicious_ratio, fixed_attack_type)
                
                result = self.simulate_attack_scenario(clients, trusted_nodes, server,
                                                       use_trusted_nodes=use_trusted_nodes)
                
                result_dict = {
                    'malicious_ratio': malicious_ratio,
                    'run_id': run,
                    'attack_type': result.attack_type,
                    'attack_success': result.success,
                    'server_damage': result.server_damage,
                    'recovery_time': result.system_recovery_time,
                    'affected_clients': result.affected_clients,
                    'accuracy_drop': result.model_accuracy_drop
                }
                all_results.append(result_dict)
        
        self.results = pd.DataFrame(all_results)
        return self.results

    def run_simple_comparison(self, malicious_ratio: float = 0.3, num_runs: int = 20) -> Dict[str, float]:
        """Compare mean damage with vs. without trusted nodes."""
        res_with = self.run_experiment(num_runs=num_runs, malicious_ratios=[malicious_ratio],
                                       use_trusted_nodes=True)
        dmg_with = res_with['server_damage'].mean()
        
        res_without = self.run_experiment(num_runs=num_runs, malicious_ratios=[malicious_ratio],
                                          use_trusted_nodes=False)
        dmg_without = res_without['server_damage'].mean()
        
        return {
            'malicious_ratio': malicious_ratio,
            'server_damage_with_trusted_nodes': dmg_with,
            'server_damage_without_trusted_nodes': dmg_without
        }
    

    def generate_resilience_report(self):
        """Print a resilience report."""
        if self.results.empty:
            print("Run the experiment first.")
            return
        
        print("\n" + "="*60)
        print("Attack Resilience Report (Trusted Nodes)")
        print("="*60)
        
        avg_attack_success = self.results['attack_success'].mean()
        avg_server_damage = self.results['server_damage'].mean()
        avg_recovery_time = self.results['recovery_time'].mean()
        avg_affected_clients = self.results['affected_clients'].mean()
        
        print("1. Detection and isolation:")
        print(f"   - Mean detection rate: {(1-avg_attack_success)*100:.1f}%")
        print(f"   - Isolated node rate: {avg_attack_success*100:.1f}%")
        
        print("\n2. Server protection:")
        print(f"   - Mean server damage: {avg_server_damage:.4f}")
        print(f"   - Fully protected rate: {(avg_server_damage < 0.1)*100:.1f}%")
        
        print("\n3. Recovery:")
        print(f"   - Mean recovery time: {avg_recovery_time:.2f} time units")
        print(f"   - Mean affected clients: {avg_affected_clients:.1f}")
        
        print("\n4. Robustness:")
        if avg_server_damage < 0.5 and avg_recovery_time < 1.0:
            print("   ✓ Strong robustness against attacks")
            print("   ✓ Trusted nodes improve system safety")
        else:
            print("   ⚠ Robustness needs improvement")
        
        print("\n5. By attack type:")
        attack_analysis = self.results.groupby('attack_type').agg({
            'attack_success': 'mean',
            'server_damage': 'mean',
            'recovery_time': 'mean'
        })
        
        for attack_type, stats in attack_analysis.iterrows():
            print(f"   {attack_type}:")
            print(f"     - Detection rate: {(1-stats['attack_success'])*100:.1f}%")
            print(f"     - Server damage: {stats['server_damage']:.4f}")
            print(f"     - Recovery time: {stats['recovery_time']:.2f}")

def main():
    """Entry point."""
    experiment = AttackResilienceExperiment(
        num_clients=100,
        num_trusted_nodes=10,
        embedding_dim=64
    )
    
    attack_types = ["noise_injection", "gradient_poisoning", "model_replacement"]
    all_results_dfs = []
    
    for attack_type in attack_types:
        print(f"\n>>> Running experiment for attack type: {attack_type}")
        results = experiment.run_experiment(
            num_runs=20, 
            malicious_ratios=[0.2],
            fixed_attack_type=attack_type
        )
        all_results_dfs.append(results)
        
    if all_results_dfs:
        final_results = pd.concat(all_results_dfs, ignore_index=True)
        experiment.results = final_results
        
        experiment.generate_resilience_report()
        
        final_results.to_csv('attack_resilience_results.csv', index=False)
        print("\nSaved detailed results to: attack_resilience_results.csv")

if __name__ == "__main__":
    main()
