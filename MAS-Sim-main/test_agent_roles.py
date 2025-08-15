#!/usr/bin/env python3
"""
Test script for agent role assignment
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_generator import AdvancedGraphGenerator
from network_topologies import TopologyType, NetworkConfig
from core_agent import AgentRole

def test_agent_role_assignment():
    """Test agent role assignment based on system prompts"""
    print("=== Testing Agent Role Assignment ===")
    
    generator = AdvancedGraphGenerator(verbose=True)
    
    # Test with different attacker ratios
    test_configs = [
        {"attacker_ratio": 0.2, "num_agents": 5},
        {"attacker_ratio": 0.5, "num_agents": 6},
        {"attacker_ratio": 0.8, "num_agents": 4}
    ]
    
    for config in test_configs:
        print(f"\n--- Testing with {config['attacker_ratio']} attacker ratio, {config['num_agents']} agents ---")
        
        # Generate system prompts and agent roles
        system_prompts, agent_roles = generator.generate_system_prompts(
            num_agents=config['num_agents'],
            attacker_ratio=config['attacker_ratio'],
            attacker_strategy="persuasion"
        )
        
        print("System prompts:")
        for i, prompt in enumerate(system_prompts):
            print(f"  Agent {i}: {prompt[:100]}...")
        
        # Test with different topologies
        topologies = [
            TopologyType.LINEAR,
            TopologyType.TREE_HIERARCHY,
            TopologyType.P2P_FLAT
        ]
        
        for topology_type in topologies:
            print(f"\n  Testing {topology_type.value} topology:")
            
            # Create network config
            config_obj = NetworkConfig(
                topology_type=topology_type,
                num_agents=config['num_agents'],
                sparsity=0.2
            )
            
            # Create topology
            topology = generator.create_network_topology(config_obj, system_prompts, agent_roles)
            
            # Check agent roles
            attacker_count = 0
            normal_count = 0
            
            for agent_id, role in topology.agent_roles.items():
                if role == AgentRole.ATTACKER:
                    attacker_count += 1
                    print(f"    {agent_id}: ATTACKER")
                else:
                    normal_count += 1
                    print(f"    {agent_id}: {role.value}")
            
            expected_attackers = int(config['num_agents'] * config['attacker_ratio'])
            print(f"    Expected attackers: {expected_attackers}, Actual: {attacker_count}")
            print(f"    Expected normal: {config['num_agents'] - expected_attackers}, Actual: {normal_count}")
            
            if attacker_count == expected_attackers:
                print("    ✓ Role assignment correct!")
            else:
                print("    ✗ Role assignment incorrect!")

if __name__ == "__main__":
    test_agent_role_assignment() 