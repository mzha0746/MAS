#!/usr/bin/env python3
"""
Test script for num_agents parameter handling
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_generator import AdvancedGraphGenerator
from network_topologies import TopologyType, NetworkConfig
from core_agent import AgentRole

def test_num_agents_handling():
    """Test num_agents parameter handling"""
    print("=== Testing num_agents Parameter Handling ===")
    
    generator = AdvancedGraphGenerator(verbose=True)
    
    # Test case 1: No num_agents specified (should use default list)
    print("\n--- Test Case 1: No num_agents specified ---")
    class MockArgs1:
        def __init__(self):
            self.num_agents = None
            self.num_graphs = 20
            self.attacker_ratio = 0.2
            self.attacker_strategy = "persuasion"
            self.model_type = "gpt-4o-mini"
    
    args1 = MockArgs1()
    # Simulate the logic from generate_advanced_graph_dataset
    if hasattr(args1, 'num_agents') and args1.num_agents:
        num_agents_list = [args1.num_agents]
    else:
        num_agents_list = [4, 6, 8, 10]
    
    print(f"Result: num_agents_list = {num_agents_list}")
    
    # Test case 2: num_agents specified (should use single value)
    print("\n--- Test Case 2: num_agents specified ---")
    class MockArgs2:
        def __init__(self):
            self.num_agents = 8
            self.num_graphs = 20
            self.attacker_ratio = 0.2
            self.attacker_strategy = "persuasion"
            self.model_type = "gpt-4o-mini"
    
    args2 = MockArgs2()
    # Simulate the logic from generate_advanced_graph_dataset
    if hasattr(args2, 'num_agents') and args2.num_agents:
        num_agents_list = [args2.num_agents]
    else:
        num_agents_list = [4, 6, 8, 10]
    
    print(f"Result: num_agents_list = {num_agents_list}")
    
    # Test case 3: Test with actual topology creation
    print("\n--- Test Case 3: Test with actual topology creation ---")
    config = NetworkConfig(
        topology_type=TopologyType.LINEAR,
        num_agents=6,
        sparsity=0.2
    )
    
    # Generate system prompts and roles
    system_prompts, agent_roles = generator.generate_system_prompts(
        num_agents=6,
        attacker_ratio=0.2,
        attacker_strategy="persuasion"
    )
    
    print(f"Generated {len(system_prompts)} system prompts")
    print(f"Generated {len(agent_roles)} agent roles")
    print(f"Attacker roles: {sum(1 for role in agent_roles if role == AgentRole.ATTACKER)}")
    print(f"Normal roles: {sum(1 for role in agent_roles if role == AgentRole.NORMAL)}")
    
    # Create topology
    topology = generator.create_network_topology(config, system_prompts, agent_roles)
    
    print(f"Created topology with {len(topology.agents)} agents")
    for agent_id, role in topology.agent_roles.items():
        print(f"  {agent_id}: {role.value}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_num_agents_handling() 