#!/usr/bin/env python3
"""
Test to verify that the create_agents method signature fix works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.network_topologies import TopologyType, NetworkConfig
from NewMA.core_agent import AgentRole

def test_create_agents_signature():
    """Test that create_agents method accepts the correct number of arguments for all topology types"""
    print("Testing create_agents method signature fix...")
    
    # Create generator
    generator = AdvancedGraphGenerator(verbose=True)
    
    # Test all topology types
    topology_types = [
        TopologyType.LINEAR,
        TopologyType.TREE_HIERARCHY,
        TopologyType.HOLARCHY,
        TopologyType.P2P_FLAT,
        TopologyType.HYBRID
    ]
    
    for topology_type in topology_types:
        print(f"\nTesting {topology_type.value} topology...")
        
        try:
            # Create config
            config = generator.generate_topology_config(
                topology_type=topology_type,
                num_agents=3,
                sparsity=0.2
            )
            
            # Create system prompts and agent roles
            system_prompts = [
                "You are Agent 1",
                "You are Agent 2", 
                "You are Agent 3"
            ]
            agent_roles = [AgentRole.NORMAL, AgentRole.ATTACKER, AgentRole.NORMAL]
            hierarchy_info = {
                "agent_1": {"level": "manager", "role": "coordinator"},
                "agent_2": {"level": "worker", "role": "specialist"},
                "agent_3": {"level": "worker", "role": "generalist"}
            }
            
            # Create topology - this should not raise an error now
            topology = generator.create_network_topology(
                config=config,
                system_prompts=system_prompts,
                agent_roles=agent_roles,
                hierarchy_info=hierarchy_info
            )
            
            print(f"✓ {topology_type.value} topology created successfully")
            print(f"  - Number of agents: {len(topology.agents)}")
            print(f"  - Agent roles: {list(topology.agent_roles.values())}")
            
            # Verify that agents were created with correct roles
            for agent_id, role in topology.agent_roles.items():
                print(f"    {agent_id}: {role.value}")
            
        except Exception as e:
            print(f"✗ Error creating {topology_type.value} topology: {e}")
            return False
    
    print("\n✓ All topology types created successfully!")
    return True

def test_holarchy_specific():
    """Test HolarchyTopology specifically since it was the source of the error"""
    print("\nTesting HolarchyTopology specifically...")
    
    try:
        from NewMA.advanced_topologies import HolarchyTopology
        from NewMA.network_topologies import NetworkConfig
        
        # Create config
        config = NetworkConfig(
            topology_type=TopologyType.HOLARCHY,
            num_agents=4,
            sparsity=0.2
        )
        
        # Create topology
        topology = HolarchyTopology(config)
        
        # Test create_agents with all arguments
        system_prompts = ["Agent 1", "Agent 2", "Agent 3", "Agent 4"]
        model_type = "gpt-4o-mini"
        agent_roles = [AgentRole.NORMAL, AgentRole.ATTACKER, AgentRole.NORMAL, AgentRole.NORMAL]
        hierarchy_info = {
            "holon_agent_0": {"level": "super", "role": "coordinator"},
            "holon_agent_1": {"level": "sub", "role": "specialist"},
            "holon_agent_2": {"level": "sub", "role": "worker"},
            "holon_agent_3": {"level": "sub", "role": "worker"}
        }
        
        # This should work now
        topology.create_agents(system_prompts, model_type, agent_roles, hierarchy_info)
        
        print("✓ HolarchyTopology.create_agents works with 4 arguments")
        print(f"  - Created {len(topology.agents)} agents")
        print(f"  - Agent roles: {list(topology.agent_roles.values())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error with HolarchyTopology: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing create_agents Method Signature Fix ===")
    
    test1_passed = test_create_agents_signature()
    test2_passed = test_holarchy_specific()
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed! create_agents method signature fix is working correctly.")
    else:
        print("\n✗ Some tests failed. create_agents method signature fix needs more work.")
        sys.exit(1) 