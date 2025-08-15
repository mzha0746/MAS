#!/usr/bin/env python3
"""
Test hierarchy integration with agents
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NewMA.agent_hierarchy import HierarchyManager, HierarchyLevel, HierarchyRole
from NewMA.network_topologies import NetworkConfig, TopologyType, LinearPipelineTopology
from NewMA.core_agent import AgentRole
from NewMA.agent_prompts import generate_unified_prompts
import networkx as nx

def test_hierarchy_integration():
    """Test that hierarchy information is properly integrated with agents"""
    print("=== Testing Hierarchy Integration ===")
    
    # Create hierarchy manager
    hierarchy_manager = HierarchyManager()
    
    # Create a simple network graph
    graph = nx.Graph()
    agent_ids = [f"Agent_{i}" for i in range(4)]
    for agent_id in agent_ids:
        graph.add_node(agent_id)
    
    # Add some edges for a star topology
    for i in range(1, 4):
        graph.add_edge("Agent_0", f"Agent_{i}")
    
    # Generate prompts with hierarchy
    prompts, roles = generate_unified_prompts(
        num_agents=4,
        attacker_ratio=0.25,
        topology_type="star",
        network_graph=graph,
        agent_ids=agent_ids,
        hierarchy_manager=hierarchy_manager
    )
    
    print(f"Generated {len(prompts)} prompts with hierarchy")
    
    # Create network configuration
    config = NetworkConfig(
        topology_type=TopologyType.LINEAR,
        num_agents=4
    )
    
    # Create topology
    topology = LinearPipelineTopology(config)
    
    # Get hierarchy information for agents
    hierarchy_info = {}
    agent_hierarchies = hierarchy_manager.analyze_topology_hierarchy("star", graph, agent_ids)
    for agent_id, agent_hierarchy in agent_hierarchies.items():
        hierarchy_info[agent_id] = {
            'level': agent_hierarchy.level.value if hasattr(agent_hierarchy.level, 'value') else str(agent_hierarchy.level),
            'role': agent_hierarchy.role.value if hasattr(agent_hierarchy.role, 'value') else str(agent_hierarchy.role),
            'authority_level': agent_hierarchy.authority_level,
            'subordinates': agent_hierarchy.subordinates,
            'supervisors': agent_hierarchy.supervisors,
            'peers': agent_hierarchy.peers,
            'responsibilities': agent_hierarchy.responsibilities,
            'permissions': agent_hierarchy.permissions
        }
    
    print(f"Generated hierarchy info for {len(hierarchy_info)} agents")
    
    # Convert roles to AgentRole enum
    agent_roles = []
    for role_str in roles:
        if role_str == "ATTACKER":
            agent_roles.append(AgentRole.ATTACKER)
        else:
            agent_roles.append(AgentRole.NORMAL)
    
    # Create agents with hierarchy information
    # Note: The topology creates agents with IDs like "linear_agent_0", but hierarchy_info uses "Agent_0"
    # We need to map the hierarchy info to the correct agent IDs
    mapped_hierarchy_info = {}
    for i, (agent_id, agent_hierarchy) in enumerate(hierarchy_info.items()):
        topology_agent_id = f"linear_agent_{i}"
        mapped_hierarchy_info[topology_agent_id] = agent_hierarchy
    
    topology.create_agents(prompts, "gpt-4o-mini", agent_roles, mapped_hierarchy_info)
    
    print(f"Created {len(topology.agents)} agents")
    
    # Verify hierarchy information is properly set
    for agent_id, agent in topology.agents.items():
        print(f"\nAgent {agent_id}:")
        print(f"  - Has hierarchy_info: {hasattr(agent, 'hierarchy_info')}")
        if hasattr(agent, 'hierarchy_info'):
            print(f"  - Hierarchy info: {agent.hierarchy_info}")
        print(f"  - State metadata: {agent.state.metadata}")
        
        # Check if hierarchy info is in state metadata
        if hasattr(agent.state, 'metadata') and 'hierarchy_info' in agent.state.metadata:
            print(f"  ✅ Hierarchy info properly stored in state")
        else:
            print(f"  ❌ Hierarchy info not found in state")
    
    # Test that system prompts don't have duplicate IMPORTANT sections
    for i, (agent_id, agent) in enumerate(topology.agents.items()):
        prompt = agent.system_prompt
        important_count = prompt.count("IMPORTANT:")
        print(f"\nAgent {agent_id} prompt analysis:")
        print(f"  - IMPORTANT sections: {important_count}")
        if important_count > 1:
            print(f"  ❌ Duplicate IMPORTANT sections found!")
        else:
            print(f"  ✅ No duplicate IMPORTANT sections")
        
        # Show first 200 characters of prompt
        print(f"  - Preview: {prompt[:200]}...")
    
    print("\n=== Hierarchy Integration Test Completed ===")

if __name__ == "__main__":
    test_hierarchy_integration() 