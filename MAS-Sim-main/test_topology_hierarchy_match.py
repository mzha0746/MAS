#!/usr/bin/env python3
"""
Test topology and hierarchy matching
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NewMA.agent_hierarchy import HierarchyManager
from NewMA.network_topologies import NetworkConfig, TopologyType, LinearPipelineTopology, TreeHierarchyTopology, HolarchyTopology, P2PTopology, HybridTopology
from NewMA.core_agent import AgentRole
import networkx as nx

def test_topology_hierarchy_matching():
    """Test that hierarchy analysis matches actual topology types"""
    print("=== Testing Topology-Hierarchy Matching ===")
    
    hierarchy_manager = HierarchyManager()
    
    # Test configurations for different topology types
    test_configs = [
        ("linear", TopologyType.LINEAR, 4),
        ("tree_hierarchy", TopologyType.TREE_HIERARCHY, 6),
        ("holarchy", TopologyType.HOLARCHY, 5),
        ("p2p_flat", TopologyType.P2P_FLAT, 4),
        ("p2p_structured", TopologyType.P2P_STRUCTURED, 5),
        ("hybrid", TopologyType.HYBRID, 6)
    ]
    
    for topology_name, topology_type, num_agents in test_configs:
        print(f"\n--- Testing {topology_name} topology ---")
        
        # Create network configuration
        config = NetworkConfig(
            topology_type=topology_type,
            num_agents=num_agents,
            sparsity=0.2
        )
        
        # Create topology
        if topology_type == TopologyType.LINEAR:
            topology = LinearPipelineTopology(config)
        elif topology_type == TopologyType.TREE_HIERARCHY:
            topology = TreeHierarchyTopology(config)
        elif topology_type == TopologyType.HOLARCHY:
            topology = HolarchyTopology(config)
        elif topology_type == TopologyType.P2P_FLAT:
            topology = P2PTopology(config)
        elif topology_type == TopologyType.P2P_STRUCTURED:
            topology = P2PTopology(config)
        elif topology_type == TopologyType.HYBRID:
            topology = HybridTopology(config)
        else:
            print(f"Unknown topology type: {topology_type}")
            continue
        
        # Create agents and setup connections
        system_prompts = [f"Prompt for agent {i}" for i in range(num_agents)]
        agent_roles = [AgentRole.NORMAL] * num_agents
        topology.create_agents(system_prompts, "gpt-4o-mini", agent_roles)
        topology.setup_connections()
        
        print(f"  Created {len(topology.agents)} agents")
        print(f"  Network graph has {topology.network_graph.number_of_nodes()} nodes and {topology.network_graph.number_of_edges()} edges")
        
        # Analyze hierarchy
        agent_ids = list(topology.agents.keys())
        hierarchies = hierarchy_manager.analyze_topology_hierarchy(
            topology_name, topology.network_graph, agent_ids
        )
        
        print(f"  Generated hierarchy for {len(hierarchies)} agents")
        
        # Check hierarchy distribution
        level_counts = {}
        role_counts = {}
        
        for agent_id, hierarchy in hierarchies.items():
            level = hierarchy.level.value
            role = hierarchy.role.value
            
            level_counts[level] = level_counts.get(level, 0) + 1
            role_counts[role] = role_counts.get(role, 0) + 1
            
            print(f"    {agent_id}: {level} - {role} (authority: {hierarchy.authority_level})")
        
        print(f"  Level distribution: {level_counts}")
        print(f"  Role distribution: {role_counts}")
        
        # Test topology type detection
        detected_type = hierarchy_manager._determine_topology_type(topology.network_graph)
        print(f"  Detected topology type: {detected_type}")
        print(f"  Expected topology type: {topology_name}")
        
        if detected_type == topology_name:
            print("  ✅ Topology type detection correct!")
        else:
            print(f"  ❌ Topology type detection incorrect! Expected {topology_name}, got {detected_type}")
    
    print("\n=== Topology-Hierarchy Matching Test Completed ===")

if __name__ == "__main__":
    test_topology_hierarchy_matching() 