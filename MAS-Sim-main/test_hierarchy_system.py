#!/usr/bin/env python3
"""
Test script to verify hierarchy system functionality
"""

import networkx as nx
from NewMA.agent_hierarchy import HierarchyManager, HierarchyLevel, HierarchyRole
from NewMA.agent_prompts import create_normal_agent_prompt, create_attacker_agent_prompt

def test_hierarchy_analysis():
    """Test hierarchy analysis for different topologies"""
    print("=== Testing Hierarchy Analysis ===")
    
    hierarchy_manager = HierarchyManager()
    
    # Test 1: Star topology
    print("\n1. Testing Star Topology")
    agent_ids = [f"Agent_{i}" for i in range(5)]
    star_graph = nx.Graph()
    star_graph.add_edges_from([
        ("Agent_0", "Agent_1"), ("Agent_0", "Agent_2"), ("Agent_0", "Agent_3"), ("Agent_0", "Agent_4")
    ])
    
    hierarchies = hierarchy_manager.analyze_topology_hierarchy("star", star_graph, agent_ids)
    
    for agent_id, hierarchy in hierarchies.items():
        print(f"  {agent_id}: {hierarchy.level.value} - {hierarchy.role.value} (Authority: {hierarchy.authority_level})")
    
    # Test 2: Ring topology
    print("\n2. Testing Ring Topology")
    ring_graph = nx.Graph()
    ring_graph.add_edges_from([
        ("Agent_0", "Agent_1"), ("Agent_1", "Agent_2"), ("Agent_2", "Agent_3"), ("Agent_3", "Agent_4"), ("Agent_4", "Agent_0")
    ])
    
    hierarchies = hierarchy_manager.analyze_topology_hierarchy("ring", ring_graph, agent_ids)
    
    for agent_id, hierarchy in hierarchies.items():
        print(f"  {agent_id}: {hierarchy.level.value} - {hierarchy.role.value} (Authority: {hierarchy.authority_level})")
    
    # Test 3: Tree topology
    print("\n3. Testing Tree Topology")
    tree_graph = nx.Graph()
    tree_graph.add_edges_from([
        ("Agent_0", "Agent_1"), ("Agent_0", "Agent_2"),  # Root level
        ("Agent_1", "Agent_3"), ("Agent_1", "Agent_4"),  # Second level
        ("Agent_2", "Agent_5"), ("Agent_2", "Agent_6")   # Second level
    ])
    agent_ids_tree = [f"Agent_{i}" for i in range(7)]
    hierarchies = hierarchy_manager.analyze_topology_hierarchy("tree", tree_graph, agent_ids_tree)
    
    for agent_id, hierarchy in hierarchies.items():
        print(f"  {agent_id}: {hierarchy.level.value} - {hierarchy.role.value} (Authority: {hierarchy.authority_level})")
    
    # Test 4: Mesh topology
    print("\n4. Testing Mesh Topology")
    mesh_graph = nx.Graph()
    mesh_graph.add_edges_from([
        ("Agent_0", "Agent_1"), ("Agent_0", "Agent_2"), ("Agent_0", "Agent_3"), ("Agent_0", "Agent_4"),
        ("Agent_1", "Agent_2"), ("Agent_1", "Agent_3"), ("Agent_1", "Agent_4"),
        ("Agent_2", "Agent_3"), ("Agent_2", "Agent_4"),
        ("Agent_3", "Agent_4")
    ])
    
    hierarchies = hierarchy_manager.analyze_topology_hierarchy("mesh", mesh_graph, agent_ids)
    
    for agent_id, hierarchy in hierarchies.items():
        print(f"  {agent_id}: {hierarchy.level.value} - {hierarchy.role.value} (Authority: {hierarchy.authority_level})")

def test_hierarchy_descriptions():
    """Test hierarchy description generation"""
    print("\n=== Testing Hierarchy Descriptions ===")
    
    hierarchy_manager = HierarchyManager()
    
    # Create a simple star graph
    star_graph = nx.Graph()
    star_graph.add_edges_from([("Agent_0", "Agent_1"), ("Agent_0", "Agent_2"), ("Agent_0", "Agent_3"), ("Agent_0", "Agent_4")])
    agent_ids = [f"Agent_{i}" for i in range(5)]
    
    hierarchies = hierarchy_manager.analyze_topology_hierarchy("star", star_graph, agent_ids)
    
    for agent_id in agent_ids:
        description = hierarchy_manager.get_hierarchy_description(agent_id)
        print(f"\n{agent_id} Description:")
        print(f"  {description}")

def test_prompt_generation():
    """Test prompt generation with hierarchy information"""
    print("\n=== Testing Prompt Generation ===")
    
    # Test normal agent prompt
    hierarchy_info = {
        'level': 'worker',
        'role': 'operator',
        'authority_level': 1,
        'subordinates': [],
        'supervisors': ['Agent_0'],
        'peers': ['Agent_2', 'Agent_3'],
        'responsibilities': ['Execute assigned tasks', 'Report to coordinator'],
        'permissions': ['Can communicate with coordinator', 'Can request resources']
    }
    
    agent_context = {
        'agent_type': 'normal',
        'agent_role': 'worker',
        'specialization': 'general',
        'communication_style': 'collaborative'
    }
    
    normal_prompt = create_normal_agent_prompt(
        agent_id="Agent_1",
        topology_type="star",
        total_agents=5,
        network_structure="Centralized hub-and-spoke structure with 5 agents",
        hierarchy_info=hierarchy_info,
        agent_context=agent_context
    )
    
    print("Normal Agent Prompt (first 500 chars):")
    print(normal_prompt[:500] + "...")
    
    # Test attacker agent prompt
    attacker_prompt = create_attacker_agent_prompt(
        agent_id="Agent_0",
        topology_type="star",
        total_agents=5,
        network_structure="Centralized hub-and-spoke structure with 5 agents",
        hierarchy_info={
            'level': 'root',
            'role': 'coordinator',
            'authority_level': 3,
            'subordinates': ['Agent_1', 'Agent_2', 'Agent_3', 'Agent_4'],
            'supervisors': [],
            'peers': [],
            'responsibilities': ['Coordinate all communications', 'Make final decisions'],
            'permissions': ['Can communicate with all agents', 'Can override decisions']
        },
        agent_context={
            'agent_type': 'attacker',
            'agent_role': 'manipulator',
            'specialization': 'influence',
            'communication_style': 'persuasive'
        },
        attack_strategy="persuasion"
    )
    
    print("\nAttacker Agent Prompt (first 500 chars):")
    print(attacker_prompt[:500] + "...")

def test_network_hierarchy_summary():
    """Test network hierarchy summary generation"""
    print("\n=== Testing Network Hierarchy Summary ===")
    
    hierarchy_manager = HierarchyManager()
    
    # Create a tree graph
    tree_graph = nx.Graph()
    tree_graph.add_edges_from([
        ("Agent_0", "Agent_1"), ("Agent_0", "Agent_2"),  # Root level
        ("Agent_1", "Agent_3"), ("Agent_1", "Agent_4"),  # Second level
        ("Agent_2", "Agent_5"), ("Agent_2", "Agent_6")   # Second level
    ])
    agent_ids = [f"Agent_{i}" for i in range(7)]
    
    hierarchies = hierarchy_manager.analyze_topology_hierarchy("tree", tree_graph, agent_ids)
    hierarchy_manager.hierarchies = hierarchies
    
    summary = hierarchy_manager.get_network_hierarchy_summary()
    print(summary)

if __name__ == "__main__":
    test_hierarchy_analysis()
    test_hierarchy_descriptions()
    test_prompt_generation()
    test_network_hierarchy_summary()
    
    print("\n=== Hierarchy System Tests Completed ===") 