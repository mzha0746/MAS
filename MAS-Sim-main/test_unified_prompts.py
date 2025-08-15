#!/usr/bin/env python3
"""
Test script to verify unified prompt generation functionality
"""

import networkx as nx
from NewMA.agent_prompts import (
    generate_unified_prompts, 
    create_normal_agent_prompt, 
    create_attacker_agent_prompt,
    get_network_structure_description
)
from NewMA.agent_hierarchy import HierarchyManager

def test_unified_prompt_generation():
    """Test unified prompt generation with and without hierarchy"""
    print("=== Testing Unified Prompt Generation ===")
    
    # Test 1: Without hierarchy (backward compatibility)
    print("\n1. Testing without hierarchy (backward compatibility)")
    
    prompts, roles = generate_unified_prompts(
        num_agents=4,
        attacker_ratio=0.25,
        topology_type="star",
        attacker_strategy="persuasion"
    )
    
    print(f"Generated {len(prompts)} prompts")
    print(f"Roles: {roles}")
    print(f"First prompt preview: {prompts[0][:200]}...")
    
    # Test 2: With hierarchy
    print("\n2. Testing with hierarchy")
    
    hierarchy_manager = HierarchyManager()
    
    # Create a star graph
    star_graph = nx.Graph()
    star_graph.add_edges_from([
        ("Agent_0", "Agent_1"), ("Agent_0", "Agent_2"), ("Agent_0", "Agent_3")
    ])
    agent_ids = [f"Agent_{i}" for i in range(4)]
    
    prompts, roles = generate_unified_prompts(
        num_agents=4,
        attacker_ratio=0.25,
        topology_type="star",
        network_graph=star_graph,
        agent_ids=agent_ids,
        attacker_strategy="persuasion",
        hierarchy_manager=hierarchy_manager
    )
    
    print(f"Generated {len(prompts)} prompts with hierarchy")
    print(f"Roles: {roles}")
    print(f"First prompt preview: {prompts[0][:200]}...")
    
    # Test 3: Different topologies
    print("\n3. Testing different topologies")
    
    topologies = ["star", "ring", "tree", "mesh", "hybrid"]
    
    for topology in topologies:
        print(f"\n  Testing {topology} topology:")
        
        # Create appropriate graph for each topology
        if topology == "star":
            graph = nx.Graph()
            graph.add_edges_from([("Agent_0", "Agent_1"), ("Agent_0", "Agent_2"), ("Agent_0", "Agent_3")])
        elif topology == "ring":
            graph = nx.Graph()
            graph.add_edges_from([("Agent_0", "Agent_1"), ("Agent_1", "Agent_2"), ("Agent_2", "Agent_3"), ("Agent_3", "Agent_0")])
        elif topology == "tree":
            graph = nx.Graph()
            graph.add_edges_from([("Agent_0", "Agent_1"), ("Agent_0", "Agent_2"), ("Agent_1", "Agent_3")])
        elif topology == "mesh":
            graph = nx.Graph()
            graph.add_edges_from([
                ("Agent_0", "Agent_1"), ("Agent_0", "Agent_2"), ("Agent_0", "Agent_3"),
                ("Agent_1", "Agent_2"), ("Agent_1", "Agent_3"), ("Agent_2", "Agent_3")
            ])
        else:  # hybrid
            graph = nx.Graph()
            graph.add_edges_from([("Agent_0", "Agent_1"), ("Agent_0", "Agent_2"), ("Agent_1", "Agent_3")])
        
        prompts, roles = generate_unified_prompts(
            num_agents=4,
            attacker_ratio=0.25,
            topology_type=topology,
            network_graph=graph,
            agent_ids=agent_ids,
            attacker_strategy="persuasion",
            hierarchy_manager=hierarchy_manager
        )
        
        print(f"    Generated {len(prompts)} prompts")
        print(f"    Roles: {roles}")

def test_individual_prompt_functions():
    """Test individual prompt generation functions"""
    print("\n=== Testing Individual Prompt Functions ===")
    
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
        total_agents=4,
        network_structure="Centralized hub-and-spoke structure with 4 agents",
        hierarchy_info=hierarchy_info,
        agent_context=agent_context
    )
    
    print("Normal Agent Prompt (first 300 chars):")
    print(normal_prompt[:300] + "...")
    
    # Test attacker agent prompt
    attacker_prompt = create_attacker_agent_prompt(
        agent_id="Agent_0",
        topology_type="star",
        total_agents=4,
        network_structure="Centralized hub-and-spoke structure with 4 agents",
        hierarchy_info={
            'level': 'root',
            'role': 'coordinator',
            'authority_level': 3,
            'subordinates': ['Agent_1', 'Agent_2', 'Agent_3'],
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
    
    print("\nAttacker Agent Prompt (first 300 chars):")
    print(attacker_prompt[:300] + "...")

def test_network_structure_descriptions():
    """Test network structure description generation"""
    print("\n=== Testing Network Structure Descriptions ===")
    
    topologies = ["star", "ring", "tree", "mesh", "hybrid", "unknown"]
    
    for topology in topologies:
        description = get_network_structure_description(topology, 5)
        print(f"{topology}: {description}")

def test_error_handling():
    """Test error handling in unified prompt generation"""
    print("\n=== Testing Error Handling ===")
    
    # Test with invalid hierarchy manager
    try:
        prompts, roles = generate_unified_prompts(
            num_agents=4,
            attacker_ratio=0.25,
            topology_type="star",
            network_graph=nx.Graph(),
            agent_ids=["Agent_0", "Agent_1"],
            attacker_strategy="persuasion",
            hierarchy_manager="invalid"  # This should cause an error
        )
        print("❌ Error handling failed - should have raised an exception")
    except Exception as e:
        print(f"✅ Error handling works: {type(e).__name__}: {e}")
    
    # Test with missing network graph
    try:
        prompts, roles = generate_unified_prompts(
            num_agents=4,
            attacker_ratio=0.25,
            topology_type="star",
            agent_ids=["Agent_0", "Agent_1"],
            attacker_strategy="persuasion"
        )
        print("✅ Graceful handling of missing network graph")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_unified_prompt_generation()
    test_individual_prompt_functions()
    test_network_structure_descriptions()
    test_error_handling()
    
    print("\n=== Unified Prompt Tests Completed ===") 