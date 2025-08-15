#!/usr/bin/env python3
"""
Test script to verify neighbor responses fix
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.network_topologies import TopologyType

async def test_neighbor_fix():
    """Test that neighbor responses are correctly identified"""
    print("=== Testing Neighbor Responses Fix ===")
    
    # Initialize graph generator
    generator = AdvancedGraphGenerator(model_type="gpt-4o-mini", verbose=True)
    
    # Generate a simple network dataset with tree hierarchy
    print("\n1. Generating network dataset...")
    network_datasets = generator.generate_network_dataset(
        topology_type=TopologyType.TREE_HIERARCHY,
        num_agents=5,
        num_networks=1,
        sparsity=0.3,
        attacker_ratio=0.0  # No attackers for simplicity
    )
    
    if not network_datasets:
        print("Error: No network datasets generated")
        return
    
    network_data = network_datasets[0]
    print(f"Generated network with {network_data['num_agents']} agents")
    
    # Print adjacency matrix for debugging
    print("\n2. Adjacency Matrix:")
    adjacency_matrix = network_data["adjacency_matrix"]
    agent_ids = list(network_data["agent_roles"].keys())
    
    print("Agent IDs:", agent_ids)
    print("Adjacency Matrix:")
    for i, row in enumerate(adjacency_matrix):
        print(f"  {agent_ids[i]}: {row}")
    
    # Test _get_neighbor_responses for each agent
    print("\n3. Testing neighbor responses for each agent:")
    
    # Create mock agent responses
    mock_responses = {}
    for agent_id in agent_ids:
        mock_responses[agent_id] = {
            "answer": f"Response from {agent_id}",
            "confidence": "HIGH",
            "reasoning": f"Reasoning from {agent_id}",
            "agent_role": "normal",
            "round": 0,
            "agent_permissions": {"authority_level": 1}
        }
    
    # Test for each agent
    for agent_id in agent_ids:
        print(f"\n  Testing agent {agent_id}:")
        
        # Get neighbors using the fixed method
        neighbors = generator._get_neighbor_responses(agent_id, mock_responses, network_data)
        
        print(f"    Incoming neighbors: {len(neighbors)}")
        for neighbor in neighbors:
            print(f"      - {neighbor}")
        
        # Also check outgoing connections for comparison
        agent_index = agent_ids.index(agent_id)
        outgoing = []
        for i, connected in enumerate(adjacency_matrix[agent_index]):
            if connected == 1 and i != agent_index:
                outgoing.append(agent_ids[i])
        
        print(f"    Outgoing connections: {outgoing}")
        
        # Verify that incoming neighbors are different from outgoing
        incoming_ids = [n.split(' (')[0] for n in neighbors]
        print(f"    Incoming IDs: {incoming_ids}")
        print(f"    Outgoing IDs: {outgoing}")
        
        # Check if there's overlap (should be minimal in tree hierarchy)
        overlap = set(incoming_ids) & set(outgoing)
        print(f"    Overlap: {overlap}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_neighbor_fix()) 