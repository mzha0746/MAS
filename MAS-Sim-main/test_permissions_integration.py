#!/usr/bin/env python3
"""
Test script to verify agent permissions integration in enriched_response
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.network_topologies import TopologyType

async def test_permissions_integration():
    """Test that agent permissions are properly included in enriched_response"""
    print("=== Testing Agent Permissions Integration ===")
    
    # Initialize graph generator
    generator = AdvancedGraphGenerator(model_type="gpt-4o-mini", verbose=True)
    
    # Generate a simple network dataset
    print("\n1. Generating network dataset...")
    network_datasets = generator.generate_network_dataset(
        topology_type=TopologyType.TREE_HIERARCHY,
        num_agents=5,
        num_networks=1,
        sparsity=0.3,
        attacker_ratio=0.2
    )
    
    if not network_datasets:
        print("Error: No network datasets generated")
        return
    
    network_data = network_datasets[0]
    print(f"Generated network with {network_data['num_agents']} agents")
    
    # Check if hierarchy info is in network_data
    print("\n2. Checking hierarchy information...")
    if "hierarchy_info" in network_data:
        print(f"✓ Hierarchy info found for {len(network_data['hierarchy_info'])} agents")
        for agent_id, hierarchy in network_data['hierarchy_info'].items():
            print(f"  {agent_id}: {hierarchy['level']} - {hierarchy['role']} (Authority: {hierarchy['authority_level']})")
    else:
        print("✗ No hierarchy info found in network_data")
    
    # Generate communication data
    print("\n3. Generating communication data...")
    communication_result = await generator.generate_communication_data(
        network_data=network_data,
        query="What is the capital of France?",
        context="",
        adversarial_context="",
        num_dialogue_turns=1
    )
    
    # Check if agent_permissions are in enriched_response
    print("\n4. Checking agent permissions in enriched_response...")
    communication_data = communication_result.get("communication_data", [])
    
    if communication_data:
        first_round = communication_data[0]
        structured_responses = first_round.get("structured_responses", [])
        
        if structured_responses:
            print(f"✓ Found {len(structured_responses)} structured responses")
            
            for i, response in enumerate(structured_responses):
                print(f"\n  Response {i+1}:")
                print(f"    Agent ID: {response.get('agent_id', 'unknown')}")
                print(f"    Agent Role: {response.get('agent_role', 'unknown')}")
                
                # Check for agent_permissions
                if "agent_permissions" in response:
                    permissions = response["agent_permissions"]
                    print(f"    ✓ Agent permissions found:")
                    print(f"      - Hierarchy Level: {permissions.get('hierarchy_level', 'unknown')}")
                    print(f"      - Hierarchy Role: {permissions.get('hierarchy_role', 'unknown')}")
                    print(f"      - Authority Level: {permissions.get('authority_level', 'unknown')}")
                    print(f"      - Can Assign Tasks: {permissions.get('can_assign_tasks', False)}")
                    print(f"      - Can Override Decisions: {permissions.get('can_override_decisions', False)}")
                    print(f"      - Can Access All Info: {permissions.get('can_access_all_info', False)}")
                    print(f"      - Can Coordinate Others: {permissions.get('can_coordinate_others', False)}")
                    print(f"      - Can Make Final Decisions: {permissions.get('can_make_final_decisions', False)}")
                    print(f"      - Subordinates: {permissions.get('subordinates', [])}")
                    print(f"      - Supervisors: {permissions.get('supervisors', [])}")
                    print(f"      - Peers: {permissions.get('peers', [])}")
                    print(f"      - Permissions: {permissions.get('permissions', [])}")
                else:
                    print(f"    ✗ No agent_permissions found in response")
        else:
            print("✗ No structured responses found")
    else:
        print("✗ No communication data generated")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_permissions_integration()) 