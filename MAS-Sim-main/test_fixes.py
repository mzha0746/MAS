#!/usr/bin/env python3
"""
Test script to verify hierarchy graph and JSON serialization fixes
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.network_topologies import TopologyType

async def test_fixes():
    """Test that hierarchy graph and JSON serialization work correctly"""
    print("=== Testing Hierarchy Graph and JSON Serialization Fixes ===")
    
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
    
    # Test JSON serialization
    print("\n2. Testing JSON serialization...")
    try:
        # Test the _convert_to_serializable method
        test_data = {
            "topology_type": TopologyType.TREE_HIERARCHY,
            "num_agents": 5,
            "nested": {
                "enum_value": TopologyType.LINEAR,
                "list_with_enums": [TopologyType.HOLARCHY, TopologyType.P2P_FLAT]
            }
        }
        
        serializable_data = generator._convert_to_serializable(test_data)
        print("✓ JSON serialization test passed")
        print(f"  Original topology_type: {type(test_data['topology_type'])}")
        print(f"  Serialized topology_type: {type(serializable_data['topology_type'])}")
        print(f"  Serialized value: {serializable_data['topology_type']}")
        
    except Exception as e:
        print(f"✗ JSON serialization test failed: {e}")
    
    # Test hierarchy info access
    print("\n3. Testing hierarchy info access...")
    if "hierarchy_info" in network_data:
        print(f"✓ Hierarchy info found for {len(network_data['hierarchy_info'])} agents")
        for agent_id, hierarchy in network_data['hierarchy_info'].items():
            print(f"  {agent_id}: {hierarchy['level']} - {hierarchy['role']} (Authority: {hierarchy['authority_level']})")
    else:
        print("✗ No hierarchy info found in network_data")
    
    # Test saving dataset
    print("\n4. Testing dataset saving...")
    try:
        test_save_path = "/work/G-safeguard/NewMA/output/test_dataset.json"
        os.makedirs("/work/G-safeguard/NewMA/output", exist_ok=True)
        
        # Create a test dataset with the network data
        test_dataset = [network_data]
        
        # Test saving
        generator._save_dataset_realtime(test_dataset, test_save_path, is_final=True)
        
        # Check if file was created
        if os.path.exists(test_save_path):
            print(f"✓ Dataset saved successfully to {test_save_path}")
            file_size = os.path.getsize(test_save_path)
            print(f"  File size: {file_size} bytes")
        else:
            print("✗ Dataset file was not created")
            
    except Exception as e:
        print(f"✗ Dataset saving test failed: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_fixes()) 