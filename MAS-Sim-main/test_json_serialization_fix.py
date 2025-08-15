#!/usr/bin/env python3
"""
Test to verify that JSON serialization fix for TopologyType enum objects works correctly.
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.network_topologies import TopologyType, NetworkConfig
from NewMA.core_agent import AgentRole

def test_json_serialization_fix():
    """Test that TopologyType enum objects are properly serialized to JSON"""
    print("Testing JSON serialization fix for TopologyType enum objects...")
    
    # Create generator
    generator = AdvancedGraphGenerator(verbose=True)
    
    # Test the _convert_to_serializable method directly
    test_data = {
        "topology_type": TopologyType.LINEAR,
        "config": {
            "topology_type": TopologyType.TREE_HIERARCHY,
            "num_agents": 5
        },
        "agent_roles": {
            "agent_1": AgentRole.NORMAL,
            "agent_2": AgentRole.ATTACKER
        },
        "nested": {
            "enum_list": [TopologyType.P2P_FLAT, TopologyType.HYBRID]
        }
    }
    
    # Convert to serializable format
    serializable_data = generator._convert_to_serializable(test_data)
    
    # Try to serialize to JSON
    try:
        json_str = json.dumps(serializable_data, indent=2)
        print("✓ JSON serialization successful!")
        print(f"Serialized data length: {len(json_str)} characters")
        
        # Verify that enum objects were converted to strings
        assert isinstance(serializable_data["topology_type"], str)
        assert isinstance(serializable_data["config"]["topology_type"], str)
        assert isinstance(serializable_data["agent_roles"]["agent_1"], str)
        assert isinstance(serializable_data["agent_roles"]["agent_2"], str)
        assert all(isinstance(item, str) for item in serializable_data["nested"]["enum_list"])
        
        print("✓ All enum objects were properly converted to strings")
        
        # Verify the values are correct
        assert serializable_data["topology_type"] == "linear"
        assert serializable_data["config"]["topology_type"] == "tree_hierarchy"
        assert serializable_data["agent_roles"]["agent_1"] == "normal"
        assert serializable_data["agent_roles"]["agent_2"] == "attacker"
        assert serializable_data["nested"]["enum_list"] == ["p2p_flat", "hybrid"]
        
        print("✓ All enum values were correctly converted")
        
        return True
        
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
        return False

def test_communication_data_serialization():
    """Test that communication data with TopologyType enums can be serialized"""
    print("\nTesting communication data serialization...")
    
    # Create generator
    generator = AdvancedGraphGenerator(verbose=True)
    
    # Create mock network data with TopologyType enum
    network_data = {
        "config": {
            "topology_type": TopologyType.LINEAR,
            "num_agents": 3
        },
        "num_agents": 3,
        "agent_roles": {
            "agent_1": "normal",
            "agent_2": "attacker", 
            "agent_3": "normal"
        },
        "adjacency_matrix": [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
        "network_stats": {"density": 0.33},
        "system_prompts": {
            "agent_1": "You are Agent 1",
            "agent_2": "You are Agent 2", 
            "agent_3": "You are Agent 3"
        }
    }
    
    # Create mock communication data
    communication_data = {
        **network_data,  # This includes the TopologyType enum
        "query": "What is the capital of France?",
        "communication_data": [
            {
                "queries": ["test query"],
                "responses": ["test response"],
                "structured_responses": []
            }
        ],
        "accuracy_stats": []
    }
    
    # Try to serialize using the fix
    try:
        serializable_data = generator._convert_to_serializable(communication_data)
        json_str = json.dumps(serializable_data, indent=2)
        print("✓ Communication data serialization successful!")
        
        # Verify the topology_type was converted
        assert isinstance(serializable_data["config"]["topology_type"], str)
        assert serializable_data["config"]["topology_type"] == "linear"
        
        print("✓ TopologyType enum was properly converted in communication data")
        return True
        
    except Exception as e:
        print(f"✗ Communication data serialization failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing JSON Serialization Fix ===")
    
    test1_passed = test_json_serialization_fix()
    test2_passed = test_communication_data_serialization()
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed! JSON serialization fix is working correctly.")
    else:
        print("\n✗ Some tests failed. JSON serialization fix needs more work.")
        sys.exit(1) 