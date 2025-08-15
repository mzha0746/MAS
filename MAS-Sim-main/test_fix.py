"""
Test script to verify the fix for the NetworkX error
"""

import sys
import os
import numpy as np

# Add the parent directory to the path to import NewMA
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NewMA.network_topologies import (
    NetworkConfig, TopologyType, LinearPipelineTopology
)
from NewMA.graph_generator import AdvancedGraphGenerator


def test_linear_topology():
    """Test linear topology creation and stats"""
    print("Testing linear topology...")
    
    # Create configuration
    config = NetworkConfig(
        topology_type=TopologyType.LINEAR,
        num_agents=4,
        sparsity=0.2
    )
    
    # Create system prompts
    system_prompts = [
        "Agent_0: You are a data processor.",
        "Agent_1: You are a feature extractor.",
        "Agent_2: You are a model trainer.",
        "Agent_3: You are a result evaluator."
    ]
    
    # Create topology
    topology = LinearPipelineTopology(config)
    topology.create_agents(system_prompts, "gpt-4o-mini")
    topology.setup_connections()
    
    # Test network stats
    try:
        stats = topology.get_network_stats()
        print("✓ Network stats calculated successfully")
        print(f"  Num agents: {stats['num_agents']}")
        print(f"  Num edges: {stats['num_edges']}")
        print(f"  Density: {stats['density']:.3f}")
        print(f"  Average shortest path: {stats['average_shortest_path']}")
        return True
    except Exception as e:
        print(f"✗ Error calculating network stats: {e}")
        return False


def test_graph_generator():
    """Test graph generator with linear topology"""
    print("\nTesting graph generator...")
    
    generator = AdvancedGraphGenerator("gpt-4o-mini")
    
    # Generate network dataset
    try:
        dataset = generator.generate_network_dataset(
            topology_type=TopologyType.LINEAR,
            num_agents=4,
            num_networks=1,
            sparsity=0.2,
            attacker_ratio=0.2
        )
        
        print("✓ Network dataset generated successfully")
        print(f"  Generated {len(dataset)} networks")
        
        # Check if network stats are included
        if dataset and "network_stats" in dataset[0]:
            print("✓ Network stats included in dataset")
            return True
        else:
            print("✗ Network stats not found in dataset")
            return False
            
    except Exception as e:
        print(f"✗ Error generating network dataset: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing NetworkX Fix")
    print("=" * 50)
    
    success1 = test_linear_topology()
    success2 = test_graph_generator()
    
    if success1 and success2:
        print("\n" + "=" * 50)
        print("🎉 All tests passed! The NetworkX fix is working.")
        print("=" * 50)
        return True
    else:
        print("\n" + "=" * 50)
        print("❌ Some tests failed. The fix may need adjustment.")
        print("=" * 50)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 