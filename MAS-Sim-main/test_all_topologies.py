"""
Test all topology types to ensure they work correctly
"""

import sys
import os
import numpy as np

# Add the parent directory to the path to import NewMA
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NewMA.network_topologies import (
    NetworkConfig, TopologyType, LinearPipelineTopology, TreeHierarchyTopology
)
from NewMA.advanced_topologies import (
    HolarchyTopology, P2PTopology, HybridTopology
)
from NewMA.graph_generator import AdvancedGraphGenerator


def test_linear_topology():
    """Test linear pipeline topology"""
    print("Testing Linear Pipeline Topology...")
    
    config = NetworkConfig(
        topology_type=TopologyType.LINEAR,
        num_agents=4,
        sparsity=0.2
    )
    
    system_prompts = [
        "Agent_0: You are a data processor.",
        "Agent_1: You are a feature extractor.",
        "Agent_2: You are a model trainer.",
        "Agent_3: You are a result evaluator."
    ]
    
    topology = LinearPipelineTopology(config)
    topology.create_agents(system_prompts, "gpt-4o-mini")
    topology.setup_connections()
    
    assert len(topology.agents) == 4
    assert len(topology.stages) == 4
    print("✓ Linear topology created successfully")
    
    # Test network stats
    stats = topology.get_network_stats()
    assert stats["num_agents"] == 4
    print("✓ Linear topology stats calculated successfully")
    
    return True


def test_tree_hierarchy_topology():
    """Test tree hierarchy topology"""
    print("Testing Tree Hierarchy Topology...")
    
    config = NetworkConfig(
        topology_type=TopologyType.TREE_HIERARCHY,
        num_agents=6,
        sparsity=0.3,
        max_depth=2,
        branching_factor=2
    )
    
    system_prompts = [
        "Manager: You are a project manager.",
        "Worker_1: You are a data analyst.",
        "Worker_2: You are a ML engineer.",
        "Worker_3: You are a software engineer.",
        "Worker_4: You are a QA specialist.",
        "Worker_5: You are a UX designer."
    ]
    
    topology = TreeHierarchyTopology(config)
    topology.create_agents(system_prompts, "gpt-4o-mini")
    topology.setup_connections()
    
    # The actual number of agents may be less than requested due to hierarchy constraints
    assert len(topology.agents) > 0
    assert len(topology.managers) > 0
    assert len(topology.workers) > 0
    print("✓ Tree hierarchy topology created successfully")
    
    # Test network stats
    stats = topology.get_network_stats()
    assert stats["num_agents"] > 0
    print("✓ Tree hierarchy topology stats calculated successfully")
    
    return True


def test_holarchy_topology():
    """Test holarchy topology"""
    print("Testing Holarchy Topology...")
    
    config = NetworkConfig(
        topology_type=TopologyType.HOLARCHY,
        num_agents=5,
        sparsity=0.4
    )
    
    system_prompts = [
        "Super_Holon: You are a super-holon.",
        "Sub_Holon_1: You are a sub-holon for decisions.",
        "Sub_Holon_2: You are a sub-holon for cooperation.",
        "Sub_Holon_3: You are a sub-holon for resources.",
        "Sub_Holon_4: You are a sub-holon for communication."
    ]
    
    topology = HolarchyTopology(config)
    topology.create_agents(system_prompts, "gpt-4o-mini")
    topology.setup_connections()
    
    assert len(topology.agents) == 5
    assert len(topology.holons) == 5
    print("✓ Holarchy topology created successfully")
    
    # Test network stats
    stats = topology.get_network_stats()
    assert stats["num_agents"] == 5
    print("✓ Holarchy topology stats calculated successfully")
    
    return True


def test_p2p_topology():
    """Test P2P topology"""
    print("Testing P2P Topology...")
    
    config = NetworkConfig(
        topology_type=TopologyType.P2P_FLAT,
        num_agents=6,
        sparsity=0.3,
        p2p_connection_type="mesh"
    )
    
    system_prompts = [
        "Peer_1: You are a peer for storage.",
        "Peer_2: You are a peer for computation.",
        "Peer_3: You are a peer for routing.",
        "Peer_4: You are a peer for security.",
        "Peer_5: You are a peer for authentication.",
        "Peer_6: You are a peer for delivery."
    ]
    
    topology = P2PTopology(config)
    topology.create_agents(system_prompts, "gpt-4o-mini")
    topology.setup_connections()
    
    assert len(topology.agents) == 6
    assert len(topology.peers) == 6
    print("✓ P2P topology created successfully")
    
    # Test network stats
    stats = topology.get_network_stats()
    assert stats["num_agents"] == 6
    print("✓ P2P topology stats calculated successfully")
    
    return True


def test_hybrid_topology():
    """Test hybrid topology"""
    print("Testing Hybrid Topology...")
    
    config = NetworkConfig(
        topology_type=TopologyType.HYBRID,
        num_agents=8,
        sparsity=0.25,
        hybrid_centralization_ratio=0.3
    )
    
    system_prompts = [
        "Coordinator: You are a hybrid coordinator.",
        "Centralized_1: You are a centralized worker.",
        "Centralized_2: You are a centralized worker.",
        "Centralized_3: You are a centralized worker.",
        "P2P_1: You are a P2P peer.",
        "P2P_2: You are a P2P peer.",
        "P2P_3: You are a P2P peer.",
        "P2P_4: You are a P2P peer."
    ]
    
    topology = HybridTopology(config)
    topology.create_agents(system_prompts, "gpt-4o-mini")
    topology.setup_connections()
    
    assert len(topology.agents) == 8
    assert len(topology.coordinators) > 0
    print("✓ Hybrid topology created successfully")
    
    # Test network stats
    stats = topology.get_network_stats()
    assert stats["num_agents"] == 8
    print("✓ Hybrid topology stats calculated successfully")
    
    return True


def test_graph_generator_all_topologies():
    """Test graph generator with all topology types"""
    print("Testing Graph Generator with All Topologies...")
    
    generator = AdvancedGraphGenerator("gpt-4o-mini")
    
    topology_types = [
        TopologyType.LINEAR,
        TopologyType.TREE_HIERARCHY,
        TopologyType.HOLARCHY,
        TopologyType.P2P_FLAT,
        TopologyType.HYBRID
    ]
    
    for topology_type in topology_types:
        try:
            dataset = generator.generate_network_dataset(
                topology_type=topology_type,
                num_agents=4,
                num_networks=1,
                sparsity=0.2,
                attacker_ratio=0.2
            )
            
            print(f"✓ {topology_type.value} topology dataset generated successfully")
            
            # Verify dataset structure
            if dataset and len(dataset) > 0:
                network_data = dataset[0]
                assert "topology_type" in network_data
                assert "num_agents" in network_data
                assert "adjacency_matrix" in network_data
                assert "network_stats" in network_data
                print(f"✓ {topology_type.value} topology dataset structure verified")
            else:
                print(f"✗ {topology_type.value} topology dataset is empty")
                return False
                
        except Exception as e:
            print(f"✗ Error generating {topology_type.value} topology dataset: {e}")
            return False
    
    return True


def main():
    """Run all topology tests"""
    print("=" * 60)
    print("Testing All Topology Types")
    print("=" * 60)
    
    tests = [
        test_linear_topology,
        test_tree_hierarchy_topology,
        test_holarchy_topology,
        test_p2p_topology,
        test_hybrid_topology,
        test_graph_generator_all_topologies
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All topology tests passed successfully!")
        print("The NewMA system is working correctly with all topology types.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 