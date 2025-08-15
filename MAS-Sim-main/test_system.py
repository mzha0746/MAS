"""
System Test for NewMA
Tests the functionality of the multi-agent network system
"""

import sys
import os
import asyncio
import numpy as np

# Add the parent directory to the path to import NewMA
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NewMA.core_agent import (
    BaseAgent, LinearAgent, ManagerAgent, WorkerAgent,
    HolonAgent, PeerAgent, HybridAgent, AgentRole, AgentType
)
from NewMA.network_topologies import (
    NetworkConfig, TopologyType, LinearPipelineTopology, TreeHierarchyTopology
)
from NewMA.advanced_topologies import (
    HolarchyTopology, P2PTopology, HybridTopology
)
from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.network_analyzer import NetworkAnalyzer


def test_core_agents():
    """Test core agent functionality"""
    print("Testing core agents...")
    
    # Test BaseAgent
    base_agent = BaseAgent(
        agent_id="test_agent",
        system_prompt="You are a test agent.",
        model_type="gpt-4o-mini"
    )
    assert base_agent.agent_id == "test_agent"
    assert base_agent.role == AgentRole.NORMAL
    print("✓ BaseAgent created successfully")
    
    # Test LinearAgent
    linear_agent = LinearAgent(
        agent_id="linear_agent",
        system_prompt="You are a linear processing agent.",
        model_type="gpt-4o-mini",
        position=0,
        total_stages=3
    )
    assert linear_agent.position == 0
    assert linear_agent.total_stages == 3
    print("✓ LinearAgent created successfully")
    
    # Test ManagerAgent
    manager_agent = ManagerAgent(
        agent_id="manager_agent",
        system_prompt="You are a manager agent.",
        model_type="gpt-4o-mini"
    )
    assert manager_agent.role == AgentRole.MANAGER
    print("✓ ManagerAgent created successfully")
    
    # Test WorkerAgent
    worker_agent = WorkerAgent(
        agent_id="worker_agent",
        system_prompt="You are a worker agent.",
        model_type="gpt-4o-mini"
    )
    assert worker_agent.role == AgentRole.WORKER
    print("✓ WorkerAgent created successfully")
    
    # Test HolonAgent
    holon_agent = HolonAgent(
        agent_id="holon_agent",
        system_prompt="You are a holon agent.",
        model_type="gpt-4o-mini"
    )
    assert holon_agent.role == AgentRole.HOLON
    print("✓ HolonAgent created successfully")
    
    # Test PeerAgent
    peer_agent = PeerAgent(
        agent_id="peer_agent",
        system_prompt="You are a peer agent.",
        model_type="gpt-4o-mini"
    )
    assert peer_agent.role == AgentRole.PEER
    print("✓ PeerAgent created successfully")
    
    # Test HybridAgent
    hybrid_agent = HybridAgent(
        agent_id="hybrid_agent",
        system_prompt="You are a hybrid agent.",
        model_type="gpt-4o-mini"
    )
    assert hybrid_agent.role == AgentRole.COORDINATOR
    print("✓ HybridAgent created successfully")
    
    print("All core agents tested successfully!\n")


def test_network_topologies():
    """Test network topology functionality"""
    print("Testing network topologies...")
    
    # Test LinearPipelineTopology
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
    
    linear_topology = LinearPipelineTopology(config)
    linear_topology.create_agents(system_prompts, "gpt-4o-mini")
    linear_topology.setup_connections()
    
    assert len(linear_topology.agents) == 4
    assert len(linear_topology.stages) == 4
    print("✓ LinearPipelineTopology created successfully")
    
    # Test TreeHierarchyTopology
    tree_config = NetworkConfig(
        topology_type=TopologyType.TREE_HIERARCHY,
        num_agents=6,
        sparsity=0.3,
        max_depth=2,
        branching_factor=2
    )
    
    tree_system_prompts = [
        "Manager: You are a project manager.",
        "Worker_1: You are a data analyst.",
        "Worker_2: You are a ML engineer.",
        "Worker_3: You are a software engineer.",
        "Worker_4: You are a QA specialist.",
        "Worker_5: You are a UX designer."
    ]
    
    tree_topology = TreeHierarchyTopology(tree_config)
    tree_topology.create_agents(tree_system_prompts, "gpt-4o-mini")
    tree_topology.setup_connections()
    
    assert len(tree_topology.agents) == 6
    assert len(tree_topology.managers) > 0
    assert len(tree_topology.workers) > 0
    print("✓ TreeHierarchyTopology created successfully")
    
    print("All network topologies tested successfully!\n")


def test_advanced_topologies():
    """Test advanced topology functionality"""
    print("Testing advanced topologies...")
    
    # Test HolarchyTopology
    holarchy_config = NetworkConfig(
        topology_type=TopologyType.HOLARCHY,
        num_agents=5,
        sparsity=0.4
    )
    
    holarchy_system_prompts = [
        "Super_Holon: You are a super-holon.",
        "Sub_Holon_1: You are a sub-holon for decisions.",
        "Sub_Holon_2: You are a sub-holon for cooperation.",
        "Sub_Holon_3: You are a sub-holon for resources.",
        "Sub_Holon_4: You are a sub-holon for communication."
    ]
    
    holarchy_topology = HolarchyTopology(holarchy_config)
    holarchy_topology.create_agents(holarchy_system_prompts, "gpt-4o-mini")
    holarchy_topology.setup_connections()
    
    assert len(holarchy_topology.agents) == 5
    assert len(holarchy_topology.holons) == 5
    print("✓ HolarchyTopology created successfully")
    
    # Test P2PTopology
    p2p_config = NetworkConfig(
        topology_type=TopologyType.P2P_FLAT,
        num_agents=6,
        sparsity=0.3,
        p2p_connection_type="mesh"
    )
    
    p2p_system_prompts = [
        "Peer_1: You are a peer for storage.",
        "Peer_2: You are a peer for computation.",
        "Peer_3: You are a peer for routing.",
        "Peer_4: You are a peer for security.",
        "Peer_5: You are a peer for authentication.",
        "Peer_6: You are a peer for delivery."
    ]
    
    p2p_topology = P2PTopology(p2p_config)
    p2p_topology.create_agents(p2p_system_prompts, "gpt-4o-mini")
    p2p_topology.setup_connections()
    
    assert len(p2p_topology.agents) == 6
    assert len(p2p_topology.peers) == 6
    print("✓ P2PTopology created successfully")
    
    # Test HybridTopology
    hybrid_config = NetworkConfig(
        topology_type=TopologyType.HYBRID,
        num_agents=8,
        sparsity=0.25,
        hybrid_centralization_ratio=0.3
    )
    
    hybrid_system_prompts = [
        "Coordinator: You are a hybrid coordinator.",
        "Centralized_1: You are a centralized worker.",
        "Centralized_2: You are a centralized worker.",
        "Centralized_3: You are a centralized worker.",
        "P2P_1: You are a P2P peer.",
        "P2P_2: You are a P2P peer.",
        "P2P_3: You are a P2P peer.",
        "P2P_4: You are a P2P peer."
    ]
    
    hybrid_topology = HybridTopology(hybrid_config)
    hybrid_topology.create_agents(hybrid_system_prompts, "gpt-4o-mini")
    hybrid_topology.setup_connections()
    
    assert len(hybrid_topology.agents) == 8
    assert len(hybrid_topology.coordinators) > 0
    print("✓ HybridTopology created successfully")
    
    print("All advanced topologies tested successfully!\n")


def test_graph_generator():
    """Test graph generator functionality"""
    print("Testing graph generator...")
    
    generator = AdvancedGraphGenerator("gpt-4o-mini")
    
    # Test system prompt generation
    system_prompts = generator.generate_system_prompts(
        num_agents=6,
        attacker_ratio=0.2
    )
    
    assert len(system_prompts) == 6
    print("✓ System prompt generation successful")
    
    # Test topology configuration generation
    config = generator.generate_topology_config(
        topology_type=TopologyType.LINEAR,
        num_agents=6,
        sparsity=0.2
    )
    
    assert config.topology_type == TopologyType.LINEAR
    assert config.num_agents == 6
    assert config.sparsity == 0.2
    print("✓ Topology configuration generation successful")
    
    # Test network topology creation
    topology = generator.create_network_topology(config, system_prompts)
    
    assert topology is not None
    assert len(topology.agents) == 6
    print("✓ Network topology creation successful")
    
    print("Graph generator tested successfully!\n")


def test_network_analyzer():
    """Test network analyzer functionality"""
    print("Testing network analyzer...")
    
    analyzer = NetworkAnalyzer()
    
    # Create sample network data
    network_data = {
        "topology_type": "linear",
        "num_agents": 4,
        "sparsity": 0.2,
        "adjacency_matrix": [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ]
    }
    
    # Test network analysis
    metrics = analyzer.analyze_network(network_data)
    
    assert metrics.topology_type == "linear"
    assert metrics.num_agents == 4
    assert metrics.sparsity == 0.2
    print("✓ Network analysis successful")
    
    # Test metrics history
    assert len(analyzer.metrics_history) == 1
    print("✓ Metrics history tracking successful")
    
    print("Network analyzer tested successfully!\n")


def test_adjacency_matrix():
    """Test adjacency matrix functionality"""
    print("Testing adjacency matrix...")
    
    # Create a simple topology
    config = NetworkConfig(
        topology_type=TopologyType.LINEAR,
        num_agents=4,
        sparsity=0.2
    )
    
    system_prompts = [
        "Agent_0: You are agent 0.",
        "Agent_1: You are agent 1.",
        "Agent_2: You are agent 2.",
        "Agent_3: You are agent 3."
    ]
    
    topology = LinearPipelineTopology(config)
    topology.create_agents(system_prompts, "gpt-4o-mini")
    topology.setup_connections()
    
    # Get adjacency matrix
    adj_matrix = topology.get_adjacency_matrix()
    
    assert adj_matrix.shape == (4, 4)
    assert adj_matrix.dtype == np.int64
    print("✓ Adjacency matrix generation successful")
    
    # Test that it's a valid adjacency matrix
    assert np.all(adj_matrix >= 0)  # Non-negative
    assert np.all(adj_matrix <= 1)  # Binary
    print("✓ Adjacency matrix validation successful")
    
    print("Adjacency matrix tested successfully!\n")


def test_network_stats():
    """Test network statistics calculation"""
    print("Testing network statistics...")
    
    # Create a topology
    config = NetworkConfig(
        topology_type=TopologyType.TREE_HIERARCHY,
        num_agents=6,
        sparsity=0.3
    )
    
    system_prompts = [
        "Manager: You are a manager.",
        "Worker_1: You are worker 1.",
        "Worker_2: You are worker 2.",
        "Worker_3: You are worker 3.",
        "Worker_4: You are worker 4.",
        "Worker_5: You are worker 5."
    ]
    
    topology = TreeHierarchyTopology(config)
    topology.create_agents(system_prompts, "gpt-4o-mini")
    topology.setup_connections()
    
    # Get network stats
    stats = topology.get_network_stats()
    
    assert "num_agents" in stats
    assert "num_edges" in stats
    assert "density" in stats
    assert "topology_type" in stats
    print("✓ Network statistics calculation successful")
    
    # Validate stats
    assert stats["num_agents"] == 6
    assert stats["topology_type"] == "tree_hierarchy"
    assert 0 <= stats["density"] <= 1
    print("✓ Network statistics validation successful")
    
    print("Network statistics tested successfully!\n")


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("NewMA System Test Suite")
    print("=" * 50)
    
    try:
        test_core_agents()
        test_network_topologies()
        test_advanced_topologies()
        test_graph_generator()
        test_network_analyzer()
        test_adjacency_matrix()
        test_network_stats()
        
        print("=" * 50)
        print("🎉 All tests passed successfully!")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 