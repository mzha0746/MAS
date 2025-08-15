"""
Debug Tree Hierarchy Topology
"""

import sys
import os
import traceback

# Add the parent directory to the path to import NewMA
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NewMA.network_topologies import (
    NetworkConfig, TopologyType, TreeHierarchyTopology
)


def debug_tree_hierarchy():
    """Debug tree hierarchy topology creation"""
    print("Debugging Tree Hierarchy Topology...")
    
    try:
        config = NetworkConfig(
            topology_type=TopologyType.TREE_HIERARCHY,
            num_agents=6,
            sparsity=0.3,
            max_depth=2,
            branching_factor=2
        )
        
        print("✓ Config created successfully")
        
        system_prompts = [
            "Manager: You are a project manager.",
            "Worker_1: You are a data analyst.",
            "Worker_2: You are a ML engineer.",
            "Worker_3: You are a software engineer.",
            "Worker_4: You are a QA specialist.",
            "Worker_5: You are a UX designer."
        ]
        
        print("✓ System prompts created")
        
        topology = TreeHierarchyTopology(config)
        print("✓ Topology instance created")
        
        topology.create_agents(system_prompts, "gpt-4o-mini")
        print("✓ Agents created")
        print(f"  - Total agents: {len(topology.agents)}")
        print(f"  - Managers: {len(topology.managers)}")
        print(f"  - Workers: {len(topology.workers)}")
        print(f"  - Hierarchy levels: {len(topology.hierarchy_levels)}")
        
        for level, agents in topology.hierarchy_levels.items():
            print(f"  - Level {level}: {len(agents)} agents")
            for agent in agents:
                print(f"    - {agent.agent_id}: {type(agent).__name__}")
        
        topology.setup_connections()
        print("✓ Connections setup completed")
        
        # Test network stats
        stats = topology.get_network_stats()
        print("✓ Network stats calculated")
        print(f"  - Num agents: {stats['num_agents']}")
        print(f"  - Num edges: {stats['num_edges']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = debug_tree_hierarchy()
    sys.exit(0 if success else 1) 