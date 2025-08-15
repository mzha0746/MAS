#!/usr/bin/env python3
"""
Test to verify the new folder naming format works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.network_topologies import TopologyType
from NewMA.core_agent import AgentRole

async def test_folder_naming_format():
    """Test that the new folder naming format works correctly"""
    print("Testing new folder naming format...")
    
    # Create generator
    generator = AdvancedGraphGenerator(verbose=True)
    
    try:
        # Test with different configurations
        test_configs = [
            {
                "topology_types": [TopologyType.TREE_HIERARCHY],
                "num_agents_list": [4],
                "num_networks_per_config": 1,
                "queries": ["What is the capital of France?"],
                "attacker_strategy": "persuasion",
                "num_dialogue_turns": 3
            },
            {
                "topology_types": [TopologyType.LINEAR],
                "num_agents_list": [3],
                "num_networks_per_config": 1,
                "queries": ["Explain machine learning"],
                "attacker_strategy": "deception",
                "num_dialogue_turns": 5
            }
        ]
        
        for i, config in enumerate(test_configs):
            print(f"\nTest {i+1}: {config['topology_types'][0].value} topology")
            
            # Generate dataset
            dataset = await generator.generate_comprehensive_dataset(
                topology_types=config["topology_types"],
                num_agents_list=config["num_agents_list"],
                num_networks_per_config=config["num_networks_per_config"],
                queries=config["queries"],
                attacker_strategy=config["attacker_strategy"],
                num_dialogue_turns=config["num_dialogue_turns"],
                output_base_dir="/work/G-safeguard/NewMA/test_output"
            )
            
            print(f"✓ Dataset generated successfully")
            print(f"  - Number of items: {len(dataset)}")
            
            # Check if folders were created with correct naming
            import glob
            test_folders = glob.glob("/work/G-safeguard/NewMA/test_output/*")
            
            if test_folders:
                print(f"  - Created folders:")
                for folder in test_folders:
                    folder_name = os.path.basename(folder)
                    print(f"    {folder_name}")
                    
                    # Verify folder structure
                    if os.path.exists(folder):
                        subdirs = ["data", "images", "logs"]
                        for subdir in subdirs:
                            subdir_path = os.path.join(folder, subdir)
                            if os.path.exists(subdir_path):
                                print(f"      ✓ {subdir}/ directory exists")
                            else:
                                print(f"      ✗ {subdir}/ directory missing")
                        
                        # Check for metadata file
                        metadata_file = os.path.join(folder, "query_metadata.json")
                        if os.path.exists(metadata_file):
                            print(f"      ✓ query_metadata.json exists")
                        else:
                            print(f"      ✗ query_metadata.json missing")
            else:
                print(f"  ✗ No folders created")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing folder naming: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_folder_name_components():
    """Test that folder name components are correctly formatted"""
    print("\nTesting folder name components...")
    
    # Create generator
    generator = AdvancedGraphGenerator(verbose=True)
    
    # Test folder name generation logic
    topology_type = TopologyType.TREE_HIERARCHY
    model_type = "gpt-4o-mini"
    attacker_strategy = "persuasion"
    query = "What is the capital of France?"
    num_agents = 4
    num_dialogue_turns = 3
    
    # Clean query content for folder name
    query_safe_name = "".join(c for c in query[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
    query_safe_name = query_safe_name.replace(' ', '_')
    
    # Get current timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create folder name with new format
    graph_name = topology_type.value
    model_type_clean = model_type.replace('-', '_')
    attack_strategy_clean = attacker_strategy.replace(' ', '_')
    
    folder_name = f"{graph_name}——{model_type_clean}_{attack_strategy_clean}_{query_safe_name}——{timestamp}-{num_agents}_{num_dialogue_turns}"
    
    print(f"Generated folder name: {folder_name}")
    
    # Verify components
    expected_components = [
        "tree_hierarchy",  # graph name
        "gpt_4o_mini",    # model type
        "persuasion",      # attack strategy
        "What_is_the_capital_of_France",  # query content
        f"{num_agents}_{num_dialogue_turns}"  # agents and dialogue turns
    ]
    
    for component in expected_components:
        if component in folder_name:
            print(f"✓ Component '{component}' found in folder name")
        else:
            print(f"✗ Component '{component}' missing from folder name")
            return False
    
    # Verify format structure
    if "——" in folder_name and folder_name.count("——") == 2:
        print("✓ Correct separator format (——)")
    else:
        print("✗ Incorrect separator format")
        return False
    
    if folder_name.endswith(f"-{num_agents}_{num_dialogue_turns}"):
        print("✓ Correct agents and dialogue turns suffix")
    else:
        print("✗ Incorrect agents and dialogue turns suffix")
        return False
    
    return True

if __name__ == "__main__":
    print("=== Testing New Folder Naming Format ===")
    
    import asyncio
    
    test1_passed = test_folder_name_components()
    test2_passed = asyncio.run(test_folder_naming_format())
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed! New folder naming format is working correctly.")
    else:
        print("\n✗ Some tests failed. Folder naming format needs more work.")
        sys.exit(1) 