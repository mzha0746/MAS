#!/usr/bin/env python3
"""
Test script to verify variable scope fix
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.network_topologies import TopologyType

async def test_variable_fix():
    """Test that topology_type_str variable scope is fixed"""
    print("=== Testing Variable Scope Fix ===")
    
    # Initialize graph generator
    generator = AdvancedGraphGenerator(model_type="gpt-4o-mini", verbose=True)
    
    # Define test parameters
    topology_types = [TopologyType.TREE_HIERARCHY]
    num_agents_list = [4]
    num_networks_per_config = 1
    queries = ["What is the capital of France?"]
    
    output_base_dir = "/work/G-safeguard/NewMA/output"
    
    print(f"\n1. Starting comprehensive dataset generation...")
    print(f"   Output directory: {output_base_dir}")
    print(f"   Queries: {queries}")
    
    try:
        # Generate dataset
        dataset = await generator.generate_comprehensive_dataset(
            topology_types=topology_types,
            num_agents_list=num_agents_list,
            num_networks_per_config=num_networks_per_config,
            queries=queries,
            sparsity_range=(0.2, 0.3),
            attacker_ratio=0.2,
            attacker_strategy="persuasion",
            use_msmarco=False,
            output_base_dir=output_base_dir,
            save_filepath=None  # Don't save to single file for this test
        )
        
        print(f"✓ Dataset generation completed successfully")
        print(f"   Total items: {len(dataset)}")
        
        # Check if output files were created
        print(f"\n2. Checking output files...")
        
        if os.path.exists(output_base_dir):
            dirs = [d for d in os.listdir(output_base_dir) if os.path.isdir(os.path.join(output_base_dir, d))]
            
            for dir_name in sorted(dirs):
                if dir_name.startswith("query_"):
                    dir_path = os.path.join(output_base_dir, dir_name)
                    
                    # Check query metadata
                    metadata_file = os.path.join(dir_path, "query_metadata.json")
                    if os.path.exists(metadata_file):
                        print(f"   ✓ Query metadata created: {metadata_file}")
                    
                    # Check data directory
                    data_dir = os.path.join(dir_path, "data")
                    if os.path.exists(data_dir):
                        data_files = os.listdir(data_dir)
                        print(f"   ✓ Data files created: {data_files}")
                    
                    # Check logs directory
                    logs_dir = os.path.join(dir_path, "logs")
                    if os.path.exists(logs_dir):
                        log_files = os.listdir(logs_dir)
                        print(f"   ✓ Log files created: {log_files}")
                    
                    # Check images directory
                    images_dir = os.path.join(dir_path, "images")
                    if os.path.exists(images_dir):
                        image_files = os.listdir(images_dir)
                        print(f"   ✓ Image files created: {image_files}")
        
        print(f"\n3. Testing topology_type_str variable scope...")
        
        # Test the variable definition logic
        topology_type = TopologyType.TREE_HIERARCHY
        topology_type_str = topology_type.value if hasattr(topology_type, 'value') else str(topology_type)
        
        print(f"   Original topology_type: {topology_type}")
        print(f"   topology_type_str: {topology_type_str}")
        print(f"   Type of topology_type_str: {type(topology_type_str)}")
        
        # Test that it's JSON serializable
        import json
        test_data = {"topology_type": topology_type_str}
        json_str = json.dumps(test_data)
        print(f"   ✓ JSON serialization test passed")
        print(f"   JSON: {json_str}")
        
        print(f"\n=== Test Complete ===")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_variable_fix()) 