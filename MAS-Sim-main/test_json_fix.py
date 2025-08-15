#!/usr/bin/env python3
"""
Test script to verify JSON serialization fixes
"""

import asyncio
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.network_topologies import TopologyType

async def test_json_fix():
    """Test that all JSON serialization issues are resolved"""
    print("=== Testing JSON Serialization Fixes ===")
    
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
        
        # Check output directory structure
        print(f"\n2. Checking output directory structure...")
        
        if os.path.exists(output_base_dir):
            print(f"✓ Output directory exists: {output_base_dir}")
            
            # List all directories
            dirs = [d for d in os.listdir(output_base_dir) if os.path.isdir(os.path.join(output_base_dir, d))]
            print(f"   Found {len(dirs)} directories:")
            
            for dir_name in sorted(dirs):
                dir_path = os.path.join(output_base_dir, dir_name)
                print(f"   - {dir_name}")
                
                # Check if it's a query directory
                if dir_name.startswith("query_"):
                    # Check query metadata
                    metadata_file = os.path.join(dir_path, "query_metadata.json")
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)
                            print(f"     ✓ Query metadata loaded successfully")
                            print(f"       Query: {metadata.get('query', 'unknown')}")
                            print(f"       Topology: {metadata.get('topology_type', 'unknown')}")
                            print(f"       Agents: {metadata.get('num_agents', 'unknown')}")
                        except Exception as e:
                            print(f"     ✗ Error reading metadata: {e}")
                    
                    # Check data directory
                    data_dir = os.path.join(dir_path, "data")
                    if os.path.exists(data_dir):
                        data_files = os.listdir(data_dir)
                        print(f"     ✓ Data directory: {data_files}")
                        
                        # Try to load communication data
                        for file in data_files:
                            if file.startswith("communication_data_"):
                                try:
                                    with open(os.path.join(data_dir, file), "r") as f:
                                        comm_data = json.load(f)
                                    print(f"       ✓ {file} loaded successfully")
                                except Exception as e:
                                    print(f"       ✗ Error loading {file}: {e}")
                    
                    # Check logs directory
                    logs_dir = os.path.join(dir_path, "logs")
                    if os.path.exists(logs_dir):
                        log_files = os.listdir(logs_dir)
                        print(f"     ✓ Logs directory: {log_files}")
                        
                        # Try to load log files
                        for file in log_files:
                            try:
                                with open(os.path.join(logs_dir, file), "r") as f:
                                    log_data = json.load(f)
                                print(f"       ✓ {file} loaded successfully")
                            except Exception as e:
                                print(f"       ✗ Error loading {file}: {e}")
                
                # Check if it's a network directory
                elif dir_name.startswith("network_"):
                    # Check network metadata
                    metadata_file = os.path.join(dir_path, "network_metadata.json")
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)
                            print(f"     ✓ Network metadata loaded successfully")
                            print(f"       Topology: {metadata.get('topology_type', 'unknown')}")
                            print(f"       Agents: {metadata.get('num_agents', 'unknown')}")
                        except Exception as e:
                            print(f"     ✗ Error reading network metadata: {e}")
        else:
            print(f"✗ Output directory not found: {output_base_dir}")
        
        print(f"\n3. Testing _convert_to_serializable method...")
        
        # Test with various data types
        test_data = {
            "topology_type": TopologyType.TREE_HIERARCHY,
            "enum_list": [TopologyType.LINEAR, TopologyType.HOLARCHY],
            "nested": {
                "enum_value": TopologyType.P2P_FLAT
            }
        }
        
        try:
            serializable_data = generator._convert_to_serializable(test_data)
            print(f"✓ _convert_to_serializable test passed")
            print(f"   Original topology_type type: {type(test_data['topology_type'])}")
            print(f"   Serialized topology_type type: {type(serializable_data['topology_type'])}")
            print(f"   Serialized value: {serializable_data['topology_type']}")
            
            # Test JSON serialization
            json_str = json.dumps(serializable_data, indent=2)
            print(f"✓ JSON serialization test passed")
            print(f"   JSON length: {len(json_str)} characters")
            
        except Exception as e:
            print(f"✗ _convert_to_serializable test failed: {e}")
        
        print(f"\n=== All Tests Complete ===")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_json_fix()) 