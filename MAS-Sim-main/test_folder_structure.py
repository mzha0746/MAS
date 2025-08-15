#!/usr/bin/env python3
"""
Test script to verify the new folder structure for query-based output
"""

import asyncio
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.network_topologies import TopologyType

async def test_folder_structure():
    """Test the new folder structure for query-based output"""
    print("=== Testing New Folder Structure ===")
    
    # Initialize graph generator
    generator = AdvancedGraphGenerator(model_type="gpt-4o-mini", verbose=True)
    
    # Define test parameters
    topology_types = [TopologyType.TREE_HIERARCHY]
    num_agents_list = [4]
    num_networks_per_config = 1
    queries = [
        "What is the capital of France?",
        "Explain the concept of machine learning."
    ]
    
    output_base_dir = "/work/G-safeguard/NewMA/output"
    
    print(f"\n1. Starting comprehensive dataset generation...")
    print(f"   Output directory: {output_base_dir}")
    print(f"   Queries: {queries}")
    
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
    
    print(f"\n2. Checking folder structure...")
    
    # Check if output directory exists
    if os.path.exists(output_base_dir):
        print(f"✓ Output directory created: {output_base_dir}")
        
        # List all directories
        dirs = [d for d in os.listdir(output_base_dir) if os.path.isdir(os.path.join(output_base_dir, d))]
        print(f"   Found {len(dirs)} directories:")
        
        for dir_name in sorted(dirs):
            dir_path = os.path.join(output_base_dir, dir_name)
            print(f"   - {dir_name}")
            
            # Check subdirectories
            if os.path.exists(dir_path):
                subdirs = [sd for sd in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, sd))]
                files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
                
                print(f"     Subdirectories: {subdirs}")
                print(f"     Files: {files}")
                
                # Check data directory
                data_dir = os.path.join(dir_path, "data")
                if os.path.exists(data_dir):
                    data_files = os.listdir(data_dir)
                    print(f"     Data files: {data_files}")
                
                # Check images directory
                images_dir = os.path.join(dir_path, "images")
                if os.path.exists(images_dir):
                    image_files = os.listdir(images_dir)
                    print(f"     Image files: {image_files}")
                
                # Check logs directory
                logs_dir = os.path.join(dir_path, "logs")
                if os.path.exists(logs_dir):
                    log_files = os.listdir(logs_dir)
                    print(f"     Log files: {log_files}")
                
                # Check query metadata
                metadata_file = os.path.join(dir_path, "query_metadata.json")
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        print(f"     Query: {metadata.get('query', 'unknown')}")
                        print(f"     Topology: {metadata.get('topology_type', 'unknown')}")
                        print(f"     Agents: {metadata.get('num_agents', 'unknown')}")
                    except Exception as e:
                        print(f"     Error reading metadata: {e}")
    else:
        print(f"✗ Output directory not found: {output_base_dir}")
    
    print(f"\n3. Dataset summary:")
    print(f"   Total items in dataset: {len(dataset)}")
    
    if dataset:
        # Show first item structure
        first_item = dataset[0]
        print(f"   First item keys: {list(first_item.keys())}")
        
        if "communication_data" in first_item:
            print(f"   Communication rounds: {len(first_item['communication_data'])}")
            
            for i, round_data in enumerate(first_item["communication_data"]):
                responses = round_data.get("structured_responses", [])
                print(f"     Round {i}: {len(responses)} responses")
                
                # Check if agent_permissions are included
                if responses:
                    first_response = responses[0]
                    if "agent_permissions" in first_response:
                        permissions = first_response["agent_permissions"]
                        print(f"       First response permissions: {list(permissions.keys())}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_folder_structure()) 