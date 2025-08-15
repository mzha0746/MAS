#!/usr/bin/env python3
"""
Example usage of graph_generator.py with msmarco.json integration
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.network_topologies import TopologyType


def example_with_msmarco():
    """Example of using graph_generator.py with msmarco.json"""
    
    # Path to msmarco.json
    msmarco_path = "../MA/datasets/msmarco.json"
    
    if not os.path.exists(msmarco_path):
        print(f"Error: msmarco.json not found at {msmarco_path}")
        print("Please ensure the file exists or update the path")
        return
    
    print("=== NewMA Graph Generator with MSMarco Integration ===")
    
    # Initialize generator
    generator = AdvancedGraphGenerator("gpt-4o-mini")
    
    # Load msmarco dataset
    print(f"Loading msmarco dataset from: {msmarco_path}")
    msmarco_data = generator.load_msmarco_dataset(msmarco_path, phase="train")
    
    if not msmarco_data:
        print("Failed to load msmarco dataset")
        return
    
    print(f"Successfully loaded {len(msmarco_data)} msmarco samples")
    
    # Get some random samples
    samples = generator.get_random_msmarco_samples(5)
    print(f"Retrieved {len(samples)} random samples")
    
    # Print sample data
    if samples:
        print("\nSample msmarco data:")
        sample = samples[0]
        print(f"- Question: {sample['question'][:50]}...")
        print(f"- Correct answer: {sample['correct_answer']}")
        print(f"- Incorrect answer: {sample['incorrect_answer']}")
        print(f"- Number of adv texts: {len(sample['adv_texts'])}")
    
    # Generate dataset with msmarco integration
    print("\nGenerating dataset with msmarco integration...")
    
    topology_types = [
        TopologyType.LINEAR,
        TopologyType.TREE_HIERARCHY,
        TopologyType.HOLARCHY,
        TopologyType.P2P_FLAT,
        TopologyType.HYBRID
    ]
    
    num_agents_list = [4, 6, 8]
    
    try:
        dataset = asyncio.run(generator.generate_comprehensive_dataset(
            topology_types=topology_types,
            num_agents_list=num_agents_list,
            num_networks_per_config=1,
            use_msmarco=True,
            msmarco_samples_per_config=2,
            attacker_ratio=0.2,
            sparsity_range=(0.1, 0.4)
        ))
        
        print(f"Generated dataset with {len(dataset)} items")
        
        # Save dataset
        output_path = "newma_msmarco_example_dataset.json"
        with open(output_path, 'w') as f:
            import json
            json.dump(dataset, f, indent=2)
        
        print(f"Dataset saved to: {output_path}")
        
        # Print sample from generated dataset
        if dataset:
            print("\nSample generated item:")
            sample = dataset[0]
            print(f"- Topology: {sample['topology_type']}")
            print(f"- Agents: {sample['num_agents']}")
            print(f"- Query: {sample['query'][:50]}...")
            if 'correct_answer' in sample:
                print(f"- Correct answer: {sample['correct_answer']}")
                print(f"- Data source: {sample.get('data_source', 'unknown')}")
        
    except Exception as e:
        print(f"Error generating dataset: {e}")
        import traceback
        traceback.print_exc()


def example_command_line_usage():
    """Example of command line usage with msmarco"""
    
    print("\n=== Command Line Usage Example ===")
    print("To use graph_generator.py with msmarco.json, run:")
    print()
    print("python graph_generator.py \\")
    print("    --msmarco_path ../MA/datasets/msmarco.json \\")
    print("    --msmarco_phase train \\")
    print("    --msmarco_samples_per_config 3 \\")
    print("    --num_agents 6 \\")
    print("    --num_graphs 10 \\")
    print("    --attacker_ratio 0.2 \\")
    print("    --save_dir ./msmarco_integrated_dataset")
    print()
    print("This will:")
    print("- Load msmarco.json dataset")
    print("- Use msmarco questions and adversarial texts")
    print("- Generate 10 network configurations")
    print("- Use 3 msmarco samples per configuration")
    print("- Save results to specified directory")


if __name__ == "__main__":
    example_with_msmarco()
    example_command_line_usage() 