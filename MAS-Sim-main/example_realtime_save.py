#!/usr/bin/env python3
"""
Example script demonstrating real-time saving functionality
"""

import asyncio
import os
from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.network_topologies import TopologyType

async def example_realtime_save():
    """Example of using real-time save functionality"""
    print("=== Real-Time Save Example ===")
    
    # Initialize generator
    generator = AdvancedGraphGenerator(
        model_type="gpt-4o-mini",
        verbose=True
    )
    
    # Example save path
    save_path = "example_realtime_dataset.json"
    
    print(f"Will save dataset to: {save_path}")
    print("Dataset will be saved every 5 items during generation")
    
    # Generate a small dataset with real-time saving
    dataset = await generator.generate_comprehensive_dataset(
        topology_types=[TopologyType.STAR, TopologyType.RING],
        num_agents_list=[4, 6],
        num_networks_per_config=2,
        queries=[
            "What is the capital of France?",
            "Explain machine learning briefly."
        ],
        sparsity_range=(0.2, 0.3),
        attacker_ratio=0.25,
        attacker_strategy="adversarial_influence",
        save_filepath=save_path,
        save_interval=5,  # Save every 5 items
        max_depth=2,
        branching_factor=2
    )
    
    print(f"\nFinal dataset size: {len(dataset)} items")
    
    # Check if file was created
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"✅ Dataset saved successfully: {file_size} bytes")
        
        # Check for backup
        backup_path = save_path.replace('.json', '_backup.json')
        if os.path.exists(backup_path):
            backup_size = os.path.getsize(backup_path)
            print(f"✅ Backup created: {backup_size} bytes")
    else:
        print("❌ Dataset file not found")

def example_with_different_intervals():
    """Example showing different save intervals"""
    print("\n=== Save Interval Examples ===")
    
    intervals = [1, 5, 10, 20]
    
    for interval in intervals:
        print(f"\nSave interval {interval}:")
        print(f"  - For 100 items: {100 // interval} saves")
        print(f"  - For 50 items: {50 // interval} saves")
        print(f"  - For 25 items: {25 // interval} saves")
        
        # Calculate expected saves
        total_items = 100
        expected_saves = total_items // interval
        print(f"  - Expected saves for {total_items} items: {expected_saves}")

def example_command_line_usage():
    """Example command line usage"""
    print("\n=== Command Line Usage Examples ===")
    
    examples = [
        "# Basic usage with default save interval (10)",
        "python NewMA/graph_generator.py --save_filepath dataset.json --num_graphs 20",
        "",
        "# Custom save interval (save every 5 items)",
        "python NewMA/graph_generator.py --save_filepath dataset.json --num_graphs 20 --save_interval 5",
        "",
        "# Very frequent saves (save every item)",
        "python NewMA/graph_generator.py --save_filepath dataset.json --num_graphs 20 --save_interval 1",
        "",
        "# Less frequent saves (save every 20 items)",
        "python NewMA/graph_generator.py --save_filepath dataset.json --num_graphs 20 --save_interval 20",
        "",
        "# With MSMarco data",
        "python NewMA/graph_generator.py --save_filepath msmarco_dataset.json --use_msmarco --save_interval 5"
    ]
    
    for example in examples:
        print(example)

if __name__ == "__main__":
    # Run the example
    asyncio.run(example_realtime_save())
    
    # Show different interval examples
    example_with_different_intervals()
    
    # Show command line usage
    example_command_line_usage()
    
    print("\n=== Real-Time Save Example Completed ===") 