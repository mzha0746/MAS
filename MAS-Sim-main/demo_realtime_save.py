#!/usr/bin/env python3
"""
Demo script to show real-time saving in action
"""

import asyncio
import time
import os
from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.network_topologies import TopologyType

async def demo_realtime_save():
    """Demo real-time saving with progress monitoring"""
    print("=== Real-Time Save Demo ===")
    
    # Initialize generator
    generator = AdvancedGraphGenerator(
        model_type="gpt-4o-mini",
        verbose=False  # Less verbose for cleaner demo
    )
    
    # Demo save path
    save_path = "demo_realtime_dataset.json"
    
    print(f"Starting demo with save path: {save_path}")
    print("Dataset will be saved every 3 items")
    print("You can monitor the file size during generation...")
    print()
    
    # Start monitoring file size in background
    monitor_task = asyncio.create_task(monitor_file_size(save_path))
    
    try:
        # Generate dataset with real-time saving
        dataset = await generator.generate_comprehensive_dataset(
            topology_types=[TopologyType.STAR],
            num_agents_list=[4],
            num_networks_per_config=3,
            queries=[
                "What is 2+2?",
                "What color is the sky?",
                "What is the capital of Japan?",
                "How many planets are in our solar system?",
                "What is the largest ocean on Earth?"
            ],
            sparsity_range=(0.2, 0.3),
            attacker_ratio=0.25,
            attacker_strategy="adversarial_influence",
            save_filepath=save_path,
            save_interval=3,  # Save every 3 items
            max_depth=2,
            branching_factor=2
        )
        
        print(f"\n✅ Demo completed! Final dataset size: {len(dataset)} items")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        # Stop monitoring
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
    
    # Final file check
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"Final file size: {file_size} bytes")
        
        # Check for backup
        backup_path = save_path.replace('.json', '_backup.json')
        if os.path.exists(backup_path):
            backup_size = os.path.getsize(backup_path)
            print(f"Backup file size: {backup_size} bytes")

async def monitor_file_size(filepath: str):
    """Monitor file size changes in real-time"""
    last_size = 0
    last_check = time.time()
    
    while True:
        try:
            if os.path.exists(filepath):
                current_size = os.path.getsize(filepath)
                current_time = time.time()
                
                if current_size != last_size:
                    time_diff = current_time - last_check
                    size_diff = current_size - last_size
                    
                    print(f"📁 File updated: {current_size} bytes (+{size_diff} bytes in {time_diff:.1f}s)")
                    
                    last_size = current_size
                    last_check = current_time
            
            await asyncio.sleep(1)  # Check every second
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Monitor error: {e}")
            break

def show_file_contents(filepath: str):
    """Show a preview of the saved file contents"""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    print(f"\n=== File Contents Preview ===")
    print(f"File: {filepath}")
    
    try:
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"Total items: {len(data)}")
        
        if len(data) > 0:
            print("\nFirst item structure:")
            first_item = data[0]
            for key, value in first_item.items():
                if isinstance(value, list):
                    print(f"  {key}: [{len(value)} items]")
                elif isinstance(value, dict):
                    print(f"  {key}: {{...}}")
                else:
                    print(f"  {key}: {value}")
        
        if len(data) > 1:
            print(f"\nLast item added at: {data[-1].get('timestamp', 'N/A')}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_realtime_save())
    
    # Show file contents
    show_file_contents("demo_realtime_dataset.json")
    
    print("\n=== Demo Completed ===") 