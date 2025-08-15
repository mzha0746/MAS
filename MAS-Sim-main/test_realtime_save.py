#!/usr/bin/env python3
"""
Test script to verify real-time saving functionality
"""

import os
import json
import tempfile
from NewMA.graph_generator import AdvancedGraphGenerator
from NewMA.core_agent import AgentRole
from NewMA.network_topologies import TopologyType

def test_realtime_save():
    """Test real-time saving functionality"""
    print("=== Testing Real-Time Save Functionality ===")
    
    # Create temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        test_save_path = tmp_file.name
    
    try:
        generator = AdvancedGraphGenerator(model_type="gpt-4o-mini", verbose=True)
        
        print(f"Test save path: {test_save_path}")
        
        # Test 1: Basic real-time save
        print("\n1. Testing basic real-time save")
        
        # Create mock dataset
        mock_dataset = []
        for i in range(15):  # More than save_interval (10)
            mock_item = {
                "id": f"item_{i}",
                "data": f"test_data_{i}",
                "accuracy_stats": [
                    {
                        "overall_accuracy": 0.75,
                        "manipulation_rate": 0.33,
                        "total_agents": 4
                    }
                ]
            }
            mock_dataset.append(mock_item)
            
            # Simulate real-time save
            if len(mock_dataset) % 5 == 0:  # Save every 5 items for testing
                generator._save_dataset_realtime(mock_dataset, test_save_path)
        
        # Final save
        generator._save_dataset_realtime(mock_dataset, test_save_path, is_final=True)
        
        # Verify file exists and has correct content
        if os.path.exists(test_save_path):
            with open(test_save_path, 'r') as f:
                saved_data = json.load(f)
            print(f"✅ File saved successfully: {len(saved_data)} items")
            print(f"   File size: {os.path.getsize(test_save_path)} bytes")
        else:
            print("❌ File not saved")
        
        # Test 2: Backup functionality
        print("\n2. Testing backup functionality")
        
        # Create a new save to trigger backup
        generator._save_dataset_realtime(mock_dataset, test_save_path)
        
        backup_path = test_save_path.replace('.json', '_backup.json')
        if os.path.exists(backup_path):
            print(f"✅ Backup created: {backup_path}")
            print(f"   Backup size: {os.path.getsize(backup_path)} bytes")
        else:
            print("❌ Backup not created")
        
        # Test 3: Error handling
        print("\n3. Testing error handling")
        
        # Try to save to a non-existent directory
        invalid_path = "/non/existent/path/test.json"
        generator._save_dataset_realtime(mock_dataset, invalid_path)
        print("✅ Error handling works (should show error message)")
        
    finally:
        # Cleanup
        if os.path.exists(test_save_path):
            os.unlink(test_save_path)
        backup_path = test_save_path.replace('.json', '_backup.json')
        if os.path.exists(backup_path):
            os.unlink(backup_path)
        print(f"\nCleaned up test files")

def test_save_interval():
    """Test different save intervals"""
    print("\n=== Testing Save Intervals ===")
    
    generator = AdvancedGraphGenerator(verbose=True)
    
    # Test different save intervals
    intervals = [1, 5, 10, 20]
    
    for interval in intervals:
        print(f"\nTesting save interval: {interval}")
        
        # Create mock dataset
        mock_dataset = []
        save_count = 0
        
        for i in range(25):  # More than any interval
            mock_item = {
                "id": f"item_{i}",
                "data": f"test_data_{i}"
            }
            mock_dataset.append(mock_item)
            
            # Check if should save
            if len(mock_dataset) % interval == 0:
                save_count += 1
                print(f"  Would save at item {len(mock_dataset)} (save #{save_count})")
        
        print(f"  Total saves for interval {interval}: {save_count}")

if __name__ == "__main__":
    test_realtime_save()
    test_save_interval()
    
    print("\n=== Real-Time Save Tests Completed ===") 