"""
Simple MSMarco Integration Example for NewMA
"""

import json
import random
import os

def load_msmarco_data(dataset_path: str):
    """Load msmarco.json dataset"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    items = []
    for item_id, item_data in data.items():
        item = {
            "id": item_id,
            "question": item_data.get("question", ""),
            "correct_answer": item_data.get("correct answer", ""),
            "incorrect_answer": item_data.get("incorrect answer", ""),
            "adv_texts": item_data.get("adv_texts", [])
        }
        items.append(item)
    
    return items

def create_newma_training_data(network_config, msmarco_samples):
    """Create training data for NewMA with msmarco samples"""
    training_data = []
    
    for sample in msmarco_samples:
        training_item = {
            "network_config": network_config,
            "query": sample["question"],
            "adv_texts": sample["adv_texts"],
            "correct_answer": sample["correct_answer"],
            "incorrect_answer": sample["incorrect_answer"]
        }
        training_data.append(training_item)
    
    return training_data

def main():
    """Main function to demonstrate msmarco integration"""
    
    # Path to msmarco.json
    msmarco_path = "../MA/datasets/msmarco.json"
    
    if not os.path.exists(msmarco_path):
        print(f"msmarco.json not found at {msmarco_path}")
        return
    
    # Load msmarco data
    msmarco_data = load_msmarco_data(msmarco_path)
    print(f"Loaded {len(msmarco_data)} items from msmarco.json")
    
    # Sample network configuration
    network_config = {
        "topology_type": "linear",
        "num_agents": 6,
        "sparsity": 0.2,
        "attacker_ratio": 0.2
    }
    
    # Get random samples
    samples = random.sample(msmarco_data, min(5, len(msmarco_data)))
    
    # Create training data
    training_data = create_newma_training_data(network_config, samples)
    
    # Save to file
    with open("newma_msmarco_example.json", "w") as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Created training data with {len(training_data)} items")
    print("Sample item:")
    if training_data:
        sample = training_data[0]
        print(f"- Query: {sample['query'][:50]}...")
        print(f"- Correct: {sample['correct_answer']}")
        print(f"- Incorrect: {sample['incorrect_answer']}")

if __name__ == "__main__":
    main() 