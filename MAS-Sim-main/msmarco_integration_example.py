"""
MSMarco Integration Example for NewMA
Demonstrates how to integrate msmarco.json dataset with NewMA system
"""

import json
import random
import asyncio
from typing import List, Dict, Any, Tuple
import os

from .graph_generator import AdvancedGraphGenerator
from .network_topologies import TopologyType, NetworkConfig
from .dataset_processor import DatasetProcessor, NewMADatasetGenerator


def load_msmarco_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load msmarco.json dataset similar to MA implementation"""
    print(f"Loading msmarco dataset from: {dataset_path}")
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
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
        
        print(f"Loaded {len(items)} items from msmarco dataset")
        return items
        
    except Exception as e:
        print(f"Error loading msmarco dataset: {e}")
        return []


def format_examples(dataset: List[Dict[str, Any]], idx: int) -> Tuple[str, List[str], str, str]:
    """Format dataset examples similar to MA implementation"""
    if idx >= len(dataset):
        raise IndexError(f"Index {idx} out of range")
    
    item = dataset[idx]
    query = item["question"]
    correct_answer = item["correct_answer"]
    incorrect_answer = item["incorrect_answer"]
    adv_texts = item["adv_texts"]
    
    return query, adv_texts, correct_answer, incorrect_answer


def gen_poisonrag_data(datapath: str, phase: str = "train") -> List[Tuple[str, List[str], str, str]]:
    """Generate poison RAG data similar to MA implementation"""
    dataset = load_msmarco_dataset(datapath)
    
    formatted_data = []
    for i in range(len(dataset)):
        example = format_examples(dataset, i)
        formatted_data.append(example)
    
    # Split dataset based on phase
    if phase == "train":
        formatted_data = formatted_data[:int(len(formatted_data) * 0.8)]
    else:  # test/val
        formatted_data = formatted_data[int(len(formatted_data) * 0.8):]
    
    return formatted_data


def create_network_training_data(network_data: Dict[str, Any], 
                               samples: List[Tuple[str, List[str], str, str]]) -> List[Dict[str, Any]]:
    """Create training data for network topologies with msmarco data"""
    training_data = []
    
    for sample in samples:
        query, adv_texts, correct_answer, incorrect_answer = sample
        
        training_item = {
            "network_config": network_data,
            "query": query,
            "adv_texts": adv_texts,
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer,
            "expected_behavior": "agents_should_provide_correct_answer",
            "attack_scenario": "adversarial_text_injection"
        }
        
        training_data.append(training_item)
    
    return training_data


async def generate_newma_dataset_with_msmarco(msmarco_path: str,
                                            topology_types: List[TopologyType],
                                            num_agents_list: List[int],
                                            num_samples_per_config: int = 3,
                                            phase: str = "train") -> List[Dict[str, Any]]:
    """Generate NewMA dataset using msmarco.json data"""
    
    # Load msmarco data
    msmarco_samples = gen_poisonrag_data(msmarco_path, phase)
    
    # Initialize graph generator
    generator = AdvancedGraphGenerator("gpt-4o-mini")
    
    comprehensive_dataset = []
    
    for topology_type in topology_types:
        for num_agents in num_agents_list:
            # Generate network configuration
            config = NetworkConfig(
                topology_type=topology_type,
                num_agents=num_agents,
                sparsity=0.2
            )
            
            # Generate system prompts
            system_prompts = generator.generate_system_prompts(
                num_agents=num_agents,
                attacker_ratio=0.2
            )
            
            # Create topology
            topology = generator.create_network_topology(config, system_prompts)
            
            # Extract network data
            network_data = {
                "topology_type": topology_type.value,
                "num_agents": num_agents,
                "sparsity": 0.2,
                "attacker_ratio": 0.2,
                "adjacency_matrix": topology.get_adjacency_matrix().tolist(),
                "agent_roles": {agent_id: role.value for agent_id, role in topology.agent_roles.items()},
                "system_prompts": system_prompts,
                "network_stats": topology.get_network_stats()
            }
            
            # Get random samples from msmarco
            selected_samples = random.sample(msmarco_samples, 
                                          min(num_samples_per_config, len(msmarco_samples)))
            
            # Create training data for this configuration
            training_data = create_network_training_data(network_data, selected_samples)
            comprehensive_dataset.extend(training_data)
    
    return comprehensive_dataset


def example_msmarco_integration():
    """Example of integrating msmarco.json with NewMA"""
    
    # Path to msmarco.json
    msmarco_path = "../MA/datasets/msmarco.json"
    
    if not os.path.exists(msmarco_path):
        print(f"Warning: msmarco.json not found at {msmarco_path}")
        print("Please ensure the file exists or update the path")
        return None
    
    # Define topology types
    topology_types = [
        TopologyType.LINEAR,
        TopologyType.TREE_HIERARCHY,
        TopologyType.HOLARCHY,
        TopologyType.P2P_FLAT,
        TopologyType.HYBRID
    ]
    
    # Define agent counts
    num_agents_list = [4, 6, 8]
    
    print("Generating NewMA dataset with msmarco.json integration...")
    
    # Generate dataset
    dataset = asyncio.run(generate_newma_dataset_with_msmarco(
        msmarco_path=msmarco_path,
        topology_types=topology_types,
        num_agents_list=num_agents_list,
        num_samples_per_config=2,
        phase="train"
    ))
    
    print(f"Generated dataset with {len(dataset)} items")
    
    # Save dataset
    output_path = "newma_msmarco_integration.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Dataset saved to: {output_path}")
    
    # Print sample
    if dataset:
        sample = dataset[0]
        print("\nSample dataset item:")
        print(f"- Topology: {sample['network_config']['topology_type']}")
        print(f"- Agents: {sample['network_config']['num_agents']}")
        print(f"- Query: {sample['query'][:50]}...")
        print(f"- Correct answer: {sample['correct_answer']}")
        print(f"- Incorrect answer: {sample['incorrect_answer']}")
        print(f"- Adv texts: {len(sample['adv_texts'])} items")
    
    return dataset


if __name__ == "__main__":
    dataset = example_msmarco_integration()
    
    if dataset:
        print(f"\nSuccessfully created dataset with {len(dataset)} items")
    else:
        print("Failed to create dataset") 