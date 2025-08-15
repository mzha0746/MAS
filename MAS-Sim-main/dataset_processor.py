"""
Dataset Processor for NewMA
Handles loading and processing of external datasets like msmarco.json
"""

import json
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import os


@dataclass
class DatasetItem:
    """Structure for dataset items"""
    id: str
    question: str
    correct_answer: str
    incorrect_answer: str
    adv_texts: List[str]


class DatasetProcessor:
    """Processor for external datasets like msmarco.json"""
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path
        self.dataset = []
        self.processed_items = []
    
    def load_msmarco_dataset(self, dataset_path: str = None) -> List[DatasetItem]:
        """Load msmarco.json dataset"""
        if dataset_path:
            self.dataset_path = dataset_path
        elif not self.dataset_path:
            raise ValueError("Dataset path must be provided")
        
        print(f"Loading dataset from: {self.dataset_path}")
        
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            items = []
            for item_id, item_data in data.items():
                item = DatasetItem(
                    id=item_id,
                    question=item_data.get("question", ""),
                    correct_answer=item_data.get("correct answer", ""),
                    incorrect_answer=item_data.get("incorrect answer", ""),
                    adv_texts=item_data.get("adv_texts", [])
                )
                items.append(item)
            
            self.dataset = items
            print(f"Loaded {len(items)} items from dataset")
            return items
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
    
    def format_examples(self, dataset: List[DatasetItem], idx: int) -> Tuple[str, List[str], str, str]:
        """Format dataset examples similar to MA implementation"""
        if idx >= len(dataset):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(dataset)}")
        
        item = dataset[idx]
        query = item.question
        correct_answer = item.correct_answer
        incorrect_answer = item.incorrect_answer
        adv_texts = item.adv_texts
        
        return query, adv_texts, correct_answer, incorrect_answer
    
    def gen_poisonrag_data(self, datapath: str = None, phase: str = "train") -> List[Tuple[str, List[str], str, str]]:
        """Generate poison RAG data similar to MA implementation"""
        if datapath:
            self.dataset_path = datapath
        
        if not self.dataset:
            self.load_msmarco_dataset()
        
        dataset = []
        for i in range(len(self.dataset)):
            example = self.format_examples(self.dataset, i)
            dataset.append(example)
        
        # Split dataset based on phase
        if phase == "train":
            dataset = dataset[:int(len(dataset) * 0.8)]
        else:  # test/val
            dataset = dataset[int(len(dataset) * 0.8):]
        
        return dataset
    
    def get_random_samples(self, num_samples: int, phase: str = "train") -> List[Tuple[str, List[str], str, str]]:
        """Get random samples from dataset"""
        if not self.dataset:
            self.load_msmarco_dataset()
        
        all_samples = self.gen_poisonrag_data(phase=phase)
        
        if num_samples >= len(all_samples):
            return all_samples
        
        return random.sample(all_samples, num_samples)
    
    def create_network_training_data(self, 
                                   network_data: Dict[str, Any],
                                   samples: List[Tuple[str, List[str], str, str]]) -> List[Dict[str, Any]]:
        """Create training data for network topologies"""
        training_data = []
        
        for sample in samples:
            query, adv_texts, correct_answer, incorrect_answer = sample
            
            # Create training item
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


class NewMADatasetGenerator:
    """Enhanced dataset generator for NewMA with external dataset support"""
    
    def __init__(self, model_type: str = "gpt-4o-mini"):
        self.model_type = model_type
        self.dataset_processor = DatasetProcessor()
    
    def load_external_dataset(self, dataset_path: str) -> DatasetProcessor:
        """Load external dataset"""
        self.dataset_processor.load_msmarco_dataset(dataset_path)
        return self.dataset_processor
    
    def generate_dataset_with_external_data(self,
                                          network_data: Dict[str, Any],
                                          dataset_path: str,
                                          num_samples: int = 10,
                                          phase: str = "train") -> List[Dict[str, Any]]:
        """Generate dataset using external data like msmarco.json"""
        
        # Load external dataset
        processor = self.load_external_dataset(dataset_path)
        
        # Get samples from external dataset
        samples = processor.get_random_samples(num_samples, phase)
        
        # Create training data
        training_data = processor.create_network_training_data(network_data, samples)
        
        return training_data
    
    def create_comprehensive_dataset(self,
                                   topology_types: List[str],
                                   num_agents_list: List[int],
                                   dataset_path: str,
                                   num_samples_per_config: int = 5,
                                   phase: str = "train") -> List[Dict[str, Any]]:
        """Create comprehensive dataset with external data"""
        
        # Load external dataset
        processor = self.load_external_dataset(dataset_path)
        
        comprehensive_dataset = []
        
        # Generate samples for each configuration
        for topology_type in topology_types:
            for num_agents in num_agents_list:
                # Create network configuration
                network_config = {
                    "topology_type": topology_type,
                    "num_agents": num_agents,
                    "sparsity": 0.2,
                    "attacker_ratio": 0.2
                }
                
                # Get samples from external dataset
                samples = processor.get_random_samples(num_samples_per_config, phase)
                
                # Create training data for this configuration
                training_data = processor.create_network_training_data(network_config, samples)
                comprehensive_dataset.extend(training_data)
        
        return comprehensive_dataset


def create_msmarco_integration_example():
    """Example of how to integrate msmarco.json with NewMA"""
    
    # Initialize dataset generator
    generator = NewMADatasetGenerator("gpt-4o-mini")
    
    # Path to msmarco.json (adjust as needed)
    msmarco_path = "../MA/datasets/msmarco.json"
    
    if not os.path.exists(msmarco_path):
        print(f"Warning: msmarco.json not found at {msmarco_path}")
        print("Please ensure the file exists or update the path")
        return None
    
    # Create comprehensive dataset
    topology_types = ["linear", "tree_hierarchy", "holarchy", "p2p_flat", "hybrid"]
    num_agents_list = [4, 6, 8]
    
    dataset = generator.create_comprehensive_dataset(
        topology_types=topology_types,
        num_agents_list=num_agents_list,
        dataset_path=msmarco_path,
        num_samples_per_config=3,
        phase="train"
    )
    
    print(f"Generated comprehensive dataset with {len(dataset)} items")
    
    # Save dataset
    output_path = "newma_msmarco_dataset.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Dataset saved to: {output_path}")
    
    return dataset


if __name__ == "__main__":
    # Example usage
    dataset = create_msmarco_integration_example()
    
    if dataset:
        print(f"Successfully created dataset with {len(dataset)} items")
        print("Sample item structure:")
        if dataset:
            sample = dataset[0]
            print(f"- Network config: {sample['network_config']['topology_type']}")
            print(f"- Query: {sample['query'][:50]}...")
            print(f"- Correct answer: {sample['correct_answer']}")
            print(f"- Incorrect answer: {sample['incorrect_answer']}")
            print(f"- Number of adv texts: {len(sample['adv_texts'])}")
    else:
        print("Failed to create dataset") 