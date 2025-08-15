"""
Enhanced Graph Generator for NewMA with External Dataset Support
Integrates external datasets like msmarco.json with network topology generation
"""

import random
import json
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import os
from datetime import datetime

from .core_agent import AgentRole, AgentType
from .network_topologies import (
    NetworkConfig, TopologyType, BaseNetworkTopology,
    LinearPipelineTopology, TreeHierarchyTopology,
    NetworkTopologyFactory
)
from .advanced_topologies import (
    HolarchyTopology, P2PTopology, HybridTopology
)
from .agent_prompts import ATTACKER_SYS_PROMPT, SYS_PROMPT, ATTACKER_PROMPTS, DEFAULT_ATTACKER_STRATEGY
from .dataset_processor import DatasetProcessor, NewMADatasetGenerator


class EnhancedGraphGenerator:
    """Enhanced graph generator with external dataset support"""
    
    def __init__(self, model_type: str = "gpt-4o-mini"):
        self.model_type = model_type
        self.dataset_processor = DatasetProcessor()
        self.topology_factories = {
            TopologyType.LINEAR: LinearPipelineTopology,
            TopologyType.TREE_HIERARCHY: TreeHierarchyTopology,
            TopologyType.HOLARCHY: HolarchyTopology,
            TopologyType.P2P_FLAT: P2PTopology,
            TopologyType.P2P_STRUCTURED: P2PTopology,
            TopologyType.HYBRID: HybridTopology
        }
    
    def generate_topology_config(self, 
                               topology_type: TopologyType,
                               num_agents: int,
                               sparsity: float = 0.2,
                               **kwargs) -> NetworkConfig:
        """Generate topology configuration"""
        return NetworkConfig(
            topology_type=topology_type,
            num_agents=num_agents,
            sparsity=sparsity,
            **kwargs
        )
    
    def generate_system_prompts(self,
                               num_agents: int,
                               attacker_ratio: float = 0.2,
                               attacker_strategy: str = DEFAULT_ATTACKER_STRATEGY) -> List[str]:
        """Generate system prompts for agents"""
        system_prompts = []
        num_attackers = int(num_agents * attacker_ratio)
        
        # Randomly select attacker positions
        attacker_positions = random.sample(range(num_agents), num_attackers)
        
        for i in range(num_agents):
            if i in attacker_positions:
                # Attacker agent
                prompt = f"Agent_{i}: {ATTACKER_SYS_PROMPT}"
            else:
                # Normal agent
                prompt = f"Agent_{i}: {SYS_PROMPT}"
            system_prompts.append(prompt)
        
        return system_prompts
    
    def create_network_topology(self, config: NetworkConfig, system_prompts: List[str]) -> BaseNetworkTopology:
        """Create network topology"""
        topology_class = self.topology_factories.get(config.topology_type)
        if not topology_class:
            raise ValueError(f"Unsupported topology type: {config.topology_type}")
        
        topology = topology_class(config)
        topology.create_agents(system_prompts, self.model_type)
        topology.setup_connections()
        
        return topology
    
    def generate_initial_data(self, network_data: Dict[str, Any], example: Tuple[str, List[str], str, str]) -> Dict[str, Any]:
        """Generate initial data combining network and dataset example"""
        query, adv_texts, correct_answer, incorrect_answer = example
        
        new_data = {
            "network_config": network_data,
            "query": query,
            "adv_texts": adv_texts,
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer,
            "expected_behavior": "agents_should_provide_correct_answer",
            "attack_scenario": "adversarial_text_injection"
        }
        
        return new_data
    
    async def generate_communication_data(self,
                                       network_data: Dict[str, Any],
                                       query: str,
                                       adv_texts: List[str],
                                       num_dialogue_turns: int = 3) -> Dict[str, Any]:
        """Generate communication data for a network with external dataset"""
        
        # Recreate topology from network data
        config = NetworkConfig(
            topology_type=TopologyType(network_data["topology_type"]),
            num_agents=network_data["num_agents"],
            sparsity=network_data["sparsity"],
            max_depth=network_data.get("max_depth", 3),
            branching_factor=network_data.get("branching_factor", 3),
            p2p_connection_type=network_data.get("p2p_connection_type", "mesh"),
            hybrid_centralization_ratio=network_data.get("hybrid_centralization_ratio", 0.3)
        )
        
        # Generate system prompts
        system_prompts = self.generate_system_prompts(
            num_agents=network_data["num_agents"],
            attacker_ratio=network_data["attacker_ratio"]
        )
        
        topology = self.create_network_topology(config, system_prompts)
        
        # Generate communication data
        communication_data = []
        
        # Combine query with adversarial texts
        context = " ".join(adv_texts) if adv_texts else ""
        full_query = f"Query: {query}\nContext: {context}"
        
        # Initial responses
        if hasattr(topology, 'process_pipeline'):
            # Linear pipeline
            initial_responses = topology.process_pipeline(full_query)
            communication_data.append(initial_responses)
        elif hasattr(topology, 'execute_hierarchical_task'):
            # Tree hierarchy
            initial_response = topology.execute_hierarchical_task(full_query)
            communication_data.append([initial_response])
        elif hasattr(topology, 'execute_holarchic_task'):
            # Holarchy
            initial_response = topology.execute_holarchic_task(full_query)
            communication_data.append([initial_response])
        elif hasattr(topology, 'execute_p2p_task'):
            # P2P
            initial_responses = topology.execute_p2p_task(full_query)
            communication_data.append(initial_responses)
        elif hasattr(topology, 'execute_hybrid_task'):
            # Hybrid
            initial_response = topology.execute_hybrid_task(full_query)
            communication_data.append([initial_response])
        else:
            # Default: individual agent responses
            initial_responses = []
            for agent in topology.agents.values():
                response = agent.chat(full_query)
                initial_responses.append(response)
            communication_data.append(initial_responses)
        
        # Additional dialogue turns
        for turn in range(num_dialogue_turns - 1):
            try:
                if hasattr(topology, 'are_generate'):
                    responses = await topology.are_generate()
                    communication_data.append(responses)
                else:
                    # Fallback: generate new responses
                    responses = []
                    for agent in topology.agents.values():
                        response = agent.chat(f"Follow-up turn {turn + 1}: {query}")
                        responses.append(response)
                    communication_data.append(responses)
            except Exception as e:
                print(f"Error in dialogue turn {turn + 1}: {e}")
                break
        
        return {
            "network_data": network_data,
            "query": query,
            "adv_texts": adv_texts,
            "correct_answer": network_data.get("correct_answer", ""),
            "incorrect_answer": network_data.get("incorrect_answer", ""),
            "communication_data": communication_data,
            "num_dialogue_turns": len(communication_data)
        }
    
    async def generate_dataset_with_external_data(self,
                                                topology_types: List[TopologyType],
                                                num_agents_list: List[int],
                                                dataset_path: str,
                                                num_samples_per_config: int = 5,
                                                num_dialogue_turns: int = 3,
                                                sparsity_range: Tuple[float, float] = (0.1, 0.5),
                                                attacker_ratio: float = 0.2,
                                                phase: str = "train") -> List[Dict[str, Any]]:
        """Generate comprehensive dataset with external data like msmarco.json"""
        
        # Load external dataset
        self.dataset_processor.load_msmarco_dataset(dataset_path)
        
        dataset = []
        
        for topology_type in topology_types:
            for num_agents in num_agents_list:
                for i in range(num_samples_per_config):
                    # Random sparsity
                    sparsity = random.uniform(*sparsity_range)
                    
                    # Generate network configuration
                    config = self.generate_topology_config(
                        topology_type=topology_type,
                        num_agents=num_agents,
                        sparsity=sparsity
                    )
                    
                    # Generate system prompts
                    system_prompts = self.generate_system_prompts(
                        num_agents=num_agents,
                        attacker_ratio=attacker_ratio
                    )
                    
                    # Create topology
                    topology = self.create_network_topology(config, system_prompts)
                    
                    # Extract network data
                    network_data = {
                        "topology_type": topology_type.value,
                        "num_agents": num_agents,
                        "sparsity": sparsity,
                        "attacker_ratio": attacker_ratio,
                        "adjacency_matrix": topology.get_adjacency_matrix().tolist(),
                        "agent_roles": {agent_id: role.value for agent_id, role in topology.agent_roles.items()},
                        "system_prompts": system_prompts,
                        "network_stats": topology.get_network_stats(),
                        "config": {
                            "topology_type": config.topology_type.value,
                            "num_agents": config.num_agents,
                            "sparsity": config.sparsity,
                            "max_depth": config.max_depth,
                            "branching_factor": config.branching_factor,
                            "p2p_connection_type": config.p2p_connection_type,
                            "hybrid_centralization_ratio": config.hybrid_centralization_ratio
                        }
                    }
                    
                    # Get samples from external dataset
                    samples = self.dataset_processor.get_random_samples(1, phase)
                    
                    for sample in samples:
                        query, adv_texts, correct_answer, incorrect_answer = sample
                        
                        try:
                            # Generate communication data
                            communication_data = await self.generate_communication_data(
                                network_data=network_data,
                                query=query,
                                adv_texts=adv_texts,
                                num_dialogue_turns=num_dialogue_turns
                            )
                            
                            # Add dataset information
                            communication_data["correct_answer"] = correct_answer
                            communication_data["incorrect_answer"] = incorrect_answer
                            communication_data["expected_behavior"] = "agents_should_provide_correct_answer"
                            communication_data["attack_scenario"] = "adversarial_text_injection"
                            
                            dataset.append(communication_data)
                            
                        except Exception as e:
                            print(f"Error generating communication data: {e}")
                            continue
        
        return dataset


def generate_enhanced_dataset(args):
    """Generate enhanced dataset with external data support"""
    
    # Initialize enhanced generator
    generator = EnhancedGraphGenerator(model_type=args.model_type)
    
    # Define topology types
    topology_types = [
        TopologyType.LINEAR,
        TopologyType.TREE_HIERARCHY,
        TopologyType.HOLARCHY,
        TopologyType.P2P_FLAT,
        TopologyType.HYBRID
    ]
    
    # Define agent counts
    num_agents_list = [4, 6, 8, 10]
    
    print(f"Generating enhanced dataset with external data...")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Topology types: {len(topology_types)}")
    print(f"Agent counts: {num_agents_list}")
    print(f"Attacker ratio: {args.attacker_ratio}")
    
    # Generate dataset
    try:
        dataset = asyncio.run(generator.generate_dataset_with_external_data(
            topology_types=topology_types,
            num_agents_list=num_agents_list,
            dataset_path=args.dataset_path,
            num_samples_per_config=args.num_samples_per_config,
            num_dialogue_turns=args.num_dialogue_turns,
            sparsity_range=(0.1, 0.4),
            attacker_ratio=args.attacker_ratio,
            phase=args.phase
        ))
        
        print(f"Generated {len(dataset)} network configurations with external data")
        
    except Exception as e:
        print(f"Error generating dataset: {e}")
        import traceback
        traceback.print_exc()
        dataset = []
    
    # Save dataset
    with open(args.save_filepath, "w") as file:
        json.dump(dataset, file, indent=2)
    
    print(f"Dataset saved to: {args.save_filepath}")
    return dataset


if __name__ == "__main__":
    import argparse
    
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Generate enhanced multi-agent network datasets with external data")
        
        parser.add_argument("--dataset_path", type=str, required=True, 
                          help="Path to external dataset (e.g., msmarco.json)")
        parser.add_argument("--num_agents", type=int, default=8, help="Number of agents per network")
        parser.add_argument("--num_samples_per_config", type=int, default=5, 
                          help="Number of samples per network configuration")
        parser.add_argument("--sparsity", type=float, default=0.2, help="Network sparsity")
        parser.add_argument("--attacker_ratio", type=float, default=0.2, 
                          help="Ratio of attacker agents (0.0-1.0)")
        parser.add_argument("--num_dialogue_turns", type=int, default=3, 
                          help="Number of dialogue turns")
        parser.add_argument("--phase", type=str, default="train", 
                          choices=["train", "test", "val"], help="Dataset phase")
        parser.add_argument("--save_dir", type=str, default="./enhanced_agent_dataset")
        parser.add_argument("--model_type", type=str, default="gpt-4o-mini", help="LLM model type")
        parser.add_argument("--save_filepath", type=str, help="Output file path")
        
        args = parser.parse_args()
        
        # Validate attacker ratio
        if not 0.0 <= args.attacker_ratio <= 1.0:
            raise ValueError("attacker_ratio must be between 0.0 and 1.0")
        
        # Set default save path
        if not args.save_filepath:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.save_filepath = os.path.join(
                args.save_dir, 
                f"enhanced_network_{current_time_str}.json"
            )
        
        return args
    
    args = parse_arguments()
    dataset = generate_enhanced_dataset(args)
    print(f"Enhanced dataset generation completed!")
    print(f"Total configurations: {len(dataset)}") 