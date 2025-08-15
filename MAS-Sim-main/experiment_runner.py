"""
Experiment Runner for Multi-Agent Networks
Executes experiments with different network topologies
"""

import asyncio
import json
import time
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import os

from .core_agent import AgentRole, AgentType
from .network_topologies import (
    NetworkConfig, TopologyType, BaseNetworkTopology,
    NetworkTopologyFactory
)
from .advanced_topologies import (
    HolarchyTopology, P2PTopology, HybridTopology
)
from .graph_generator import AdvancedGraphGenerator
from .network_analyzer import NetworkAnalyzer


@dataclass
class ExperimentConfig:
    """Configuration for network experiments"""
    topology_type: TopologyType
    num_agents: int
    sparsity: float
    num_experiments: int
    queries: List[str]
    num_dialogue_turns: int
    model_type: str
    attacker_ratio: float = 0.2
    max_depth: int = 3
    branching_factor: int = 3
    p2p_connection_type: str = "mesh"
    hybrid_centralization_ratio: float = 0.3


@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    experiment_id: str
    topology_type: str
    num_agents: int
    sparsity: float
    query: str
    execution_time: float
    success: bool
    response_quality: float
    communication_efficiency: float
    error_message: Optional[str] = None
    detailed_results: Optional[Dict[str, Any]] = None


class ExperimentRunner:
    """Runner for multi-agent network experiments"""
    
    def __init__(self, model_type: str = "gpt-4o-mini"):
        self.model_type = model_type
        self.generator = AdvancedGraphGenerator(model_type)
        self.analyzer = NetworkAnalyzer()
        self.results: List[ExperimentResult] = []
        
    def run_single_experiment(self, config: ExperimentConfig, 
                             query: str, experiment_id: str) -> ExperimentResult:
        """Run a single experiment"""
        start_time = time.time()
        
        try:
            # Generate network configuration
            network_config = NetworkConfig(
                topology_type=config.topology_type,
                num_agents=config.num_agents,
                sparsity=config.sparsity,
                max_depth=config.max_depth,
                branching_factor=config.branching_factor,
                p2p_connection_type=config.p2p_connection_type,
                hybrid_centralization_ratio=config.hybrid_centralization_ratio
            )
            
            # Generate system prompts
            system_prompts = self.generator.generate_system_prompts(
                num_agents=config.num_agents,
                attacker_ratio=config.attacker_ratio
            )
            
            # Create network topology
            topology = self.generator.create_network_topology(network_config, system_prompts)
            
            # Execute task based on topology type
            if config.topology_type == TopologyType.LINEAR:
                result = topology.process_pipeline(query)
                response_quality = self._evaluate_response_quality(result)
                communication_efficiency = self._calculate_communication_efficiency(topology, result)
            elif config.topology_type == TopologyType.TREE_HIERARCHY:
                result = topology.execute_hierarchical_task(query)
                response_quality = self._evaluate_response_quality([result])
                communication_efficiency = self._calculate_communication_efficiency(topology, [result])
            elif config.topology_type == TopologyType.HOLARCHY:
                result = topology.execute_holarchic_task(query)
                response_quality = self._evaluate_response_quality([result])
                communication_efficiency = self._calculate_communication_efficiency(topology, [result])
            elif config.topology_type == TopologyType.P2P_FLAT:
                result = topology.execute_p2p_task(query)
                response_quality = self._evaluate_response_quality(result)
                communication_efficiency = self._calculate_communication_efficiency(topology, result)
            elif config.topology_type == TopologyType.HYBRID:
                result = topology.execute_hybrid_task(query)
                response_quality = self._evaluate_response_quality([result])
                communication_efficiency = self._calculate_communication_efficiency(topology, [result])
            else:
                # Default: individual agent responses
                result = []
                for agent in topology.agents.values():
                    response = agent.chat(f"Query: {query}")
                    result.append(response)
                response_quality = self._evaluate_response_quality(result)
                communication_efficiency = self._calculate_communication_efficiency(topology, result)
            
            execution_time = time.time() - start_time
            
            return ExperimentResult(
                experiment_id=experiment_id,
                topology_type=config.topology_type.value,
                num_agents=config.num_agents,
                sparsity=config.sparsity,
                query=query,
                execution_time=execution_time,
                success=True,
                response_quality=response_quality,
                communication_efficiency=communication_efficiency,
                detailed_results={
                    "result": result,
                    "network_stats": topology.get_network_stats(),
                    "adjacency_matrix": topology.get_adjacency_matrix().tolist()
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExperimentResult(
                experiment_id=experiment_id,
                topology_type=config.topology_type.value,
                num_agents=config.num_agents,
                sparsity=config.sparsity,
                query=query,
                execution_time=execution_time,
                success=False,
                response_quality=0.0,
                communication_efficiency=0.0,
                error_message=str(e)
            )
    
    async def arun_single_experiment(self, config: ExperimentConfig, 
                                    query: str, experiment_id: str) -> ExperimentResult:
        """Run a single experiment asynchronously"""
        start_time = time.time()
        
        try:
            # Generate network configuration
            network_config = NetworkConfig(
                topology_type=config.topology_type,
                num_agents=config.num_agents,
                sparsity=config.sparsity,
                max_depth=config.max_depth,
                branching_factor=config.branching_factor,
                p2p_connection_type=config.p2p_connection_type,
                hybrid_centralization_ratio=config.hybrid_centralization_ratio
            )
            
            # Generate system prompts
            system_prompts = self.generator.generate_system_prompts(
                num_agents=config.num_agents,
                attacker_ratio=config.attacker_ratio
            )
            
            # Create network topology
            topology = self.generator.create_network_topology(network_config, system_prompts)
            
            # Execute task based on topology type
            if config.topology_type == TopologyType.LINEAR:
                result = await topology.aprocess_pipeline(query)
                response_quality = self._evaluate_response_quality(result)
                communication_efficiency = self._calculate_communication_efficiency(topology, result)
            elif config.topology_type == TopologyType.TREE_HIERARCHY:
                result = await topology.aexecute_hierarchical_task(query)
                response_quality = self._evaluate_response_quality([result])
                communication_efficiency = self._calculate_communication_efficiency(topology, [result])
            elif config.topology_type == TopologyType.HOLARCHY:
                result = await topology.aexecute_holarchic_task(query)
                response_quality = self._evaluate_response_quality([result])
                communication_efficiency = self._calculate_communication_efficiency(topology, [result])
            elif config.topology_type == TopologyType.P2P_FLAT:
                result = await topology.aexecute_p2p_task(query)
                response_quality = self._evaluate_response_quality(result)
                communication_efficiency = self._calculate_communication_efficiency(topology, result)
            elif config.topology_type == TopologyType.HYBRID:
                result = await topology.aexecute_hybrid_task(query)
                response_quality = self._evaluate_response_quality([result])
                communication_efficiency = self._calculate_communication_efficiency(topology, [result])
            else:
                # Default: individual agent responses
                result = []
                tasks = []
                for agent in topology.agents.values():
                    tasks.append(asyncio.create_task(
                        agent.achat(f"Query: {query}")
                    ))
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                result = [r for r in responses if not isinstance(r, Exception)]
                response_quality = self._evaluate_response_quality(result)
                communication_efficiency = self._calculate_communication_efficiency(topology, result)
            
            execution_time = time.time() - start_time
            
            return ExperimentResult(
                experiment_id=experiment_id,
                topology_type=config.topology_type.value,
                num_agents=config.num_agents,
                sparsity=config.sparsity,
                query=query,
                execution_time=execution_time,
                success=True,
                response_quality=response_quality,
                communication_efficiency=communication_efficiency,
                detailed_results={
                    "result": result,
                    "network_stats": topology.get_network_stats(),
                    "adjacency_matrix": topology.get_adjacency_matrix().tolist()
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExperimentResult(
                experiment_id=experiment_id,
                topology_type=config.topology_type.value,
                num_agents=config.num_agents,
                sparsity=config.sparsity,
                query=query,
                execution_time=execution_time,
                success=False,
                response_quality=0.0,
                communication_efficiency=0.0,
                error_message=str(e)
            )
    
    def _evaluate_response_quality(self, responses: List[Any]) -> float:
        """Evaluate the quality of responses"""
        if not responses:
            return 0.0
        
        # Simple quality metrics
        quality_scores = []
        
        for response in responses:
            if isinstance(response, str):
                # Basic quality heuristics
                length_score = min(1.0, len(response) / 100)  # Prefer longer responses
                coherence_score = 1.0 if len(response.split()) > 5 else 0.5  # Prefer coherent responses
                quality_scores.append((length_score + coherence_score) / 2)
            else:
                quality_scores.append(0.5)  # Default score for non-string responses
        
        return np.mean(quality_scores)
    
    def _calculate_communication_efficiency(self, topology: BaseNetworkTopology, 
                                         results: List[Any]) -> float:
        """Calculate communication efficiency"""
        if not results:
            return 0.0
        
        # Consider network density and response quality
        network_stats = topology.get_network_stats()
        density = network_stats.get("density", 0.0)
        
        # Lower density (fewer connections) is more efficient
        density_efficiency = 1.0 - density
        
        # Consider number of responses vs number of agents
        response_ratio = len(results) / len(topology.agents)
        
        # Combine metrics
        efficiency = (density_efficiency + response_ratio) / 2
        return min(1.0, efficiency)
    
    def run_experiment_suite(self, configs: List[ExperimentConfig]) -> List[ExperimentResult]:
        """Run a suite of experiments"""
        all_results = []
        
        for config in configs:
            print(f"Running experiments for {config.topology_type.value} topology...")
            
            for exp_idx in range(config.num_experiments):
                for query_idx, query in enumerate(config.queries):
                    experiment_id = f"{config.topology_type.value}_{exp_idx}_{query_idx}"
                    
                    result = self.run_single_experiment(config, query, experiment_id)
                    all_results.append(result)
                    
                    print(f"  Experiment {exp_idx+1}/{config.num_experiments}, "
                          f"Query {query_idx+1}/{len(config.queries)}: "
                          f"Success={result.success}, "
                          f"Quality={result.response_quality:.3f}, "
                          f"Efficiency={result.communication_efficiency:.3f}")
        
        self.results = all_results
        return all_results
    
    async def arun_experiment_suite(self, configs: List[ExperimentConfig]) -> List[ExperimentResult]:
        """Run a suite of experiments asynchronously"""
        all_results = []
        
        for config in configs:
            print(f"Running experiments for {config.topology_type.value} topology...")
            
            tasks = []
            for exp_idx in range(config.num_experiments):
                for query_idx, query in enumerate(config.queries):
                    experiment_id = f"{config.topology_type.value}_{exp_idx}_{query_idx}"
                    
                    task = asyncio.create_task(
                        self.arun_single_experiment(config, query, experiment_id)
                    )
                    tasks.append(task)
            
            # Run all experiments for this config concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"  Experiment failed: {result}")
                else:
                    all_results.append(result)
                    exp_idx = i // len(config.queries)
                    query_idx = i % len(config.queries)
                    print(f"  Experiment {exp_idx+1}/{config.num_experiments}, "
                          f"Query {query_idx+1}/{len(config.queries)}: "
                          f"Success={result.success}, "
                          f"Quality={result.response_quality:.3f}, "
                          f"Efficiency={result.communication_efficiency:.3f}")
        
        self.results = all_results
        return all_results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results"""
        if not self.results:
            return {}
        
        analysis = {
            "summary": self._generate_summary(),
            "topology_comparison": self._compare_topologies(),
            "performance_analysis": self._analyze_performance(),
            "recommendations": self._generate_recommendations()
        }
        
        return analysis
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of experiment results"""
        total_experiments = len(self.results)
        successful_experiments = sum(1 for r in self.results if r.success)
        
        summary = {
            "total_experiments": total_experiments,
            "successful_experiments": successful_experiments,
            "success_rate": successful_experiments / total_experiments if total_experiments > 0 else 0,
            "topology_types": list(set(r.topology_type for r in self.results)),
            "average_execution_time": np.mean([r.execution_time for r in self.results]),
            "average_response_quality": np.mean([r.response_quality for r in self.results if r.success]),
            "average_communication_efficiency": np.mean([r.communication_efficiency for r in self.results if r.success])
        }
        
        return summary
    
    def _compare_topologies(self) -> Dict[str, Any]:
        """Compare performance across topology types"""
        topology_results = {}
        
        for result in self.results:
            if result.topology_type not in topology_results:
                topology_results[result.topology_type] = []
            topology_results[result.topology_type].append(result)
        
        comparison = {}
        for topology_type, results in topology_results.items():
            successful_results = [r for r in results if r.success]
            
            if successful_results:
                comparison[topology_type] = {
                    "success_rate": len(successful_results) / len(results),
                    "avg_execution_time": np.mean([r.execution_time for r in successful_results]),
                    "avg_response_quality": np.mean([r.response_quality for r in successful_results]),
                    "avg_communication_efficiency": np.mean([r.communication_efficiency for r in successful_results]),
                    "num_experiments": len(results)
                }
        
        return comparison
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance characteristics"""
        performance = {
            "execution_time_analysis": {},
            "quality_analysis": {},
            "efficiency_analysis": {}
        }
        
        # Execution time analysis
        execution_times = [r.execution_time for r in self.results if r.success]
        if execution_times:
            performance["execution_time_analysis"] = {
                "mean": np.mean(execution_times),
                "std": np.std(execution_times),
                "min": np.min(execution_times),
                "max": np.max(execution_times)
            }
        
        # Quality analysis
        qualities = [r.response_quality for r in self.results if r.success]
        if qualities:
            performance["quality_analysis"] = {
                "mean": np.mean(qualities),
                "std": np.std(qualities),
                "min": np.min(qualities),
                "max": np.max(qualities)
            }
        
        # Efficiency analysis
        efficiencies = [r.communication_efficiency for r in self.results if r.success]
        if efficiencies:
            performance["efficiency_analysis"] = {
                "mean": np.mean(efficiencies),
                "std": np.std(efficiencies),
                "min": np.min(efficiencies),
                "max": np.max(efficiencies)
            }
        
        return performance
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        # Find best performing topology
        topology_comparison = self._compare_topologies()
        if topology_comparison:
            best_quality = max(topology_comparison.items(), 
                             key=lambda x: x[1].get("avg_response_quality", 0))
            best_efficiency = max(topology_comparison.items(), 
                                key=lambda x: x[1].get("avg_communication_efficiency", 0))
            
            recommendations.append(f"Best response quality: {best_quality[0]} topology")
            recommendations.append(f"Best communication efficiency: {best_efficiency[0]} topology")
        
        # General recommendations
        success_rate = self._generate_summary().get("success_rate", 0)
        if success_rate < 0.8:
            recommendations.append("Consider reducing network complexity or agent count for better reliability")
        
        avg_time = self._generate_summary().get("average_execution_time", 0)
        if avg_time > 10:  # More than 10 seconds
            recommendations.append("Consider using asynchronous execution for better performance")
        
        return recommendations
    
    def save_results(self, filepath: str):
        """Save experiment results to file"""
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_result = {
                "experiment_id": result.experiment_id,
                "topology_type": result.topology_type,
                "num_agents": result.num_agents,
                "sparsity": result.sparsity,
                "query": result.query,
                "execution_time": result.execution_time,
                "success": result.success,
                "response_quality": result.response_quality,
                "communication_efficiency": result.communication_efficiency,
                "error_message": result.error_message
            }
            serializable_results.append(serializable_result)
        
        # Save results and analysis
        output = {
            "results": serializable_results,
            "analysis": self.analyze_results(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)


def create_experiment_configs(attacker_ratio: float = 0.2) -> List[ExperimentConfig]:
    """Create default experiment configurations with custom attacker ratio"""
    configs = []
    
    # Sample queries
    queries = [
        "What is the capital of France?",
        "Explain the concept of machine learning.",
        "How does photosynthesis work?",
        "What are the benefits of renewable energy?",
        "Explain the difference between AI and AGI."
    ]
    
    # Linear pipeline experiments
    configs.append(ExperimentConfig(
        topology_type=TopologyType.LINEAR,
        num_agents=6,
        sparsity=0.2,
        num_experiments=3,
        queries=queries,
        num_dialogue_turns=3,
        model_type="gpt-4o-mini",
        attacker_ratio=attacker_ratio
    ))
    
    # Tree hierarchy experiments
    configs.append(ExperimentConfig(
        topology_type=TopologyType.TREE_HIERARCHY,
        num_agents=8,
        sparsity=0.3,
        num_experiments=3,
        queries=queries,
        num_dialogue_turns=3,
        model_type="gpt-4o-mini",
        max_depth=3,
        branching_factor=2,
        attacker_ratio=attacker_ratio
    ))
    
    # Holarchy experiments
    configs.append(ExperimentConfig(
        topology_type=TopologyType.HOLARCHY,
        num_agents=6,
        sparsity=0.4,
        num_experiments=3,
        queries=queries,
        num_dialogue_turns=3,
        model_type="gpt-4o-mini",
        attacker_ratio=attacker_ratio
    ))
    
    # P2P experiments
    configs.append(ExperimentConfig(
        topology_type=TopologyType.P2P_FLAT,
        num_agents=8,
        sparsity=0.3,
        num_experiments=3,
        queries=queries,
        num_dialogue_turns=3,
        model_type="gpt-4o-mini",
        p2p_connection_type="mesh",
        attacker_ratio=attacker_ratio
    ))
    
    # Hybrid experiments
    configs.append(ExperimentConfig(
        topology_type=TopologyType.HYBRID,
        num_agents=10,
        sparsity=0.25,
        num_experiments=3,
        queries=queries,
        num_dialogue_turns=3,
        model_type="gpt-4o-mini",
        hybrid_centralization_ratio=0.3,
        attacker_ratio=attacker_ratio
    ))
    
    return configs


if __name__ == "__main__":
    import argparse
    
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Run multi-agent network experiments")
        
        parser.add_argument("--model_type", type=str, default="gpt-4o-mini", help="LLM model type")
        parser.add_argument("--output_dir", type=str, default="./experiment_results", help="Output directory")
        parser.add_argument("--async_mode", action="store_true", help="Run experiments asynchronously")
        parser.add_argument("--attacker_ratio", type=float, default=0.2, help="Ratio of attacker agents (0.0-1.0)")
        parser.add_argument("--num_agents", type=int, default=None, help="Number of agents per network (overrides default)")
        parser.add_argument("--topology_type", type=str, default=None, 
                          choices=["linear", "tree_hierarchy", "holarchy", "p2p_flat", "hybrid"],
                          help="Specific topology type to test")
        
        args = parser.parse_args()
        
        # Validate attacker ratio
        if not 0.0 <= args.attacker_ratio <= 1.0:
            raise ValueError("attacker_ratio must be between 0.0 and 1.0")
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        return args
    
    args = parse_arguments()
    
    # Create experiment runner
    runner = ExperimentRunner(model_type=args.model_type)
    
    # Create experiment configurations with custom attacker ratio
    if args.topology_type:
        # Create single topology configuration
        topology_map = {
            "linear": TopologyType.LINEAR,
            "tree_hierarchy": TopologyType.TREE_HIERARCHY,
            "holarchy": TopologyType.HOLARCHY,
            "p2p_flat": TopologyType.P2P_FLAT,
            "hybrid": TopologyType.HYBRID
        }
        
        queries = [
            "What is the capital of France?",
            "Explain the concept of machine learning.",
            "How does photosynthesis work?",
            "What are the benefits of renewable energy?",
            "Explain the difference between AI and AGI."
        ]
        
        num_agents = args.num_agents or 6
        
        configs = [ExperimentConfig(
            topology_type=topology_map[args.topology_type],
            num_agents=num_agents,
            sparsity=0.2,
            num_experiments=3,
            queries=queries,
            num_dialogue_turns=3,
            model_type=args.model_type,
            attacker_ratio=args.attacker_ratio
        )]
    else:
        # Create all topology configurations
        configs = create_experiment_configs(attacker_ratio=args.attacker_ratio)
        
        # Override num_agents if specified
        if args.num_agents:
            for config in configs:
                config.num_agents = args.num_agents
    
    # Run experiments
    if args.async_mode:
        results = asyncio.run(runner.arun_experiment_suite(configs))
    else:
        results = runner.run_experiment_suite(configs)
    
    # Analyze results
    analysis = runner.analyze_results()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"experiment_results_{timestamp}.json")
    runner.save_results(output_file)
    
    print(f"Experiment completed. Results saved to: {output_file}")
    print(f"Attacker ratio used: {args.attacker_ratio}")
    print(f"Summary: {analysis['summary']}") 