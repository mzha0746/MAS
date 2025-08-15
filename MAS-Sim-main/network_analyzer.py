"""
Network Analyzer for Multi-Agent Networks
Analyzes performance and characteristics of different topologies
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import pandas as pd
from collections import defaultdict

from .network_topologies import TopologyType, NetworkConfig
from .core_agent import AgentRole, AgentType


@dataclass
class NetworkMetrics:
    """Network performance metrics"""
    topology_type: str
    num_agents: int
    sparsity: float
    density: float
    average_clustering: float
    average_shortest_path: float
    diameter: float
    efficiency: float
    robustness: float
    scalability_score: float
    communication_overhead: float
    fault_tolerance: float


class NetworkAnalyzer:
    """Analyzer for multi-agent network topologies"""
    
    def __init__(self):
        self.metrics_history: List[NetworkMetrics] = []
        self.analysis_results: Dict[str, Any] = {}
    
    def analyze_network(self, network_data: Dict[str, Any]) -> NetworkMetrics:
        """Analyze a single network configuration"""
        adjacency_matrix = np.array(network_data["adjacency_matrix"])
        topology_type = network_data["topology_type"]
        num_agents = network_data["num_agents"]
        sparsity = network_data["sparsity"]
        
        # Create networkx graph
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Calculate basic metrics
        density = nx.density(G)
        average_clustering = nx.average_clustering(G)
        
        # Calculate path metrics
        if nx.is_connected(G):
            average_shortest_path = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
        else:
            # For disconnected graphs, calculate for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            average_shortest_path = nx.average_shortest_path_length(subgraph)
            diameter = nx.diameter(subgraph)
        
        # Calculate efficiency (inverse of average shortest path)
        efficiency = 1.0 / (average_shortest_path + 1e-6)
        
        # Calculate robustness (based on connectivity)
        robustness = self._calculate_robustness(G)
        
        # Calculate scalability score
        scalability_score = self._calculate_scalability_score(G, num_agents)
        
        # Calculate communication overhead
        communication_overhead = self._calculate_communication_overhead(G)
        
        # Calculate fault tolerance
        fault_tolerance = self._calculate_fault_tolerance(G)
        
        metrics = NetworkMetrics(
            topology_type=topology_type,
            num_agents=num_agents,
            sparsity=sparsity,
            density=density,
            average_clustering=average_clustering,
            average_shortest_path=average_shortest_path,
            diameter=diameter,
            efficiency=efficiency,
            robustness=robustness,
            scalability_score=scalability_score,
            communication_overhead=communication_overhead,
            fault_tolerance=fault_tolerance
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _calculate_robustness(self, G: nx.Graph) -> float:
        """Calculate network robustness"""
        if len(G.nodes()) <= 1:
            return 1.0
        
        # Calculate node connectivity
        try:
            node_connectivity = nx.node_connectivity(G)
        except:
            node_connectivity = 0
        
        # Calculate edge connectivity
        try:
            edge_connectivity = nx.edge_connectivity(G)
        except:
            edge_connectivity = 0
        
        # Normalize by number of nodes
        max_connectivity = len(G.nodes()) - 1
        robustness = (node_connectivity + edge_connectivity) / (2 * max_connectivity + 1e-6)
        
        return min(1.0, robustness)
    
    def _calculate_scalability_score(self, G: nx.Graph, num_agents: int) -> float:
        """Calculate scalability score"""
        # Consider factors like degree distribution and clustering
        degrees = [d for n, d in G.degree()]
        avg_degree = np.mean(degrees)
        
        # Lower average degree indicates better scalability
        degree_score = 1.0 / (1.0 + avg_degree / num_agents)
        
        # Consider clustering coefficient
        clustering = nx.average_clustering(G)
        clustering_score = 1.0 - clustering  # Lower clustering is better for scalability
        
        # Consider diameter
        if nx.is_connected(G):
            diameter = nx.diameter(G)
            diameter_score = 1.0 / (1.0 + diameter / num_agents)
        else:
            diameter_score = 0.5
        
        # Combine scores
        scalability = (degree_score + clustering_score + diameter_score) / 3
        return scalability
    
    def _calculate_communication_overhead(self, G: nx.Graph) -> float:
        """Calculate communication overhead"""
        num_edges = G.number_of_edges()
        num_nodes = G.number_of_nodes()
        
        # Normalize by maximum possible edges
        max_edges = num_nodes * (num_nodes - 1) / 2
        overhead = num_edges / (max_edges + 1e-6)
        
        return overhead
    
    def _calculate_fault_tolerance(self, G: nx.Graph) -> float:
        """Calculate fault tolerance"""
        if len(G.nodes()) <= 1:
            return 1.0
        
        # Calculate node connectivity
        try:
            node_connectivity = nx.node_connectivity(G)
        except:
            node_connectivity = 0
        
        # Calculate edge connectivity
        try:
            edge_connectivity = nx.edge_connectivity(G)
        except:
            edge_connectivity = 0
        
        # Consider redundancy
        avg_degree = np.mean([d for n, d in G.degree()])
        redundancy = avg_degree / (len(G.nodes()) - 1)
        
        # Combine metrics
        fault_tolerance = (node_connectivity + edge_connectivity + redundancy) / 3
        return min(1.0, fault_tolerance)
    
    def analyze_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze entire dataset"""
        results = {
            "topology_analysis": {},
            "performance_comparison": {},
            "scalability_analysis": {},
            "robustness_analysis": {}
        }
        
        # Group by topology type
        topology_groups = defaultdict(list)
        for network_data in dataset:
            topology_type = network_data["topology_type"]
            topology_groups[topology_type].append(network_data)
        
        # Analyze each topology type
        for topology_type, networks in topology_groups.items():
            metrics_list = []
            for network in networks:
                metrics = self.analyze_network(network)
                metrics_list.append(metrics)
            
            # Calculate average metrics for this topology
            avg_metrics = self._calculate_average_metrics(metrics_list)
            results["topology_analysis"][topology_type] = avg_metrics
        
        # Performance comparison
        results["performance_comparison"] = self._compare_topologies()
        
        # Scalability analysis
        results["scalability_analysis"] = self._analyze_scalability()
        
        # Robustness analysis
        results["robustness_analysis"] = self._analyze_robustness()
        
        self.analysis_results = results
        return results
    
    def _calculate_average_metrics(self, metrics_list: List[NetworkMetrics]) -> Dict[str, float]:
        """Calculate average metrics for a group"""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for field in NetworkMetrics.__dataclass_fields__:
            if field != "topology_type":
                values = [getattr(m, field) for m in metrics_list]
                avg_metrics[field] = np.mean(values)
        
        return avg_metrics
    
    def _compare_topologies(self) -> Dict[str, Any]:
        """Compare different topology types"""
        comparison = {}
        
        # Group metrics by topology type
        topology_metrics = defaultdict(list)
        for metrics in self.metrics_history:
            topology_metrics[metrics.topology_type].append(metrics)
        
        # Calculate rankings for each metric
        metric_fields = ["efficiency", "robustness", "scalability_score", 
                        "communication_overhead", "fault_tolerance"]
        
        for field in metric_fields:
            rankings = {}
            for topology_type, metrics_list in topology_metrics.items():
                avg_value = np.mean([getattr(m, field) for m in metrics_list])
                rankings[topology_type] = avg_value
            
            # Sort by value (higher is better for most metrics)
            sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
            comparison[field] = dict(sorted_rankings)
        
        return comparison
    
    def _analyze_scalability(self) -> Dict[str, Any]:
        """Analyze scalability characteristics"""
        scalability_data = defaultdict(list)
        
        for metrics in self.metrics_history:
            scalability_data[metrics.topology_type].append({
                "num_agents": metrics.num_agents,
                "scalability_score": metrics.scalability_score,
                "communication_overhead": metrics.communication_overhead,
                "efficiency": metrics.efficiency
            })
        
        # Calculate scalability trends
        trends = {}
        for topology_type, data in scalability_data.items():
            if len(data) > 1:
                # Calculate correlation with number of agents
                agent_counts = [d["num_agents"] for d in data]
                scalability_scores = [d["scalability_score"] for d in data]
                
                correlation = np.corrcoef(agent_counts, scalability_scores)[0, 1]
                trends[topology_type] = {
                    "scalability_trend": correlation,
                    "avg_scalability": np.mean(scalability_scores),
                    "scalability_variance": np.var(scalability_scores)
                }
        
        return trends
    
    def _analyze_robustness(self) -> Dict[str, Any]:
        """Analyze robustness characteristics"""
        robustness_data = defaultdict(list)
        
        for metrics in self.metrics_history:
            robustness_data[metrics.topology_type].append({
                "num_agents": metrics.num_agents,
                "robustness": metrics.robustness,
                "fault_tolerance": metrics.fault_tolerance,
                "density": metrics.density
            })
        
        # Calculate robustness characteristics
        characteristics = {}
        for topology_type, data in robustness_data.items():
            if data:
                characteristics[topology_type] = {
                    "avg_robustness": np.mean([d["robustness"] for d in data]),
                    "avg_fault_tolerance": np.mean([d["fault_tolerance"] for d in data]),
                    "robustness_variance": np.var([d["robustness"] for d in data]),
                    "fault_tolerance_variance": np.var([d["fault_tolerance"] for d in data])
                }
        
        return characteristics
    
    def generate_visualizations(self, save_path: Optional[str] = None):
        """Generate visualization plots"""
        if not self.metrics_history:
            print("No metrics available for visualization")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Multi-Agent Network Topology Analysis", fontsize=16)
        
        # 1. Efficiency comparison
        self._plot_efficiency_comparison(axes[0, 0])
        
        # 2. Scalability analysis
        self._plot_scalability_analysis(axes[0, 1])
        
        # 3. Robustness comparison
        self._plot_robustness_comparison(axes[0, 2])
        
        # 4. Communication overhead
        self._plot_communication_overhead(axes[1, 0])
        
        # 5. Fault tolerance
        self._plot_fault_tolerance(axes[1, 1])
        
        # 6. Topology distribution
        self._plot_topology_distribution(axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def _plot_efficiency_comparison(self, ax):
        """Plot efficiency comparison"""
        topology_metrics = defaultdict(list)
        for metrics in self.metrics_history:
            topology_metrics[metrics.topology_type].append(metrics.efficiency)
        
        topologies = list(topology_metrics.keys())
        efficiencies = [np.mean(topology_metrics[t]) for t in topologies]
        
        bars = ax.bar(topologies, efficiencies, color='skyblue')
        ax.set_title("Network Efficiency Comparison")
        ax.set_ylabel("Efficiency")
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, efficiencies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_scalability_analysis(self, ax):
        """Plot scalability analysis"""
        topology_metrics = defaultdict(list)
        for metrics in self.metrics_history:
            topology_metrics[metrics.topology_type].append({
                'num_agents': metrics.num_agents,
                'scalability': metrics.scalability_score
            })
        
        for topology_type, data in topology_metrics.items():
            if data:
                agent_counts = [d['num_agents'] for d in data]
                scalability_scores = [d['scalability'] for d in data]
                ax.scatter(agent_counts, scalability_scores, label=topology_type, alpha=0.7)
        
        ax.set_title("Scalability Analysis")
        ax.set_xlabel("Number of Agents")
        ax.set_ylabel("Scalability Score")
        ax.legend()
    
    def _plot_robustness_comparison(self, ax):
        """Plot robustness comparison"""
        topology_metrics = defaultdict(list)
        for metrics in self.metrics_history:
            topology_metrics[metrics.topology_type].append(metrics.robustness)
        
        topologies = list(topology_metrics.keys())
        robustness_scores = [np.mean(topology_metrics[t]) for t in topologies]
        
        bars = ax.bar(topologies, robustness_scores, color='lightcoral')
        ax.set_title("Network Robustness Comparison")
        ax.set_ylabel("Robustness")
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, robustness_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_communication_overhead(self, ax):
        """Plot communication overhead"""
        topology_metrics = defaultdict(list)
        for metrics in self.metrics_history:
            topology_metrics[metrics.topology_type].append(metrics.communication_overhead)
        
        topologies = list(topology_metrics.keys())
        overhead_scores = [np.mean(topology_metrics[t]) for t in topologies]
        
        bars = ax.bar(topologies, overhead_scores, color='lightgreen')
        ax.set_title("Communication Overhead")
        ax.set_ylabel("Overhead")
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, overhead_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_fault_tolerance(self, ax):
        """Plot fault tolerance"""
        topology_metrics = defaultdict(list)
        for metrics in self.metrics_history:
            topology_metrics[metrics.topology_type].append(metrics.fault_tolerance)
        
        topologies = list(topology_metrics.keys())
        tolerance_scores = [np.mean(topology_metrics[t]) for t in topologies]
        
        bars = ax.bar(topologies, tolerance_scores, color='gold')
        ax.set_title("Fault Tolerance")
        ax.set_ylabel("Tolerance")
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, tolerance_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_topology_distribution(self, ax):
        """Plot topology distribution"""
        topology_counts = defaultdict(int)
        for metrics in self.metrics_history:
            topology_counts[metrics.topology_type] += 1
        
        topologies = list(topology_counts.keys())
        counts = list(topology_counts.values())
        
        ax.pie(counts, labels=topologies, autopct='%1.1f%%', startangle=90)
        ax.set_title("Topology Distribution")
    
    def save_analysis_report(self, filepath: str):
        """Save analysis report to file"""
        report = {
            "analysis_results": self.analysis_results,
            "metrics_summary": self._generate_metrics_summary(),
            "recommendations": self._generate_recommendations()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _generate_metrics_summary(self) -> Dict[str, Any]:
        """Generate metrics summary"""
        summary = {
            "total_networks_analyzed": len(self.metrics_history),
            "topology_types": list(set(m.topology_type for m in self.metrics_history)),
            "agent_count_range": {
                "min": min(m.num_agents for m in self.metrics_history),
                "max": max(m.num_agents for m in self.metrics_history)
            },
            "average_metrics": {}
        }
        
        # Calculate overall averages
        metric_fields = ["efficiency", "robustness", "scalability_score", 
                        "communication_overhead", "fault_tolerance"]
        
        for field in metric_fields:
            values = [getattr(m, field) for m in self.metrics_history]
            summary["average_metrics"][field] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        
        return summary
    
    def _generate_recommendations(self) -> Dict[str, List[str]]:
        """Generate recommendations based on analysis"""
        recommendations = {
            "efficiency": [],
            "robustness": [],
            "scalability": [],
            "general": []
        }
        
        # Analyze topology performance
        topology_performance = defaultdict(list)
        for metrics in self.metrics_history:
            topology_performance[metrics.topology_type].append(metrics)
        
        # Efficiency recommendations
        efficiency_rankings = self.analysis_results.get("performance_comparison", {}).get("efficiency", {})
        if efficiency_rankings:
            best_efficiency = max(efficiency_rankings.items(), key=lambda x: x[1])
            recommendations["efficiency"].append(
                f"Best efficiency: {best_efficiency[0]} topology ({best_efficiency[1]:.3f})"
            )
        
        # Robustness recommendations
        robustness_rankings = self.analysis_results.get("performance_comparison", {}).get("robustness", {})
        if robustness_rankings:
            best_robustness = max(robustness_rankings.items(), key=lambda x: x[1])
            recommendations["robustness"].append(
                f"Best robustness: {best_robustness[0]} topology ({best_robustness[1]:.3f})"
            )
        
        # Scalability recommendations
        scalability_analysis = self.analysis_results.get("scalability_analysis", {})
        if scalability_analysis:
            best_scalability = max(scalability_analysis.items(), key=lambda x: x[1].get("avg_scalability", 0))
            recommendations["scalability"].append(
                f"Best scalability: {best_scalability[0]} topology"
            )
        
        # General recommendations
        recommendations["general"].append(
            "Consider hybrid topologies for balanced performance across multiple metrics"
        )
        recommendations["general"].append(
            "Linear topologies are efficient but lack fault tolerance"
        )
        recommendations["general"].append(
            "P2P topologies provide good fault tolerance but higher communication overhead"
        )
        
        return recommendations 