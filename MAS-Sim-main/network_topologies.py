"""
Network Topology Implementations
Implements various multi-agent network architectures
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import random
import json

from NewMA.core_agent import (
    BaseAgent, LinearAgent, ManagerAgent, WorkerAgent, 
    HolonAgent, PeerAgent, HybridAgent, AgentRole, AgentType
)


class TopologyType(Enum):
    """Topology type enumeration"""
    LINEAR = "linear"
    TREE_HIERARCHY = "tree_hierarchy"
    HOLARCHY = "holarchy"
    P2P_FLAT = "p2p_flat"
    P2P_STRUCTURED = "p2p_structured"
    HYBRID = "hybrid"


@dataclass
class NetworkConfig:
    """Network configuration"""
    topology_type: TopologyType
    num_agents: int
    sparsity: float = 0.2
    max_depth: int = 3
    branching_factor: int = 3
    p2p_connection_type: str = "mesh"
    hybrid_centralization_ratio: float = 0.3
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseNetworkTopology:
    """Base class for network topologies"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.adjacency_matrix: Optional[np.ndarray] = None
        self.network_graph: Optional[nx.Graph] = None
        self.agent_roles: Dict[str, AgentRole] = {}
        
    def create_agents(self, system_prompts: List[str], model_type: str = "gpt-4o-mini", agent_roles: List[AgentRole] = None, hierarchy_info: Dict[str, dict] = None):
        """Create agents with given system prompts, roles, and hierarchy information"""
        raise NotImplementedError
    
    def setup_connections(self):
        """Setup connections between agents"""
        raise NotImplementedError
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix representation"""
        if self.adjacency_matrix is None:
            self._build_adjacency_matrix()
        return self.adjacency_matrix
    
    def _build_adjacency_matrix(self):
        """Build adjacency matrix from network graph"""
        if self.network_graph is None:
            self._build_network_graph()
        
        num_agents = len(self.agents)
        self.adjacency_matrix = np.zeros((num_agents, num_agents), dtype=int)
        
        for i, agent_id_i in enumerate(self.agents.keys()):
            for j, agent_id_j in enumerate(self.agents.keys()):
                if self.network_graph.has_edge(agent_id_i, agent_id_j):
                    self.adjacency_matrix[i, j] = 1
    
    def _build_network_graph(self):
        """Build network graph representation"""
        self.network_graph = nx.Graph()
        for agent_id in self.agents.keys():
            self.network_graph.add_node(agent_id)
    
    def get_agent_by_id(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agent_neighbors(self, agent_id: str) -> List[BaseAgent]:
        """Get neighbors of an agent"""
        if self.network_graph is None:
            self._build_network_graph()
        
        neighbors = []
        if agent_id in self.network_graph:
            for neighbor_id in self.network_graph.neighbors(agent_id):
                if neighbor_id in self.agents:
                    neighbors.append(self.agents[neighbor_id])
        return neighbors
    
    def broadcast_message(self, message_type: str, content: Any, sender_id: Optional[str] = None):
        """Broadcast message to all agents"""
        for agent_id, agent in self.agents.items():
            if sender_id != agent_id:
                agent.send_message(agent_id, message_type, content)
    
    async def abroadcast_message(self, message_type: str, content: Any, sender_id: Optional[str] = None):
        """Asynchronous broadcast message"""
        tasks = []
        for agent_id, agent in self.agents.items():
            if sender_id != agent_id:
                tasks.append(asyncio.create_task(
                    agent.achat(f"Received {message_type}: {content}")
                ))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        if self.network_graph is None:
            self._build_network_graph()
        
        # Handle directed vs undirected graphs
        if self.network_graph.is_directed():
            # For directed graphs, check if it's weakly connected
            is_connected = nx.is_weakly_connected(self.network_graph)
            if is_connected:
                try:
                    average_shortest_path = nx.average_shortest_path_length(self.network_graph)
                except nx.NetworkXError:
                    average_shortest_path = float('inf')
            else:
                average_shortest_path = float('inf')
        else:
            # For undirected graphs
            is_connected = nx.is_connected(self.network_graph)
            if is_connected:
                try:
                    average_shortest_path = nx.average_shortest_path_length(self.network_graph)
                except nx.NetworkXError:
                    average_shortest_path = float('inf')
            else:
                average_shortest_path = float('inf')
        
        return {
            "num_agents": len(self.agents),
            "num_edges": self.network_graph.number_of_edges(),
            "density": nx.density(self.network_graph),
            "average_clustering": nx.average_clustering(self.network_graph),
            "average_shortest_path": average_shortest_path,
            "topology_type": self.config.topology_type.value
        }


class LinearPipelineTopology(BaseNetworkTopology):
    """Linear pipeline topology implementation"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__(config)
        self.stages: List[LinearAgent] = []
    
    def create_agents(self, system_prompts: List[str], model_type: str = "gpt-4o-mini", agent_roles: List[AgentRole] = None, hierarchy_info: Dict[str, dict] = None):
        """Create linear pipeline agents with hierarchy information"""
        num_agents = min(len(system_prompts), self.config.num_agents)
        
        for i in range(num_agents):
            agent_id = f"linear_agent_{i}"
            agent = LinearAgent(
                agent_id=agent_id,
                system_prompt=system_prompts[i],
                model_type=model_type,
                position=i,
                total_stages=num_agents
            )
            self.agents[agent_id] = agent
            self.stages.append(agent)
            # Use provided role or default to NORMAL
            if agent_roles and i < len(agent_roles):
                self.agent_roles[agent_id] = agent_roles[i]
            else:
                self.agent_roles[agent_id] = AgentRole.NORMAL
            
            # Update hierarchy information if provided
            if hierarchy_info and agent_id in hierarchy_info:
                agent.update_hierarchy_info(hierarchy_info[agent_id])
    
    def setup_connections(self):
        """Setup linear connections"""
        self.network_graph = nx.DiGraph()
        
        # Add nodes
        for agent_id in self.agents.keys():
            self.network_graph.add_node(agent_id)
        
        # Add edges in linear sequence
        agent_ids = list(self.agents.keys())
        for i in range(len(agent_ids) - 1):
            self.network_graph.add_edge(agent_ids[i], agent_ids[i + 1])
            
            # Setup neighbor connections
            current_agent = self.agents[agent_ids[i]]
            next_agent = self.agents[agent_ids[i + 1]]
            current_agent.add_neighbor(next_agent)
    
    def process_pipeline(self, input_data: Any) -> List[Any]:
        """Process data through the linear pipeline"""
        results = []
        current_data = input_data
        
        for stage in self.stages:
            result = stage.process_stage(current_data)
            results.append(result)
            current_data = result
        
        return results
    
    async def aprocess_pipeline(self, input_data: Any) -> List[Any]:
        """Asynchronous pipeline processing"""
        results = []
        current_data = input_data
        
        for stage in self.stages:
            result = await stage.aprocess_stage(current_data)
            results.append(result)
            current_data = result
        
        return results


class TreeHierarchyTopology(BaseNetworkTopology):
    """Tree hierarchy topology implementation"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__(config)
        self.managers: List[ManagerAgent] = []
        self.workers: List[WorkerAgent] = []
        self.hierarchy_levels: Dict[int, List[BaseAgent]] = {}
    
    def create_agents(self, system_prompts: List[str], model_type: str = "gpt-4o-mini", agent_roles: List[AgentRole] = None, hierarchy_info: Dict[str, dict] = None):
        """Create tree hierarchy agents with hierarchy information"""
        num_agents = min(len(system_prompts), self.config.num_agents)
        
        # Calculate hierarchy structure
        total_levels = min(self.config.max_depth, int(np.log2(num_agents)) + 1)
        agents_per_level = self._calculate_agents_per_level(num_agents, total_levels)
        
        agent_idx = 0
        for level in range(total_levels):
            level_agents = []
            num_level_agents = agents_per_level[level]
            
            for i in range(num_level_agents):
                if agent_idx >= len(system_prompts):
                    break
                
                agent_id = f"tree_agent_{level}_{i}"
                
                if level == 0:  # Root level - managers
                    agent = ManagerAgent(
                        agent_id=agent_id,
                        system_prompt=system_prompts[agent_idx],
                        model_type=model_type
                    )
                    self.managers.append(agent)
                    # Use provided role or default to MANAGER
                    if agent_roles and agent_idx < len(agent_roles):
                        role = agent_roles[agent_idx]
                        self.agent_roles[agent_id] = role if role == AgentRole.ATTACKER else AgentRole.MANAGER
                    else:
                        self.agent_roles[agent_id] = AgentRole.MANAGER
                else:  # Worker levels
                    agent = WorkerAgent(
                        agent_id=agent_id,
                        system_prompt=system_prompts[agent_idx],
                        model_type=model_type
                    )
                    self.workers.append(agent)
                    # Use provided role or default to WORKER
                    if agent_roles and agent_idx < len(agent_roles):
                        role = agent_roles[agent_idx]
                        self.agent_roles[agent_id] = role if role == AgentRole.ATTACKER else AgentRole.WORKER
                    else:
                        self.agent_roles[agent_id] = AgentRole.WORKER
                
                self.agents[agent_id] = agent
                level_agents.append(agent)
                
                # Update hierarchy information if provided
                if hierarchy_info and agent_id in hierarchy_info:
                    agent.update_hierarchy_info(hierarchy_info[agent_id])
                
                agent_idx += 1
            
            self.hierarchy_levels[level] = level_agents
    
    def _calculate_agents_per_level(self, total_agents: int, levels: int) -> List[int]:
        """Calculate number of agents per hierarchy level"""
        agents_per_level = []
        remaining_agents = total_agents
        
        for level in range(levels):
            if level == 0:  # Root level
                num_agents = min(1, remaining_agents)
            else:  # Worker levels
                max_workers = self.config.branching_factor ** level
                num_agents = min(max_workers, remaining_agents - 1)
            
            agents_per_level.append(num_agents)
            remaining_agents -= num_agents
            
            if remaining_agents <= 0:
                break
        
        return agents_per_level
    
    def setup_connections(self):
        """Setup tree hierarchy connections"""
        self.network_graph = nx.DiGraph()
        
        # Add nodes
        for agent_id in self.agents.keys():
            self.network_graph.add_node(agent_id)
        
        # Setup manager-worker relationships
        for level in range(len(self.hierarchy_levels) - 1):
            managers = self.hierarchy_levels[level]
            workers = self.hierarchy_levels[level + 1]
            
            for i, manager in enumerate(managers):
                # Only managers can add workers
                if hasattr(manager, 'add_worker'):
                    # Assign workers to this manager
                    start_idx = i * self.config.branching_factor
                    end_idx = min(start_idx + self.config.branching_factor, len(workers))
                    
                    for j in range(start_idx, end_idx):
                        worker = workers[j]
                        manager.add_worker(worker)
                        worker.set_manager(manager)
                        
                        # Add edge in graph
                        self.network_graph.add_edge(manager.agent_id, worker.agent_id)
    
    def execute_hierarchical_task(self, task: Any) -> Any:
        """Execute task using hierarchical decomposition"""
        if not self.managers:
            return None
        
        root_manager = self.managers[0]
        subtasks = root_manager.decompose_task(task)
        
        # Distribute subtasks to workers
        worker_results = []
        for i, subtask in enumerate(subtasks):
            if i < len(self.workers):
                result = self.workers[i].execute_task(subtask)
                worker_results.append(result)
        
        # Aggregate results
        final_result = root_manager.aggregate_results(worker_results)
        return final_result
    
    async def aexecute_hierarchical_task(self, task: Any) -> Any:
        """Asynchronous hierarchical task execution"""
        if not self.managers:
            return None
        
        root_manager = self.managers[0]
        subtasks = await root_manager.adecompose_task(task)
        
        # Distribute subtasks to workers
        tasks = []
        for i, subtask in enumerate(subtasks):
            if i < len(self.workers):
                tasks.append(asyncio.create_task(
                    self.workers[i].aexecute_task(subtask)
                ))
        
        worker_results = await asyncio.gather(*tasks, return_exceptions=True)
        worker_results = [r for r in worker_results if not isinstance(r, Exception)]
        
        # Aggregate results
        final_result = await root_manager.aaggregate_results(worker_results)
        return final_result


class HolarchyTopology(BaseNetworkTopology):
    """Holarchy topology implementation"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__(config)
        self.holons: List[HolonAgent] = []
        self.holon_hierarchy: Dict[str, List[str]] = {}
    
    def create_agents(self, system_prompts: List[str], model_type: str = "gpt-4o-mini", agent_roles: List[AgentRole] = None, hierarchy_info: Dict[str, dict] = None):
        """Create holon agents with hierarchy information"""
        num_agents = min(len(system_prompts), self.config.num_agents)
        
        for i in range(num_agents):
            agent_id = f"holon_agent_{i}"
            agent = HolonAgent(
                agent_id=agent_id,
                system_prompt=system_prompts[i],
                model_type=model_type
            )
            self.agents[agent_id] = agent
            self.holons.append(agent)
            # Use provided role or default to HOLON
            if agent_roles and i < len(agent_roles):
                role = agent_roles[i]
                self.agent_roles[agent_id] = role if role == AgentRole.ATTACKER else AgentRole.HOLON
            else:
                self.agent_roles[agent_id] = AgentRole.HOLON
            
            # Update hierarchy information if provided
            if hierarchy_info and agent_id in hierarchy_info:
                agent.update_hierarchy_info(hierarchy_info[agent_id])
    
    def setup_connections(self):
        """Setup holarchy connections"""
        self.network_graph = nx.Graph()
        
        # Add nodes
        for agent_id in self.agents.keys():
            self.network_graph.add_node(agent_id)
        
        # Create hierarchical holon structure
        self._build_holon_hierarchy()
        
        # Setup connections based on hierarchy
        for super_holon_id, sub_holon_ids in self.holon_hierarchy.items():
            super_holon = self.agents[super_holon_id]
            for sub_holon_id in sub_holon_ids:
                sub_holon = self.agents[sub_holon_id]
                super_holon.add_sub_holon(sub_holon)
                
                # Add edge in graph
                self.network_graph.add_edge(super_holon_id, sub_holon_id)
        
        # Add horizontal connections between holons at same level
        self._add_horizontal_connections()
    
    def _build_holon_hierarchy(self):
        """Build holon hierarchy structure"""
        holon_ids = list(self.agents.keys())
        num_holons = len(holon_ids)
        
        # Simple hierarchical structure
        if num_holons <= 1:
            return
        
        # Create 2-level hierarchy
        num_super_holons = max(1, num_holons // 3)
        super_holon_ids = holon_ids[:num_super_holons]
        sub_holon_ids = holon_ids[num_super_holons:]
        
        for super_holon_id in super_holon_ids:
            self.holon_hierarchy[super_holon_id] = []
        
        # Distribute sub-holons to super-holons
        for i, sub_holon_id in enumerate(sub_holon_ids):
            super_holon_idx = i % len(super_holon_ids)
            super_holon_id = super_holon_ids[super_holon_idx]
            self.holon_hierarchy[super_holon_id].append(sub_holon_id)
    
    def _add_horizontal_connections(self):
        """Add horizontal connections between holons at same level"""
        # Add connections between super-holons
        super_holon_ids = list(self.holon_hierarchy.keys())
        for i in range(len(super_holon_ids)):
            for j in range(i + 1, len(super_holon_ids)):
                holon1 = self.agents[super_holon_ids[i]]
                holon2 = self.agents[super_holon_ids[j]]
                holon1.add_neighbor(holon2)
                holon2.add_neighbor(holon1)
                
                # Add edge in graph
                self.network_graph.add_edge(super_holon_ids[i], super_holon_ids[j])
    
    def execute_holarchic_task(self, task: Any) -> Any:
        """Execute task using holarchic approach"""
        if not self.holons:
            return None
        
        # Start with autonomous decisions
        autonomous_results = []
        for holon in self.holons:
            result = holon.autonomous_decision(task)
            autonomous_results.append(result)
        
        # Cooperative decision making
        if len(self.holons) > 1:
            # Use first holon as coordinator
            coordinator = self.holons[0]
            other_holons = self.holons[1:]
            final_result = coordinator.cooperative_decision(task, other_holons)
        else:
            final_result = autonomous_results[0] if autonomous_results else None
        
        return final_result
    
    async def aexecute_holarchic_task(self, task: Any) -> Any:
        """Asynchronous holarchic task execution"""
        if not self.holons:
            return None
        
        # Start with autonomous decisions
        tasks = []
        for holon in self.holons:
            tasks.append(asyncio.create_task(
                holon.aautonomous_decision(task)
            ))
        
        autonomous_results = await asyncio.gather(*tasks, return_exceptions=True)
        autonomous_results = [r for r in autonomous_results if not isinstance(r, Exception)]
        
        # Cooperative decision making
        if len(self.holons) > 1:
            coordinator = self.holons[0]
            other_holons = self.holons[1:]
            final_result = await coordinator.acooperative_decision(task, other_holons)
        else:
            final_result = autonomous_results[0] if autonomous_results else None
        
        return final_result


class P2PTopology(BaseNetworkTopology):
    """P2P topology implementation"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__(config)
        self.peers: List[PeerAgent] = []
        self.connection_type = config.p2p_connection_type
    
    def create_agents(self, system_prompts: List[str], model_type: str = "gpt-4o-mini", agent_roles: List[AgentRole] = None, hierarchy_info: Dict[str, dict] = None):
        """Create P2P peer agents with hierarchy information"""
        num_agents = min(len(system_prompts), self.config.num_agents)
        
        for i in range(num_agents):
            agent_id = f"peer_agent_{i}"
            agent = PeerAgent(
                agent_id=agent_id,
                system_prompt=system_prompts[i],
                model_type=model_type
            )
            self.agents[agent_id] = agent
            self.peers.append(agent)
            # Use provided role or default to PEER
            if agent_roles and i < len(agent_roles):
                role = agent_roles[i]
                self.agent_roles[agent_id] = role if role == AgentRole.ATTACKER else AgentRole.PEER
            else:
                self.agent_roles[agent_id] = AgentRole.PEER
            
            # Update hierarchy information if provided
            if hierarchy_info and agent_id in hierarchy_info:
                agent.update_hierarchy_info(hierarchy_info[agent_id])
    
    def setup_connections(self):
        """Setup P2P connections"""
        self.network_graph = nx.Graph()
        
        # Add nodes
        for agent_id in self.agents.keys():
            self.network_graph.add_node(agent_id)
        
        peer_ids = list(self.agents.keys())
        
        if self.connection_type == "fully_connected":
            self._setup_fully_connected(peer_ids)
        elif self.connection_type == "ring":
            self._setup_ring_topology(peer_ids)
        else:  # mesh
            self._setup_mesh_topology(peer_ids)
    
    def _setup_fully_connected(self, peer_ids: List[str]):
        """Setup fully connected topology"""
        for i in range(len(peer_ids)):
            for j in range(i + 1, len(peer_ids)):
                peer1 = self.agents[peer_ids[i]]
                peer2 = self.agents[peer_ids[j]]
                peer1.add_peer(peer2)
                peer2.add_peer(peer1)
                
                # Add edge in graph
                self.network_graph.add_edge(peer_ids[i], peer_ids[j])
    
    def _setup_ring_topology(self, peer_ids: List[str]):
        """Setup ring topology"""
        for i in range(len(peer_ids)):
            peer1 = self.agents[peer_ids[i]]
            peer2 = self.agents[peer_ids[(i + 1) % len(peer_ids)]]
            peer1.add_peer(peer2)
            peer2.add_peer(peer1)
            
            # Add edge in graph
            self.network_graph.add_edge(peer_ids[i], peer_ids[(i + 1) % len(peer_ids)])
    
    def _setup_mesh_topology(self, peer_ids: List[str]):
        """Setup mesh topology with limited connections"""
        num_peers = len(peer_ids)
        max_connections = max(2, int(num_peers * self.config.sparsity))
        
        for i in range(len(peer_ids)):
            peer1 = self.agents[peer_ids[i]]
            
            # Connect to random peers
            connections = 0
            attempts = 0
            while connections < max_connections and attempts < num_peers * 2:
                j = random.randint(0, num_peers - 1)
                if i != j and peer_ids[j] not in peer1.peers:
                    peer2 = self.agents[peer_ids[j]]
                    peer1.add_peer(peer2)
                    peer2.add_peer(peer1)
                    
                    # Add edge in graph
                    self.network_graph.add_edge(peer_ids[i], peer_ids[j])
                    connections += 1
                attempts += 1
    
    def execute_p2p_task(self, task: Any, query_type: str = "broadcast") -> List[Any]:
        """Execute task using P2P approach"""
        if not self.peers:
            return []
        
        results = []
        if query_type == "broadcast":
            # Use first peer as initiator
            initiator = self.peers[0]
            results = initiator.broadcast_query(str(task))
        else:
            # Individual peer processing
            for peer in self.peers:
                result = peer.handle_query(str(task))
                results.append(result)
        
        return results
    
    async def aexecute_p2p_task(self, task: Any, query_type: str = "broadcast") -> List[Any]:
        """Asynchronous P2P task execution"""
        if not self.peers:
            return []
        
        if query_type == "broadcast":
            initiator = self.peers[0]
            results = await initiator.abroadcast_query(str(task))
        else:
            tasks = []
            for peer in self.peers:
                tasks.append(asyncio.create_task(
                    peer.ahandle_query(str(task))
                ))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            results = [r for r in results if not isinstance(r, Exception)]
        
        return results


class HybridTopology(BaseNetworkTopology):
    """Hybrid topology implementation"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__(config)
        self.coordinators: List[HybridAgent] = []
        self.centralized_agents: List[BaseAgent] = []
        self.p2p_agents: List[BaseAgent] = []
        self.centralization_ratio = config.hybrid_centralization_ratio
    
    def create_agents(self, system_prompts: List[str], model_type: str = "gpt-4o-mini", agent_roles: List[AgentRole] = None, hierarchy_info: Dict[str, dict] = None):
        """Create hybrid agents with hierarchy information"""
        num_agents = min(len(system_prompts), self.config.num_agents)
        
        # Create coordinators
        num_coordinators = max(1, int(num_agents * self.centralization_ratio))
        for i in range(num_coordinators):
            agent_id = f"hybrid_coordinator_{i}"
            agent = HybridAgent(
                agent_id=agent_id,
                system_prompt=system_prompts[i],
                model_type=model_type
            )
            self.agents[agent_id] = agent
            self.coordinators.append(agent)
            # Use provided role or default to COORDINATOR
            if agent_roles and i < len(agent_roles):
                role = agent_roles[i]
                self.agent_roles[agent_id] = role if role == AgentRole.ATTACKER else AgentRole.COORDINATOR
            else:
                self.agent_roles[agent_id] = AgentRole.COORDINATOR
            
            # Update hierarchy information if provided
            if hierarchy_info and agent_id in hierarchy_info:
                agent.update_hierarchy_info(hierarchy_info[agent_id])
        
        # Create other agents (mix of different types)
        for i in range(num_coordinators, num_agents):
            agent_id = f"hybrid_agent_{i}"
            
            # Randomly assign agent type
            if random.random() < 0.5:
                agent = WorkerAgent(
                    agent_id=agent_id,
                    system_prompt=system_prompts[i],
                    model_type=model_type
                )
                self.centralized_agents.append(agent)
                # Use provided role or default to WORKER
                if agent_roles and i < len(agent_roles):
                    role = agent_roles[i]
                    self.agent_roles[agent_id] = role if role == AgentRole.ATTACKER else AgentRole.WORKER
                else:
                    self.agent_roles[agent_id] = AgentRole.WORKER
            else:
                agent = PeerAgent(
                    agent_id=agent_id,
                    system_prompt=system_prompts[i],
                    model_type=model_type
                )
                self.p2p_agents.append(agent)
                # Use provided role or default to PEER
                if agent_roles and i < len(agent_roles):
                    role = agent_roles[i]
                    self.agent_roles[agent_id] = role if role == AgentRole.ATTACKER else AgentRole.PEER
                else:
                    self.agent_roles[agent_id] = AgentRole.PEER
            
            self.agents[agent_id] = agent
            
            # Update hierarchy information if provided
            if hierarchy_info and agent_id in hierarchy_info:
                agent.update_hierarchy_info(hierarchy_info[agent_id])
    
    def setup_connections(self):
        """Setup hybrid connections"""
        self.network_graph = nx.Graph()
        
        # Add nodes
        for agent_id in self.agents.keys():
            self.network_graph.add_node(agent_id)
        
        # Setup coordinator-centralized agent connections
        for i, coordinator in enumerate(self.coordinators):
            start_idx = i * len(self.centralized_agents) // len(self.coordinators)
            end_idx = (i + 1) * len(self.centralized_agents) // len(self.coordinators)
            
            for j in range(start_idx, end_idx):
                if j < len(self.centralized_agents):
                    agent = self.centralized_agents[j]
                    coordinator.add_centralized_peer(agent)
                    agent.add_neighbor(coordinator)
                    
                    # Add edge in graph
                    self.network_graph.add_edge(coordinator.agent_id, agent.agent_id)
        
        # Setup P2P connections between P2P agents
        for i in range(len(self.p2p_agents)):
            for j in range(i + 1, len(self.p2p_agents)):
                if random.random() < self.config.sparsity:
                    peer1 = self.p2p_agents[i]
                    peer2 = self.p2p_agents[j]
                    peer1.add_peer(peer2)
                    peer2.add_peer(peer1)
                    
                    # Add edge in graph
                    self.network_graph.add_edge(peer1.agent_id, peer2.agent_id)
        
        # Connect coordinators to P2P agents
        for coordinator in self.coordinators:
            for p2p_agent in self.p2p_agents:
                if random.random() < 0.3:  # 30% connection probability
                    coordinator.add_p2p_peer(p2p_agent)
                    p2p_agent.add_neighbor(coordinator)
                    
                    # Add edge in graph
                    self.network_graph.add_edge(coordinator.agent_id, p2p_agent.agent_id)
    
    def execute_hybrid_task(self, task: Any, context: Dict[str, Any] = None) -> Any:
        """Execute task using hybrid approach"""
        if not self.coordinators:
            return None
        
        if context is None:
            context = {}
        
        # Use first coordinator
        coordinator = self.coordinators[0]
        result = coordinator.adaptive_coordination(task, context)
        return result
    
    async def aexecute_hybrid_task(self, task: Any, context: Dict[str, Any] = None) -> Any:
        """Asynchronous hybrid task execution"""
        if not self.coordinators:
            return None
        
        if context is None:
            context = {}
        
        coordinator = self.coordinators[0]
        result = await coordinator.aadaptive_coordination(task, context)
        return result


class NetworkTopologyFactory:
    """Factory for creating network topologies"""
    
    @staticmethod
    def create_topology(config: NetworkConfig) -> BaseNetworkTopology:
        """Create topology based on configuration"""
        # Handle enum comparison issues by using string comparison
        topology_type_str = config.topology_type.value if hasattr(config.topology_type, 'value') else str(config.topology_type)
        
        if topology_type_str == "linear":
            return LinearPipelineTopology(config)
        elif topology_type_str == "tree_hierarchy":
            return TreeHierarchyTopology(config)
        elif topology_type_str == "holarchy":
            return HolarchyTopology(config)
        elif topology_type_str == "p2p_flat":
            return P2PTopology(config)
        elif topology_type_str == "p2p_structured":
            return P2PTopology(config)
        elif topology_type_str == "hybrid":
            return HybridTopology(config)
        else:
            raise ValueError(f"Unknown topology type: {config.topology_type}") 