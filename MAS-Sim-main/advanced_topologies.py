"""
Advanced Network Topologies
Implements Holarchy, P2P, and Hybrid topologies
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple, Union
import asyncio
import random
import json

from NewMA.core_agent import (
    BaseAgent, HolonAgent, PeerAgent, HybridAgent, AgentRole, AgentType
)
from NewMA.network_topologies import BaseNetworkTopology, NetworkConfig, TopologyType


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
                from NewMA.core_agent import WorkerAgent
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