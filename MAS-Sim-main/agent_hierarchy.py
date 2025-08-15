"""
Agent Hierarchy System for NewMA
Defines hierarchical roles and responsibilities based on network topology
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import networkx as nx


class HierarchyLevel(Enum):
    """Hierarchy level enumeration"""
    ROOT = "root"           # Top level (e.g., CEO, main coordinator)
    EXECUTIVE = "executive"  # High level (e.g., VP, senior manager)
    MANAGER = "manager"      # Middle level (e.g., team lead, supervisor)
    WORKER = "worker"        # Operational level (e.g., specialist, operator)
    PEER = "peer"           # Equal level (e.g., colleague, team member)
    LEAF = "leaf"           # End level (e.g., individual contributor)


class HierarchyRole(Enum):
    """Hierarchy role enumeration"""
    # Leadership roles
    COORDINATOR = "coordinator"      # Main coordinator/leader
    SUPERVISOR = "supervisor"        # Oversees others
    LEADER = "leader"               # Team leader
    
    # Operational roles
    SPECIALIST = "specialist"        # Domain expert
    OPERATOR = "operator"           # Task executor
    ANALYST = "analyst"             # Data processor
    
    # Support roles
    ADVISOR = "advisor"             # Provides guidance
    VALIDATOR = "validator"         # Verifies results
    FACILITATOR = "facilitator"     # Enables communication
    
    # Peer roles
    COLLEAGUE = "colleague"         # Equal peer
    PARTNER = "partner"             # Collaborative peer
    
    # Special roles
    GATEWAY = "gateway"             # Communication hub
    BRIDGE = "bridge"               # Connects different groups
    ISOLATED = "isolated"           # Standalone agent


@dataclass
class HierarchyInfo:
    """Hierarchy information for an agent"""
    level: HierarchyLevel
    role: HierarchyRole
    authority_level: int = 0  # 0 = lowest, higher numbers = more authority
    subordinates: List[str] = field(default_factory=list)
    supervisors: List[str] = field(default_factory=list)
    peers: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchyManager:
    """Manages agent hierarchies based on network topology"""
    
    def __init__(self):
        self.hierarchies: Dict[str, HierarchyInfo] = {}
        self.network_graph: Optional[nx.Graph] = None
    
    def analyze_topology_hierarchy(self, 
                                 topology_type: str,
                                 network_graph: nx.Graph,
                                 agent_ids: List[str]) -> Dict[str, HierarchyInfo]:
        """Analyze network topology and assign hierarchical roles"""
        self.network_graph = network_graph
        
        # Map topology types to analysis methods
        if topology_type == "linear":
            return self._analyze_linear_hierarchy(agent_ids)
        elif topology_type == "tree_hierarchy":
            return self._analyze_tree_hierarchy(agent_ids)
        elif topology_type == "holarchy":
            return self._analyze_holarchy_hierarchy(agent_ids)
        elif topology_type == "p2p_flat":
            return self._analyze_p2p_flat_hierarchy(agent_ids)
        elif topology_type == "p2p_structured":
            return self._analyze_p2p_structured_hierarchy(agent_ids)
        elif topology_type == "hybrid":
            return self._analyze_hybrid_hierarchy(agent_ids)
        else:
            return self._analyze_default_hierarchy(agent_ids)
    
    def _analyze_linear_hierarchy(self, agent_ids: List[str]) -> Dict[str, HierarchyInfo]:
        """Analyze linear pipeline topology hierarchy"""
        hierarchies = {}
        
        # In linear topology, agents are arranged in a sequence
        # First agent is the initiator, last agent is the final processor
        for i, agent_id in enumerate(agent_ids):
            if i == 0:
                # First agent - initiator
                hierarchies[agent_id] = HierarchyInfo(
                    level=HierarchyLevel.MANAGER,
                    role=HierarchyRole.COORDINATOR,
                    authority_level=2,
                    subordinates=[agent_ids[i+1]] if i+1 < len(agent_ids) else [],
                    responsibilities=["Initiate processing", "Coordinate pipeline flow", "Manage input data"],
                    permissions=["Can initiate tasks", "Can coordinate with next stage", "Can manage data flow"]
                )
            elif i == len(agent_ids) - 1:
                # Last agent - final processor
                hierarchies[agent_id] = HierarchyInfo(
                    level=HierarchyLevel.WORKER,
                    role=HierarchyRole.OPERATOR,
                    authority_level=1,
                    supervisors=[agent_ids[i-1]],
                    responsibilities=["Final processing", "Output generation", "Quality assurance"],
                    permissions=["Can process final stage", "Can generate output", "Can report results"]
                )
            else:
                # Middle agents - processors
                hierarchies[agent_id] = HierarchyInfo(
                    level=HierarchyLevel.WORKER,
                    role=HierarchyRole.ANALYST,
                    authority_level=1,
                    supervisors=[agent_ids[i-1]],
                    subordinates=[agent_ids[i+1]],
                    responsibilities=["Process intermediate data", "Pass results to next stage", "Maintain pipeline integrity"],
                    permissions=["Can process data", "Can pass to next stage", "Can report progress"]
                )
        
        return hierarchies
    
    def _analyze_holarchy_hierarchy(self, agent_ids: List[str]) -> Dict[str, HierarchyInfo]:
        """Analyze holarchy topology hierarchy"""
        hierarchies = {}
        
        # In holarchy, agents form autonomous units that can act independently
        # but also cooperate at higher levels
        for agent_id in agent_ids:
            neighbors = list(self.network_graph.neighbors(agent_id))
            
            # Determine if this is a super-holon (higher level coordinator)
            if len(neighbors) > len(agent_ids) * 0.3:  # High connectivity indicates super-holon
                hierarchies[agent_id] = HierarchyInfo(
                    level=HierarchyLevel.MANAGER,
                    role=HierarchyRole.COORDINATOR,
                    authority_level=2,
                    subordinates=neighbors,
                    responsibilities=["Coordinate autonomous units", "Bridge different holons", "Enable cooperation"],
                    permissions=["Can coordinate holons", "Can bridge different areas", "Can enable cooperation"]
                )
            else:
                # Regular holon
                hierarchies[agent_id] = HierarchyInfo(
                    level=HierarchyLevel.WORKER,
                    role=HierarchyRole.SPECIALIST,
                    authority_level=1,
                    supervisors=[n for n in neighbors if len(list(self.network_graph.neighbors(n))) > len(agent_ids) * 0.3],
                    peers=[n for n in neighbors if len(list(self.network_graph.neighbors(n))) <= len(agent_ids) * 0.3],
                    responsibilities=["Autonomous decision making", "Specialized expertise", "Cooperative work"],
                    permissions=["Can make autonomous decisions", "Can cooperate with peers", "Can access specialized resources"]
                )
        
        return hierarchies
    
    def _analyze_tree_hierarchy(self, agent_ids: List[str]) -> Dict[str, HierarchyInfo]:
        """Analyze tree topology hierarchy"""
        hierarchies = {}
        
        # Find root node (node with highest betweenness centrality)
        betweenness = nx.betweenness_centrality(self.network_graph)
        root_node = max(betweenness, key=betweenness.get)
        
        # Calculate levels using BFS
        levels = self._calculate_tree_levels(root_node)
        
        for agent_id in agent_ids:
            level_num = levels.get(agent_id, 0)
            
            if level_num == 0:
                # Root level
                hierarchies[agent_id] = HierarchyInfo(
                    level=HierarchyLevel.ROOT,
                    role=HierarchyRole.COORDINATOR,
                    authority_level=3,
                    subordinates=[aid for aid in agent_ids if levels.get(aid, 0) == 1],
                    responsibilities=["Strategic planning", "Resource allocation", "Final decision making"],
                    permissions=["Can assign tasks to all levels", "Can override any decision", "Can access all information"]
                )
            elif level_num == 1:
                # Manager level
                children = [n for n in self.network_graph.neighbors(agent_id) if levels.get(n, 0) > level_num]
                hierarchies[agent_id] = HierarchyInfo(
                    level=HierarchyLevel.MANAGER,
                    role=HierarchyRole.SUPERVISOR,
                    authority_level=2,
                    subordinates=children,
                    supervisors=[root_node],
                    responsibilities=["Team management", "Task delegation", "Progress monitoring"],
                    permissions=["Can assign tasks to subordinates", "Can report to superiors", "Can coordinate with peers"]
                )
            else:
                # Worker level
                parent = [n for n in self.network_graph.neighbors(agent_id) if levels.get(n, 0) < level_num][0]
                hierarchies[agent_id] = HierarchyInfo(
                    level=HierarchyLevel.WORKER,
                    role=HierarchyRole.OPERATOR,
                    authority_level=1,
                    supervisors=[parent],
                    responsibilities=["Task execution", "Specialized work", "Reporting to supervisor"],
                    permissions=["Can request resources", "Can report progress", "Can collaborate with peers"]
                )
        
        return hierarchies
    
    def _analyze_p2p_flat_hierarchy(self, agent_ids: List[str]) -> Dict[str, HierarchyInfo]:
        """Analyze P2P flat topology hierarchy"""
        hierarchies = {}
        
        # In P2P flat topology, all agents are equal peers
        for agent_id in agent_ids:
            neighbors = list(self.network_graph.neighbors(agent_id))
            
            hierarchies[agent_id] = HierarchyInfo(
                level=HierarchyLevel.PEER,
                role=HierarchyRole.PARTNER,
                authority_level=2,
                peers=neighbors,
                responsibilities=["Equal collaboration", "Resource sharing", "Distributed decision making"],
                permissions=["Can communicate with peers", "Can share resources", "Can participate equally"]
            )
        
        return hierarchies
    
    def _analyze_p2p_structured_hierarchy(self, agent_ids: List[str]) -> Dict[str, HierarchyInfo]:
        """Analyze P2P structured topology hierarchy"""
        hierarchies = {}
        
        # In P2P structured topology, agents may have different roles based on connectivity
        for agent_id in agent_ids:
            neighbors = list(self.network_graph.neighbors(agent_id))
            degree = len(neighbors)
            
            # High-degree nodes act as facilitators
            if degree > len(agent_ids) * 0.4:
                hierarchies[agent_id] = HierarchyInfo(
                    level=HierarchyLevel.MANAGER,
                    role=HierarchyRole.FACILITATOR,
                    authority_level=2,
                    subordinates=neighbors,
                    responsibilities=["Facilitate communication", "Bridge different groups", "Enable coordination"],
                    permissions=["Can facilitate communication", "Can bridge groups", "Can coordinate activities"]
                )
            else:
                # Regular peer
                hierarchies[agent_id] = HierarchyInfo(
                    level=HierarchyLevel.PEER,
                    role=HierarchyRole.COLLEAGUE,
                    authority_level=1,
                    supervisors=[n for n in neighbors if len(list(self.network_graph.neighbors(n))) > len(agent_ids) * 0.4],
                    peers=[n for n in neighbors if len(list(self.network_graph.neighbors(n))) <= len(agent_ids) * 0.4],
                    responsibilities=["Collaborative work", "Information sharing", "Peer-to-peer communication"],
                    permissions=["Can communicate with peers", "Can share information", "Can collaborate"]
                )
        
        return hierarchies
    
    def _analyze_hybrid_hierarchy(self, agent_ids: List[str]) -> Dict[str, HierarchyInfo]:
        """Analyze hybrid topology hierarchy"""
        hierarchies = {}
        
        # Find central nodes (high degree nodes)
        degrees = dict(self.network_graph.degree())
        central_nodes = [aid for aid in agent_ids if degrees[aid] > len(agent_ids) * 0.3]
        
        for agent_id in agent_ids:
            if agent_id in central_nodes:
                # Central nodes are managers
                hierarchies[agent_id] = HierarchyInfo(
                    level=HierarchyLevel.MANAGER,
                    role=HierarchyRole.LEADER,
                    authority_level=2,
                    subordinates=[aid for aid in agent_ids if aid not in central_nodes],
                    responsibilities=["Coordinate subgroups", "Bridge different areas", "Manage local decisions"],
                    permissions=["Can coordinate with other managers", "Can assign tasks to subordinates", "Can make local decisions"]
                )
            else:
                # Peripheral nodes are workers
                supervisors = [n for n in self.network_graph.neighbors(agent_id) if n in central_nodes]
                hierarchies[agent_id] = HierarchyInfo(
                    level=HierarchyLevel.WORKER,
                    role=HierarchyRole.SPECIALIST,
                    authority_level=1,
                    supervisors=supervisors,
                    responsibilities=["Specialized tasks", "Local expertise", "Collaboration with peers"],
                    permissions=["Can communicate with supervisors", "Can collaborate with peers", "Can request resources"]
                )
        
        return hierarchies
    
    def _analyze_default_hierarchy(self, agent_ids: List[str]) -> Dict[str, HierarchyInfo]:
        """Default hierarchy analysis"""
        hierarchies = {}
        
        for agent_id in agent_ids:
            hierarchies[agent_id] = HierarchyInfo(
                level=HierarchyLevel.PEER,
                role=HierarchyRole.COLLEAGUE,
                authority_level=1,
                peers=agent_ids,
                responsibilities=["General tasks", "Collaboration", "Information sharing"],
                permissions=["Can communicate with others", "Can participate in decisions"]
            )
        
        return hierarchies
    
    def _calculate_tree_levels(self, root_node: str) -> Dict[str, int]:
        """Calculate tree levels using BFS"""
        levels = {root_node: 0}
        queue = [(root_node, 0)]
        visited = {root_node}
        
        while queue:
            node, level = queue.pop(0)
            
            for neighbor in self.network_graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    levels[neighbor] = level + 1
                    queue.append((neighbor, level + 1))
        
        return levels
    
    def get_hierarchy_description(self, agent_id: str) -> str:
        """Get human-readable hierarchy description for an agent"""
        if agent_id not in self.hierarchies:
            return "Unknown hierarchy"
        
        hierarchy = self.hierarchies[agent_id]
        
        description = f"You are a {hierarchy.role.value} at the {hierarchy.level.value} level "
        description += f"with authority level {hierarchy.authority_level}. "
        
        if hierarchy.subordinates:
            description += f"You supervise {len(hierarchy.subordinates)} subordinates. "
        
        if hierarchy.supervisors:
            description += f"You report to {len(hierarchy.supervisors)} supervisor(s). "
        
        if hierarchy.peers:
            description += f"You collaborate with {len(hierarchy.peers)} peers. "
        
        if hierarchy.responsibilities:
            description += f"Your responsibilities include: {', '.join(hierarchy.responsibilities)}. "
        
        if hierarchy.permissions:
            description += f"Your permissions include: {', '.join(hierarchy.permissions)}."
        
        return description
    
    def get_network_hierarchy_summary(self) -> str:
        """Get summary of network hierarchy"""
        if not self.hierarchies:
            return "No hierarchy information available"
        
        level_counts = {}
        role_counts = {}
        
        for hierarchy in self.hierarchies.values():
            level_counts[hierarchy.level.value] = level_counts.get(hierarchy.level.value, 0) + 1
            role_counts[hierarchy.role.value] = role_counts.get(hierarchy.role.value, 0) + 1
        
        summary = "Network Hierarchy Summary:\n"
        summary += f"Total agents: {len(self.hierarchies)}\n"
        summary += "Level distribution:\n"
        for level, count in level_counts.items():
            summary += f"  {level}: {count} agents\n"
        summary += "Role distribution:\n"
        for role, count in role_counts.items():
            summary += f"  {role}: {count} agents\n"
        
        return summary
    
    def get_agent_hierarchy(self, agent_id: str, network_graph: nx.Graph) -> Optional[HierarchyInfo]:
        """Get hierarchy information for a specific agent"""
        # First check if we already have hierarchy information for this agent
        if hasattr(self, 'hierarchies') and agent_id in self.hierarchies:
            return self.hierarchies[agent_id]
        
        # If not, analyze the topology to get hierarchy information
        if network_graph is None:
            return None
        
        # Determine topology type based on network structure
        topology_type = self._determine_topology_type(network_graph)
        
        # Get all agent IDs from the graph
        agent_ids = list(network_graph.nodes())
        
        # Analyze hierarchy for the entire topology
        hierarchies = self.analyze_topology_hierarchy(topology_type, network_graph, agent_ids)
        
        # Return the hierarchy for the specific agent
        return hierarchies.get(agent_id)
    
    def _determine_topology_type(self, network_graph: nx.Graph) -> str:
        """Determine topology type based on network structure"""
        if network_graph is None or network_graph.number_of_nodes() == 0:
            return "unknown"
        
        num_nodes = network_graph.number_of_nodes()
        num_edges = network_graph.number_of_edges()
        
        # Calculate network metrics
        degrees = dict(network_graph.degree())
        max_degree = max(degrees.values()) if degrees else 0
        min_degree = min(degrees.values()) if degrees else 0
        
        # Linear topology: chain-like structure with most nodes having degree 2
        if max_degree <= 2 and min_degree >= 1:
            return "linear"
        
        # Tree hierarchy: hierarchical structure with no cycles
        if nx.is_tree(network_graph):
            return "tree_hierarchy"
        
        # Holarchy: complex hierarchical structure with some cycles
        if network_graph.is_directed():
            # For directed graphs, check weak connectivity
            if nx.is_weakly_connected(network_graph) and not nx.is_tree(network_graph):
                # Check for holarchy characteristics (autonomous units with cooperation)
                try:
                    avg_clustering = nx.average_clustering(network_graph)
                    if avg_clustering > 0.3:  # High clustering indicates holarchy
                        return "holarchy"
                except:
                    pass
        else:
            # For undirected graphs
            if nx.is_connected(network_graph) and not nx.is_tree(network_graph):
                # Check for holarchy characteristics (autonomous units with cooperation)
                try:
                    avg_clustering = nx.average_clustering(network_graph)
                    if avg_clustering > 0.3:  # High clustering indicates holarchy
                        return "holarchy"
                except:
                    pass
        
        # P2P flat: all nodes have similar degree (flat structure)
        degree_variance = sum((d - sum(degrees.values())/len(degrees))**2 for d in degrees.values()) / len(degrees)
        if degree_variance < 2.0:  # Low variance indicates flat structure
            return "p2p_flat"
        
        # P2P structured: some nodes have higher degree (structured but not hierarchical)
        if max_degree > 2 and min_degree >= 1:
            return "p2p_structured"
        
        # Hybrid: mixed characteristics
        return "hybrid" 
        import networkx as nx
import matplotlib.pyplot as plt

# 假设你已经定义了 agents 和图
agent_ids = ['A', 'B', 'C', 'D', 'E', 'F']
G = nx.Graph()

# 添加节点
G.add_nodes_from(agent_ids)

# 添加边（注意别遗漏或错写）
edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F')]  # linear
G.add_edges_from(edges)

# 可视化
nx.draw(G, with_labels=True)
plt.show()
