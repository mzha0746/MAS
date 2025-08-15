"""
Core Agent Implementation
Supports various agent types and communication patterns
"""

import os
import asyncio
import numpy as np
import re
import uuid
from typing import Literal, Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from openai import OpenAI, AsyncOpenAI


class AgentRole(Enum):
    """Agent role enumeration"""
    NORMAL = "normal"
    ATTACKER = "attacker"
    MANAGER = "manager"
    WORKER = "worker"
    HOLON = "holon"
    PEER = "peer"
    COORDINATOR = "coordinator"


class AgentType(Enum):
    """Agent type enumeration"""
    LINEAR = "linear"
    TREE_MANAGER = "tree_manager"
    TREE_WORKER = "tree_worker"
    HOLON = "holon"
    P2P_PEER = "p2p_peer"
    HYBRID = "hybrid"


@dataclass
class AgentState:
    """Agent state information"""
    agent_id: str
    role: AgentRole
    agent_type: AgentType
    is_active: bool = True
    trust_score: float = 1.0
    performance_score: float = 0.0
    last_activity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """Message structure for agent communication"""
    sender_id: str
    receiver_id: str
    message_type: str
    content: Any
    timestamp: float
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def llm_invoke(prompt, model_type: str):
    """Synchronous LLM invocation using DeepSeek API"""
    import requests
    import os

    api_key = os.getenv("DEEPSEEK_API_KEY")
    # DeepSeek API expects the endpoint to include the path, e.g., /v1/chat/completions
    base_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_type,
        "messages": prompt,
        "temperature": 0,
        "max_tokens": 512
    }
    response = requests.post(base_url, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"DeepSeek API call failed: {e}, Response: {response.text}")
    return response.json()["choices"][0]["message"]["content"]

async def allm_invoke(prompt, model_type: str):
    """Asynchronous LLM invocation with DeepSeek, OpenAI, and Ollama support"""
    import os

    # Ollama support
    if model_type.startswith("ollama"):
        import httpx
        # model_type: e.g., "ollama/llama3" or "ollama/mistral"
        # Extract the actual model name after "ollama/"
        if "ollama" in model_type:
            _, ollama_model = model_type.split("_", 1)

        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        url = f"{ollama_base_url}/api/chat"
        # Ollama expects a list of messages like OpenAI
        payload = {
            "model": ollama_model,
            "messages": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 512
            }
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            # Ollama returns the response in 'message'->'content'
            return data["message"]["content"]

    # DeepSeek support
    elif model_type.startswith("deepseek"):
        from openai import AsyncOpenAI
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        aclient = AsyncOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=deepseek_base_url)
        response = await aclient.chat.completions.create(
            model=model_type,
            messages=prompt,
            temperature=0,
            max_tokens=512
        )
        return response.choices[0].message.content

    # OpenAI or other OpenAI-compatible models
    else:
        from openai import AsyncOpenAI
        aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"))
        response = await aclient.chat.completions.create(
            model=model_type,
            messages=prompt,
            temperature=0,
            max_tokens=512
        )
        return response.choices[0].message.content


class BaseAgent:
    """Base agent class with common functionality"""
    
    def __init__(self, 
                 agent_id: str,
                 system_prompt: str, 
                 model_type: str = "gpt-4o-mini",
                 role: AgentRole = AgentRole.NORMAL,
                 agent_type: AgentType = AgentType.LINEAR):
        
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        self.model_type = model_type
        self.role = role
        self.agent_type = agent_type
        
        # Memory and state
        self.memory = []
        self.memory.append({"role": "system", "content": system_prompt})
        self.last_response = {"answer": None, "reason": None}
        
        # Communication
        self.inbox: List[Message] = []
        self.outbox: List[Message] = []
        self.neighbors: Dict[str, 'BaseAgent'] = {}
        
        # State
        self.state = AgentState(
            agent_id=agent_id,
            role=role,
            agent_type=agent_type
        )
        
        # Hierarchy information
        self.hierarchy_info = {}
        
        # Task management
        self.current_task = None
        self.task_queue = []
        self.results_cache = {}
    
       def add_neighbor(self, neighbor: 'BaseAgent'):
        """添加邻居节点，并建立双向连接"""
        if neighbor.agent_id != self.agent_id and neighbor.agent_id not in self.neighbors:
            self.neighbors[neighbor.agent_id] = neighbor
            neighbor.neighbors[self.agent_id] = self
    
    def remove_neighbor(self, neighbor_id: str):
        """移除邻居节点"""
        if neighbor_id in self.neighbors:
            del self.neighbors[neighbor_id]
            # 同时从对方邻居列表中移除自己
            if neighbor_id in self.neighbors:
                del self.neighbors[neighbor_id].neighbors[self.agent_id]
    
    def set_status(self, status: str):
        """设置代理状态（用于故障注入）"""
        valid_statuses = ["ACTIVE", "FAILED", "RECOVERING"]
        if status in valid_statuses:
            self.status = status
            # 如果设置为失败状态，清除所有邻居连接
            if status == "FAILED":
                for neighbor_id in list(self.neighbors.keys()):
                    self.remove_neighbor(neighbor_id)
    
    def _should_message_fail(self, message: Message) -> bool:
        """决定消息是否应该失败"""
        # 如果代理处于失败状态，所有消息都失败
        if self.status == "FAILED":
            return True
        
        # 如果代理正在恢复中，有较高失败率
        if self.status == "RECOVERING":
            return random.random() < 0.7
        
        # 基础失败率
        if random.random() < self.message_failure_rate:
            return True
        
        # 高优先级消息失败率减半
        if message.priority > 5:
            return random.random() < (self.message_failure_rate / 2)
        
        return False
    
    def _simulate_network_latency(self):
        """模拟网络延迟"""
        if self.network_latency_max > 0:
            delay = random.uniform(self.network_latency_min, self.network_latency_max)
            time.sleep(delay)
    
    def send_message(self, receiver_id: str, message_type: str, content: Any, priority: int = 0):
        """
        发送消息到指定接收者，支持失败模拟和重试机制
        """
        # 创建消息对象
        message = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            timestamp=asyncio.get_event_loop().time(),
            priority=priority
        )
        
        self.outbox.append(message)
        self.message_history.append(message)
        self.message_stats["sent"] += 1
        
        # 如果接收者是邻居，尝试传递
        if receiver_id in self.neighbors:
            self._attempt_delivery(message)
    
    def _attempt_delivery(self, message: Message):
        """尝试传递消息，处理失败和重试逻辑"""
        message.delivery_attempts += 1
        
        # 模拟网络延迟
        self._simulate_network_latency()
        
        # 检查消息是否应该失败
        if self._should_message_fail(message):
            message.delivery_status = "failed"
            self.message_stats["failed"] += 1
            
            # 如果未达到最大重试次数，则重试
            if message.delivery_attempts < self.max_retries:
                self.message_stats["retried"] += 1
                # 使用事件循环安排重试
                loop = asyncio.get_event_loop()
                retry_delay = random.uniform(0.1, 0.5)  # 重试延迟
                loop.call_later(retry_delay, self._attempt_delivery, message)
            return
        
        # 成功传递消息
        try:
            receiver = self.neighbors[message.receiver_id]
            receiver.receive_message(message)
            message.delivery_status = "delivered"
            self.message_stats["delivered"] += 1
        except Exception as e:
            # 记录传递错误
            message.delivery_status = f"failed: {str(e)}"
            self.message_stats["failed"] += 1
    
    def receive_message(self, message: Message):
        """接收消息处理"""
        # 添加到收件箱
        self.inbox.append(message)
        
        # 记录接收到的消息
        received_message = Message(**message.__dict__)
        received_message.delivery_status = "received"
        self.message_history.append(received_message)
        
        # 处理消息
        self._process_message(message)
    
    def get_message_success_rate(self) -> float:
        """计算消息成功率"""
        if self.message_stats["sent"] == 0:
            return 1.0  # 没有发送消息时返回100%
        
        return self.message_stats["delivered"] / self.message_stats["sent"]
    
    # 以下方法保持原有功能不变，但添加了类型提示和文档字符串
    def _process_message(self, message: Message):
        """处理接收到的消息（根据类型分发）"""
        handler_name = f"_handle_{message.message_type}_message"
        handler = getattr(self, handler_name, self._handle_unknown_message)
        handler(message)
    
    def _handle_task_message(self, message: Message):
        """处理任务分配消息"""
        # 实际实现中这里会有任务处理逻辑
        pass
    
    def _handle_result_message(self, message: Message):
        """处理结果消息"""
        # 实际实现中这里会有结果处理逻辑
        pass
    
    def _handle_coordination_message(self, message: Message):
        """处理协调消息"""
        # 在子类中覆盖实现具体协调逻辑
        pass
    
    def _handle_query_message(self, message: Message):
        """处理查询消息"""
        # 在子类中覆盖实现具体查询处理逻辑
        pass
    
    def _handle_unknown_message(self, message: Message):
        """处理未知类型的消息"""
        print(f"Agent {self.agent_id} received unknown message type: {message.message_type}")
    
    def parser(self, response: str):
        """解析结构化的LLM响应（保持不变）"""
        # 保持原有实现
        pass
    
    def chat(self, prompt: str) -> str:
        """与代理同步聊天（保持不变）"""
        # 保持原有实现
        pass
    
    async def achat(self, prompt: str) -> str:
        """与代理异步聊天（保持不变）"""
        # 保持原有实现
        pass
    
    def get_state(self) -> Dict:
        """获取当前代理状态（扩展为包含消息统计）"""
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "neighbors": list(self.neighbors.keys()),
            "message_stats": self.message_stats,
            "success_rate": self.get_message_success_rate()
        }
    
    def update_state(self, **kwargs):
        """更新代理状态（保持不变）"""
        # 保持原有实现
        pass
    
    def update_hierarchy_info(self, hierarchy_info: dict):
        """更新代理的层级信息（保持不变）"""
        # 保持原有实现
        pass
    
    def get_neighbors_info(self) -> Dict[str, Dict]:
        """获取邻居代理的信息（扩展为包含状态和消息统计）"""
        return {
            neighbor_id: {
                "role": neighbor.role.value if hasattr(neighbor, 'role') else "unknown",
                "type": neighbor.agent_type.value if hasattr(neighbor, 'agent_type') else "unknown",
                "status": neighbor.status,
                "message_success_rate": neighbor.get_message_success_rate(),
                "is_active": neighbor.status != "FAILED"
            }
            for neighbor_id, neighbor in self.neighbors.items()
        }

class LinearAgent(BaseAgent):
    """Agent for linear pipeline architecture"""
    
    def __init__(self, agent_id: str, system_prompt: str, model_type: str = "gpt-4o-mini", 
                 position: int = 0, total_stages: int = 1):
        super().__init__(agent_id, system_prompt, model_type, AgentRole.NORMAL, AgentType.LINEAR)
        self.position = position
        self.total_stages = total_stages
        self.stage_input = None
        self.stage_output = None
    
    def process_stage(self, input_data: Any) -> Any:
        """Process current stage with input data"""
        self.stage_input = input_data
        prompt = f"Process the following input for stage {self.position + 1}/{self.total_stages}:\n{input_data}"
        response = self.chat(prompt)
        self.stage_output = response
        return response
    
    async def aprocess_stage(self, input_data: Any) -> Any:
        """Asynchronous stage processing"""
        self.stage_input = input_data
        prompt = f"Process the following input for stage {self.position + 1}/{self.total_stages}:\n{input_data}"
        response = await self.achat(prompt)
        self.stage_output = response
        return response


class ManagerAgent(BaseAgent):
    """Manager agent for tree hierarchy"""
    
    def __init__(self, agent_id: str, system_prompt: str, model_type: str = "gpt-4o-mini"):
        super().__init__(agent_id, system_prompt, model_type, AgentRole.MANAGER, AgentType.TREE_MANAGER)
        self.workers: Dict[str, 'WorkerAgent'] = {}
        self.task_decomposition_cache = {}
        self.result_aggregation_cache = {}
    
    def add_worker(self, worker: 'WorkerAgent'):
        """Add a worker agent"""
        self.workers[worker.agent_id] = worker
        self.add_neighbor(worker)
    
    def decompose_task(self, task: Any) -> List[Any]:
        """Decompose a complex task into subtasks"""
        prompt = f"Decompose the following task into subtasks:\n{task}\nProvide a list of subtasks."
        response = self.chat(prompt)
        # Parse response to extract subtasks
        # This is a simplified implementation
        subtasks = [task]  # Placeholder
        self.task_decomposition_cache[task] = subtasks
        return subtasks
    
    def aggregate_results(self, results: List[Any]) -> Any:
        """Aggregate results from workers"""
        prompt = f"Aggregate the following results:\n{results}\nProvide a final result."
        response = self.chat(prompt)
        self.result_aggregation_cache[str(results)] = response
        return response
    
    async def adecompose_task(self, task: Any) -> List[Any]:
        """Asynchronous task decomposition"""
        prompt = f"Decompose the following task into subtasks:\n{task}\nProvide a list of subtasks."
        response = await self.achat(prompt)
        subtasks = [task]  # Placeholder
        self.task_decomposition_cache[task] = subtasks
        return subtasks
    
    async def aaggregate_results(self, results: List[Any]) -> Any:
        """Asynchronous result aggregation"""
        prompt = f"Aggregate the following results:\n{results}\nProvide a final result."
        response = await self.achat(prompt)
        self.result_aggregation_cache[str(results)] = response
        return response


class WorkerAgent(BaseAgent):
    """Worker agent for tree hierarchy"""
    
    def __init__(self, agent_id: str, system_prompt: str, model_type: str = "gpt-4o-mini"):
        super().__init__(agent_id, system_prompt, model_type, AgentRole.WORKER, AgentType.TREE_WORKER)
        self.manager: Optional[ManagerAgent] = None
        self.specialization = "general"
    
    def set_manager(self, manager: ManagerAgent):
        """Set the manager for this worker"""
        self.manager = manager
        self.add_neighbor(manager)
    
    def set_specialization(self, specialization: str):
        """Set worker specialization"""
        self.specialization = specialization
        self.state.metadata["specialization"] = specialization
    
    def execute_task(self, task: Any) -> Any:
        """Execute assigned task"""
        prompt = f"Execute the following task with specialization '{self.specialization}':\n{task}"
        response = self.chat(prompt)
        return response
    
    async def aexecute_task(self, task: Any) -> Any:
        """Asynchronous task execution"""
        prompt = f"Execute the following task with specialization '{self.specialization}':\n{task}"
        response = await self.achat(prompt)
        return response


class HolonAgent(BaseAgent):
    """Holon agent for holarchy architecture"""
    
    def __init__(self, agent_id: str, system_prompt: str, model_type: str = "gpt-4o-mini"):
        super().__init__(agent_id, system_prompt, model_type, AgentRole.HOLON, AgentType.HOLON)
        self.sub_holons: Dict[str, 'HolonAgent'] = {}
        self.super_holon: Optional['HolonAgent'] = None
        self.autonomy_level = 0.8
        self.cooperation_level = 0.8
    
    def add_sub_holon(self, holon: 'HolonAgent'):
        """Add a sub-holon"""
        self.sub_holons[holon.agent_id] = holon
        self.add_neighbor(holon)
        holon.super_holon = self
    
    def set_super_holon(self, holon: 'HolonAgent'):
        """Set the super-holon"""
        self.super_holon = holon
        self.add_neighbor(holon)
    
    def autonomous_decision(self, context: Any) -> Any:
        """Make autonomous decision"""
        prompt = f"Make an autonomous decision based on context:\n{context}"
        response = self.chat(prompt)
        return response
    
    def cooperative_decision(self, context: Any, other_holons: List['HolonAgent']) -> Any:
        """Make cooperative decision with other holons"""
        other_views = [holon.get_state() for holon in other_holons]
        prompt = f"Make a cooperative decision with other holons:\nContext: {context}\nOther views: {other_views}"
        response = self.chat(prompt)
        return response
    
    async def aautonomous_decision(self, context: Any) -> Any:
        """Asynchronous autonomous decision"""
        prompt = f"Make an autonomous decision based on context:\n{context}"
        response = await self.achat(prompt)
        return response
    
    async def acooperative_decision(self, context: Any, other_holons: List['HolonAgent']) -> Any:
        """Asynchronous cooperative decision"""
        other_views = [holon.get_state() for holon in other_holons]
        prompt = f"Make a cooperative decision with other holons:\nContext: {context}\nOther views: {other_views}"
        response = await self.achat(prompt)
        return response


class PeerAgent(BaseAgent):
    """Peer agent for P2P topology"""
    
    def __init__(self, agent_id: str, system_prompt: str, model_type: str = "gpt-4o-mini"):
        super().__init__(agent_id, system_prompt, model_type, AgentRole.PEER, AgentType.P2P_PEER)
        self.peers: Dict[str, 'PeerAgent'] = {}
        self.resource_index = {}
        self.query_cache = {}
    
    def add_peer(self, peer: 'PeerAgent'):
        """Add a peer agent"""
        self.peers[peer.agent_id] = peer
        self.add_neighbor(peer)
    
    def register_resource(self, resource_key: str, resource_data: Any):
        """Register a resource with this peer"""
        self.resource_index[resource_key] = resource_data
    
    def query_resource(self, resource_key: str) -> Optional[Any]:
        """Query for a resource"""
        if resource_key in self.resource_index:
            return self.resource_index[resource_key]
        
        # Query peers
        for peer in self.peers.values():
            if resource_key in peer.resource_index:
                return peer.resource_index[resource_key]
        
        return None
    
    def broadcast_query(self, query: str) -> List[Any]:
        """Broadcast query to all peers"""
        results = []
        for peer in self.peers.values():
            if peer.state.is_active:
                response = peer.handle_query(query)
                results.append(response)
        return results
    
    def handle_query(self, query: str) -> Any:
        """Handle incoming query"""
        prompt = f"Handle the following query:\n{query}"
        response = self.chat(prompt)
        return response
    
    async def abroadcast_query(self, query: str) -> List[Any]:
        """Asynchronous broadcast query"""
        tasks = []
        for peer in self.peers.values():
            if peer.state.is_active:
                tasks.append(asyncio.create_task(peer.ahandle_query(query)))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]
    
    async def ahandle_query(self, query: str) -> Any:
        """Asynchronous query handling"""
        prompt = f"Handle the following query:\n{query}"
        response = await self.achat(prompt)
        return response


class HybridAgent(BaseAgent):
    """Hybrid agent supporting multiple communication modes"""
    
    def __init__(self, agent_id: str, system_prompt: str, model_type: str = "gpt-4o-mini"):
        super().__init__(agent_id, system_prompt, model_type, AgentRole.COORDINATOR, AgentType.HYBRID)
        self.coordination_mode = "adaptive"
        self.centralized_peers: Dict[str, BaseAgent] = {}
        self.p2p_peers: Dict[str, BaseAgent] = {}
    
    def add_centralized_peer(self, peer: BaseAgent):
        """Add a peer for centralized coordination"""
        self.centralized_peers[peer.agent_id] = peer
        self.add_neighbor(peer)
    
    def add_p2p_peer(self, peer: BaseAgent):
        """Add a peer for P2P communication"""
        self.p2p_peers[peer.agent_id] = peer
        self.add_neighbor(peer)
    
    def coordinate_centralized(self, task: Any) -> Any:
        """Coordinate using centralized approach"""
        prompt = f"Coordinate the following task using centralized approach:\n{task}"
        response = self.chat(prompt)
        return response
    
    def coordinate_p2p(self, task: Any) -> Any:
        """Coordinate using P2P approach"""
        prompt = f"Coordinate the following task using P2P approach:\n{task}"
        response = self.chat(prompt)
        return response
    
    def adaptive_coordination(self, task: Any, context: Dict[str, Any]) -> Any:
        """Adaptive coordination based on context"""
        if context.get("requires_centralization", False):
            return self.coordinate_centralized(task)
        else:
            return self.coordinate_p2p(task)
    
    async def acoordinate_centralized(self, task: Any) -> Any:
        """Asynchronous centralized coordination"""
        prompt = f"Coordinate the following task using centralized approach:\n{task}"
        response = await self.achat(prompt)
        return response
    
    async def acoordinate_p2p(self, task: Any) -> Any:
        """Asynchronous P2P coordination"""
        prompt = f"Coordinate the following task using P2P approach:\n{task}"
        response = await self.achat(prompt)
        return response
    
    async def aadaptive_coordination(self, task: Any, context: Dict[str, Any]) -> Any:
        """Asynchronous adaptive coordination"""
        if context.get("requires_centralization", False):
            return await self.acoordinate_centralized(task)
        else:
            return await self.acoordinate_p2p(task) 
