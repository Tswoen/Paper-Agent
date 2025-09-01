import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

import autogen
from autogen.agentchat import AssistantAgent, UserProxyAgent
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

class AgentOrchestrator:
    """智能体协调器，负责管理和协调多个智能体的工作"""
    
    def __init__(self):
        """初始化智能体协调器"""
        # 配置AutoGen代理
        config_list = [{
            "model": "gpt-4o-mini",  # 示例模型
            "api_key": "your-api-key"  # 实际使用时需要配置
        }]
        
        # 创建协调助手代理
        self.orchestrator_assistant = AssistantAgent(
            name="orchestrator_assistant",
            system_message="你是一个智能体协调器，负责管理和协调多个智能体的工作，根据用户请求分配任务给合适的智能体。",
            llm_config={"config_list": config_list}
        )
        
        # 创建用户代理
        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            system_message="用户代理，负责与用户交互并调用协调器助手。",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10
        )
        
        # 注册协调器工具
        self.user_proxy.register_function(
            function_map={
                "process_request": self.process_request,
                "get_available_agents": self.get_available_agents,
                "get_agent_status": self.get_agent_status,
                "register_agent": self.register_agent,
                "unregister_agent": self.unregister_agent
            }
        )
        
        # 智能体注册表
        self.agents = {
            # 键: 智能体名称, 值: 智能体实例或配置
        }
        
        # 任务历史
        self.task_history = []
        
        # 系统状态
        self.system_status = {
            "initialized_at": datetime.now(),
            "last_activity_at": datetime.now(),
            "active_tasks": 0,
            "completed_tasks": 0
        }
    
    def register_agent(self, agent_name: str, agent_config: Dict) -> Dict:
        """
        注册新的智能体
        
        参数:
            agent_name: 智能体名称
            agent_config: 智能体配置
        
        返回:
            注册结果
        """
        logger.info(f"注册智能体: {agent_name}")
        
        if agent_name in self.agents:
            logger.error(f"智能体 '{agent_name}' 已存在")
            return {
                "success": False,
                "error": f"智能体 '{agent_name}' 已存在",
                "message": "注册智能体失败"
            }
        
        # 注册智能体
        self.agents[agent_name] = agent_config
        
        # 更新系统状态
        self._update_system_status()
        
        logger.info(f"智能体 '{agent_name}' 注册成功")
        return {
            "success": True,
            "message": f"智能体 '{agent_name}' 注册成功",
            "agent_name": agent_name,
            "agent_count": len(self.agents)
        }
    
    def unregister_agent(self, agent_name: str) -> Dict:
        """
        注销智能体
        
        参数:
            agent_name: 智能体名称
        
        返回:
            注销结果
        """
        logger.info(f"注销智能体: {agent_name}")
        
        if agent_name not in self.agents:
            logger.error(f"智能体 '{agent_name}' 不存在")
            return {
                "success": False,
                "error": f"智能体 '{agent_name}' 不存在",
                "message": "注销智能体失败"
            }
        
        # 注销智能体
        del self.agents[agent_name]
        
        # 更新系统状态
        self._update_system_status()
        
        logger.info(f"智能体 '{agent_name}' 注销成功")
        return {
            "success": True,
            "message": f"智能体 '{agent_name}' 注销成功",
            "agent_count": len(self.agents)
        }
    
    def get_available_agents(self) -> Dict:
        """
        获取所有可用的智能体
        
        返回:
            可用智能体列表
        """
        logger.info("获取可用智能体列表")
        
        # 构建智能体列表，不包含敏感信息
        available_agents = []
        for agent_name, agent_config in self.agents.items():
            # 提取智能体的基本信息，避免暴露敏感配置
            agent_info = {
                "name": agent_name,
                "type": agent_config.get("type", "unknown"),
                "description": agent_config.get("description", "无描述"),
                "capabilities": agent_config.get("capabilities", [])
            }
            available_agents.append(agent_info)
        
        return {
            "success": True,
            "message": "获取可用智能体列表成功",
            "agents": available_agents,
            "total_count": len(available_agents)
        }
    
    def get_agent_status(self, agent_name: Optional[str] = None) -> Dict:
        """
        获取智能体状态
        
        参数:
            agent_name: 智能体名称，如果为None则获取所有智能体状态
        
        返回:
            智能体状态信息
        """
        logger.info(f"获取智能体状态: {agent_name if agent_name else '所有'}")
        
        if agent_name:
            # 获取特定智能体状态
            if agent_name not in self.agents:
                logger.error(f"智能体 '{agent_name}' 不存在")
                return {
                    "success": False,
                    "error": f"智能体 '{agent_name}' 不存在",
                    "message": "获取智能体状态失败"
                }
            
            agent_config = self.agents[agent_name]
            # 构建智能体状态信息
            agent_status = {
                "name": agent_name,
                "type": agent_config.get("type", "unknown"),
                "status": agent_config.get("status", "online"),
                "last_active_at": agent_config.get("last_active_at", "未知"),
                "tasks_completed": agent_config.get("tasks_completed", 0)
            }
            
            return {
                "success": True,
                "message": f"获取智能体 '{agent_name}' 状态成功",
                "agent_status": agent_status
            }
        else:
            # 获取所有智能体状态
            all_agent_status = []
            for name, config in self.agents.items():
                agent_status = {
                    "name": name,
                    "type": config.get("type", "unknown"),
                    "status": config.get("status", "online"),
                    "last_active_at": config.get("last_active_at", "未知"),
                    "tasks_completed": config.get("tasks_completed", 0)
                }
                all_agent_status.append(agent_status)
            
            return {
                "success": True,
                "message": "获取所有智能体状态成功",
                "agents_status": all_agent_status,
                "total_count": len(all_agent_status)
            }
    
    def process_request(self, request: Dict) -> Dict:
        """
        处理用户请求，将其分配给合适的智能体
        
        参数:
            request: 请求信息
        
        返回:
            处理结果
        """
        logger.info(f"处理用户请求: {request.get('type', 'unknown')}")
        
        # 记录任务
        task_id = self._generate_task_id()
        task = {
            "id": task_id,
            "request": request,
            "status": "pending",
            "created_at": datetime.now(),
            "assigned_to": None,
            "result": None
        }
        self.task_history.append(task)
        
        # 更新系统状态
        self.system_status["active_tasks"] += 1
        self._update_system_status()
        
        try:
            # 确定请求类型
            request_type = request.get('type', '').lower()
            
            # 根据请求类型选择合适的智能体
            target_agent = self._select_agent_for_request(request_type)
            
            if not target_agent:
                logger.error(f"没有找到适合处理请求类型 '{request_type}' 的智能体")
                # 更新任务状态
                self._update_task_status(task_id, "failed", error="没有找到适合的智能体")
                return {
                    "success": False,
                    "error": f"没有找到适合处理请求类型 '{request_type}' 的智能体",
                    "message": "处理请求失败",
                    "task_id": task_id
                }
            
            # 分配任务给目标智能体
            logger.info(f"将任务 '{task_id}' 分配给智能体: {target_agent}")
            self._update_task_status(task_id, "processing", assigned_to=target_agent)
            
            # 调用智能体处理请求
            result = self._call_agent(target_agent, request)
            
            # 更新任务状态为完成
            self._update_task_status(task_id, "completed", result=result)
            
            logger.info(f"任务 '{task_id}' 处理完成")
            return {
                "success": True,
                "message": "请求处理完成",
                "task_id": task_id,
                "result": result,
                "assigned_agent": target_agent
            }
        except Exception as e:
            logger.error(f"处理请求时发生错误: {str(e)}")
            # 更新任务状态为失败
            self._update_task_status(task_id, "failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "message": "处理请求时发生错误",
                "task_id": task_id
            }
    
    def get_system_status(self) -> Dict:
        """
        获取系统状态
        
        返回:
            系统状态信息
        """
        logger.info("获取系统状态")
        
        # 创建系统状态副本
        status = self.system_status.copy()
        status["agent_count"] = len(self.agents)
        
        return {
            "success": True,
            "message": "获取系统状态成功",
            "system_status": status
        }
    
    def get_task_history(self, limit: int = 10, status: Optional[str] = None) -> Dict:
        """
        获取任务历史
        
        参数:
            limit: 返回的任务数量限制
            status: 任务状态过滤
        
        返回:
            任务历史列表
        """
        logger.info(f"获取任务历史，限制: {limit}, 状态: {status if status else '所有'}")
        
        # 过滤任务历史
        filtered_tasks = []
        for task in reversed(self.task_history):  # 从最新的任务开始
            if status and task["status"] != status:
                continue
            
            # 创建任务信息副本，不包含敏感数据
            task_info = {
                "id": task["id"],
                "request_type": task["request"].get('type', 'unknown'),
                "status": task["status"],
                "created_at": task["created_at"],
                "assigned_to": task["assigned_to"],
                "completed_at": task.get("completed_at", None)
            }
            
            filtered_tasks.append(task_info)
            
            if len(filtered_tasks) >= limit:
                break
        
        return {
            "success": True,
            "message": "获取任务历史成功",
            "tasks": filtered_tasks,
            "total_count": len(filtered_tasks),
            "limit": limit,
            "status_filter": status
        }
    
    def _generate_task_id(self) -> str:
        """生成唯一的任务ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"task_{timestamp}"
    
    def _update_system_status(self) -> None:
        """更新系统状态"""
        self.system_status["last_activity_at"] = datetime.now()
        
        # 计算已完成的任务数量
        completed_tasks = sum(1 for task in self.task_history if task["status"] == "completed")
        self.system_status["completed_tasks"] = completed_tasks
        
        # 计算活跃的任务数量
        active_tasks = sum(1 for task in self.task_history if task["status"] in ["pending", "processing"])
        self.system_status["active_tasks"] = active_tasks
    
    def _select_agent_for_request(self, request_type: str) -> Optional[str]:
        """
        根据请求类型选择合适的智能体
        
        参数:
            request_type: 请求类型
        
        返回:
            智能体名称，如果没有找到合适的智能体则返回None
        """
        # 简单的匹配逻辑，实际应用中可以使用更复杂的算法
        agent_mapping = {
            "search": "paper_retrieval_agent",
            "retrieve": "paper_retrieval_agent",
            "analysis": "paper_analysis_agent",
            "read": "paper_reading_agent",
            "summarize": "paper_analysis_agent",
            "extract": "paper_reading_agent",
            "write": "paper_writing_agent"
        }
        
        # 查找匹配的智能体
        for keyword, agent_name in agent_mapping.items():
            if keyword in request_type and agent_name in self.agents:
                return agent_name
        
        # 如果没有找到完全匹配的，尝试寻找具有相应能力的智能体
        for agent_name, agent_config in self.agents.items():
            capabilities = agent_config.get("capabilities", [])
            for capability in capabilities:
                if capability.lower() in request_type:
                    return agent_name
        
        return None
    
    def _call_agent(self, agent_name: str, request: Dict) -> Any:
        """
        调用智能体处理请求
        
        参数:
            agent_name: 智能体名称
            request: 请求信息
        
        返回:
            智能体的处理结果
        """
        logger.info(f"调用智能体 '{agent_name}' 处理请求")
        
        # 获取智能体配置
        agent_config = self.agents.get(agent_name)
        
        if not agent_config:
            raise ValueError(f"智能体 '{agent_name}' 不存在")
        
        # 检查智能体实例是否可用
        agent_instance = agent_config.get("instance")
        
        if agent_instance:
            # 直接调用智能体实例的方法
            # 这里需要根据不同的智能体类型和请求类型调用相应的方法
            request_type = request.get('type', '').lower()
            
            if "search" in request_type or "retrieve" in request_type:
                # 调用检索智能体
                return agent_instance.process_search_request(request)
            elif "analysis" in request_type or "summarize" in request_type:
                # 调用分析智能体
                return agent_instance.process_analysis_request(request.get('paper', {}), request.get('analysis_type', 'full'))
            elif "read" in request_type or "extract" in request_type:
                # 调用阅读智能体
                return agent_instance.start_reading(request.get('paper', {}))
            else:
                # 默认调用方式
                process_method = getattr(agent_instance, "process_request", None)
                if process_method:
                    return process_method(request)
                else:
                    raise ValueError(f"智能体 '{agent_name}' 没有合适的处理方法")
        else:
            # 如果没有智能体实例，返回模拟结果
            # 实际应用中，这里可能需要根据智能体配置创建实例或通过其他方式调用
            return {
                "simulated_result": True,
                "message": f"智能体 '{agent_name}' 模拟处理了请求",
                "request_type": request.get('type', 'unknown')
            }
    
    def _update_task_status(self, task_id: str, status: str, result: Any = None, error: str = None, assigned_to: str = None) -> None:
        """
        更新任务状态
        
        参数:
            task_id: 任务ID
            status: 新的状态
            result: 任务结果
            error: 错误信息
            assigned_to: 分配的智能体
        """
        for task in self.task_history:
            if task["id"] == task_id:
                task["status"] = status
                if result is not None:
                    task["result"] = result
                if error is not None:
                    task["error"] = error
                if assigned_to is not None:
                    task["assigned_to"] = assigned_to
                if status == "completed" or status == "failed":
                    task["completed_at"] = datetime.now()
                break
        
        # 更新系统状态
        self._update_system_status()

# 示例用法
if __name__ == "__main__":
    try:
        orchestrator = AgentOrchestrator()
        
        # 注册一些示例智能体
        orchestrator.register_agent(
            "paper_retrieval_agent",
            {
                "type": "retrieval",
                "description": "论文检索智能体，负责从学术数据库检索相关论文",
                "capabilities": ["search", "retrieve", "fetch"],
                "status": "online",
                "tasks_completed": 0,
                "last_active_at": datetime.now()
            }
        )
        
        orchestrator.register_agent(
            "paper_analysis_agent",
            {
                "type": "analysis",
                "description": "论文分析智能体，负责分析论文内容和提取关键信息",
                "capabilities": ["analyze", "summarize", "evaluate"],
                "status": "online",
                "tasks_completed": 0,
                "last_active_at": datetime.now()
            }
        )
        
        orchestrator.register_agent(
            "paper_reading_agent",
            {
                "type": "reading",
                "description": "论文阅读智能体，负责模拟论文阅读过程",
                "capabilities": ["read", "extract", "note"],
                "status": "online",
                "tasks_completed": 0,
                "last_active_at": datetime.now()
            }
        )
        
        # 获取可用智能体
        agents_result = orchestrator.get_available_agents()
        print(f"可用智能体: {agents_result}")
        
        # 处理请求
        search_request = {
            "type": "search",
            "query": "大型语言模型应用",
            "limit": 10,
            "sort_by": "relevance"
        }
        
        search_result = orchestrator.process_request(search_request)
        print(f"处理结果: {search_result}")
        
        # 获取系统状态
        system_status = orchestrator.get_system_status()
        print(f"系统状态: {system_status}")
        
        # 获取任务历史
        task_history = orchestrator.get_task_history(limit=5)
        print(f"任务历史: {task_history}")
        
    except Exception as e:
        print(f"错误: {str(e)}")