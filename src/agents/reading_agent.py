import logging
from typing import Dict, List, Optional, Union
import time
from datetime import datetime

import autogen
from autogen.agentchat import AssistantAgent, UserProxyAgent
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

class PaperReadingAgent:
    """基于AutoGen框架的论文阅读智能体"""
    
    def __init__(self):
        """初始化论文阅读智能体"""
        # 配置AutoGen代理
        config_list = [{
            "model": "gpt-4o-mini",  # 示例模型
            "api_key": "your-api-key"  # 实际使用时需要配置
        }]
        
        # 创建论文阅读助手代理
        self.reading_assistant = AssistantAgent(
            name="paper_reading_assistant",
            system_message="你是一个论文阅读专家，负责模拟论文阅读过程，记录阅读进度和理解情况，提供阅读建议和辅助。",
            llm_config={"config_list": config_list}
        )
        
        # 创建用户代理
        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            system_message="用户代理，负责与用户交互并调用论文阅读助手。",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10
        )
        
        # 注册论文阅读工具
        self.user_proxy.register_function(
            function_map={
                "start_reading": self.start_reading,
                "continue_reading": self.continue_reading,
                "stop_reading": self.stop_reading,
                "get_reading_progress": self.get_reading_progress,
                "ask_question_about_paper": self.ask_question_about_paper
            }
        )
        
        # 存储阅读状态
        self.reading_state = {
            "is_reading": False,
            "current_paper": None,
            "progress": 0.0,  # 0.0 - 1.0
            "current_section": None,
            "sections_read": [],
            "reading_history": [],
            "start_time": None,
            "last_update_time": None,
            "notes": {}
        }
    
    def start_reading(self, paper: Dict) -> Dict:
        """
        开始阅读论文
        
        参数:
            paper: 论文信息字典
        
        返回:
            阅读开始状态
        """
        logger.info(f"开始阅读论文: {paper.get('title', '未知标题')}")
        
        # 重置阅读状态
        self.reading_state = {
            "is_reading": True,
            "current_paper": paper,
            "progress": 0.0,
            "current_section": "摘要",
            "sections_read": [],
            "reading_history": [],
            "start_time": datetime.now(),
            "last_update_time": datetime.now(),
            "notes": {}
        }
        
        # 记录阅读开始事件
        self._record_reading_event("start", "开始阅读论文")
        
        return {
            "success": True,
            "message": f"已开始阅读论文: {paper.get('title', '未知标题')}",
            "reading_state": self._get_public_reading_state()
        }
    
    def continue_reading(self, section: Optional[str] = None, duration_minutes: int = 15) -> Dict:
        """
        继续阅读论文
        
        参数:
            section: 要阅读的部分，如果为None则继续阅读当前部分
            duration_minutes: 阅读持续时间（分钟）
        
        返回:
            阅读进度更新
        """
        if not self.reading_state["is_reading"]:
            logger.error("没有正在阅读的论文，请先调用start_reading开始阅读")
            return {
                "success": False,
                "error": "没有正在阅读的论文，请先调用start_reading开始阅读",
                "message": "阅读失败"
            }
        
        # 更新当前阅读部分
        if section:
            self.reading_state["current_section"] = section
        
        # 模拟阅读过程
        logger.info(f"继续阅读论文: {self.reading_state['current_paper'].get('title', '未知标题')}, 部分: {self.reading_state['current_section']}, 持续时间: {duration_minutes}分钟")
        
        # 模拟时间流逝
        # 实际应用中，这部分可能会有更复杂的逻辑，如跟踪实际阅读时间
        time.sleep(1)  # 仅作为演示，实际应用中不需要这行代码
        
        # 更新阅读进度
        # 假设每阅读一个部分，进度增加0.2
        section_progress = 0.2
        self.reading_state["progress"] = min(1.0, self.reading_state["progress"] + section_progress)
        
        # 记录已阅读的部分
        if self.reading_state["current_section"] not in self.reading_state["sections_read"]:
            self.reading_state["sections_read"].append(self.reading_state["current_section"])
        
        # 更新最后更新时间
        self.reading_state["last_update_time"] = datetime.now()
        
        # 记录阅读事件
        self._record_reading_event(
            "continue", 
            f"阅读部分: {self.reading_state['current_section']}, 持续时间: {duration_minutes}分钟"
        )
        
        # 生成阅读建议
        reading_suggestions = self._generate_reading_suggestions()
        
        return {
            "success": True,
            "message": f"已完成{self.reading_state['current_section']}的阅读",
            "reading_state": self._get_public_reading_state(),
            "suggestions": reading_suggestions
        }
    
    def stop_reading(self) -> Dict:
        """
        停止阅读论文
        
        返回:
            阅读停止状态
        """
        if not self.reading_state["is_reading"]:
            logger.error("没有正在阅读的论文")
            return {
                "success": False,
                "error": "没有正在阅读的论文",
                "message": "停止阅读失败"
            }
        
        # 计算阅读总时间
        total_reading_time = (datetime.now() - self.reading_state["start_time"]).total_seconds() / 60  # 转换为分钟
        
        # 记录阅读停止事件
        self._record_reading_event("stop", f"停止阅读论文，总阅读时间: {total_reading_time:.2f}分钟")
        
        # 更新阅读状态
        self.reading_state["is_reading"] = False
        self.reading_state["last_update_time"] = datetime.now()
        
        paper_title = self.reading_state["current_paper"].get('title', '未知标题')
        logger.info(f"停止阅读论文: {paper_title}, 阅读进度: {self.reading_state['progress'] * 100:.1f}%")
        
        return {
            "success": True,
            "message": f"已停止阅读论文: {paper_title}",
            "reading_summary": {
                "paper_title": paper_title,
                "total_reading_time": total_reading_time,
                "progress_percentage": self.reading_state["progress"] * 100,
                "sections_read": self.reading_state["sections_read"],
                "reading_state": self._get_public_reading_state()
            }
        }
    
    def get_reading_progress(self) -> Dict:
        """
        获取当前阅读进度
        
        返回:
            阅读进度信息
        """
        if not self.reading_state["is_reading"]:
            logger.warning("当前没有正在阅读的论文")
            return {
                "success": True,
                "is_reading": False,
                "message": "当前没有正在阅读的论文",
                "reading_state": None
            }
        
        return {
            "success": True,
            "is_reading": True,
            "message": "阅读进度查询成功",
            "reading_state": self._get_public_reading_state()
        }
    
    def ask_question_about_paper(self, question: str) -> Dict:
        """
        就当前阅读的论文提问
        
        参数:
            question: 问题文本
        
        返回:
            问题回答
        """
        if not self.reading_state["is_reading"]:
            logger.error("没有正在阅读的论文，无法回答问题")
            return {
                "success": False,
                "error": "没有正在阅读的论文，无法回答问题",
                "message": "回答问题失败"
            }
        
        logger.info(f"收到关于论文的问题: {question}")
        
        # 模拟回答问题
        # 实际应用中，这里可以使用LLM来生成更准确的回答
        paper = self.reading_state["current_paper"]
        
        # 生成基于论文内容的回答
        # 这里仅作为示例，实际应用中可以使用更复杂的逻辑
        answer = f"这是关于论文 '{paper.get('title', '未知标题')}' 的问题 '{question}' 的回答。"\
               f"根据论文的内容，这个问题可以从以下几个方面来理解：\n" \
               f"1. {paper.get('summary', '').split('。')[0]}。\n" \
               f"2. 关于这个问题，建议你重点阅读论文的{self.reading_state['current_section']}部分。\n" \
               f"3. 如果你需要更深入的理解，我可以为你提供更多相关信息。"
        
        # 记录问题和回答
        self._record_reading_event("question", f"问题: {question}", additional_info={"answer": answer})
        
        return {
            "success": True,
            "message": "问题回答成功",
            "question": question,
            "answer": answer
        }
    
    def _record_reading_event(self, event_type: str, description: str, additional_info: Dict = None) -> None:
        """记录阅读事件"""
        event = {
            "timestamp": datetime.now(),
            "event_type": event_type,
            "description": description,
            "additional_info": additional_info or {}
        }
        self.reading_state["reading_history"].append(event)
    
    def _get_public_reading_state(self) -> Dict:
        """获取可公开的阅读状态信息"""
        # 创建一个不包含敏感信息的阅读状态副本
        public_state = {
            "is_reading": self.reading_state["is_reading"],
            "progress": self.reading_state["progress"],
            "progress_percentage": self.reading_state["progress"] * 100,
            "current_section": self.reading_state["current_section"],
            "sections_read": self.reading_state["sections_read"],
            "time_spent_minutes": None,
            "paper_title": self.reading_state["current_paper"].get('title', '未知标题') if self.reading_state["current_paper"] else None
        }
        
        # 计算已阅读时间
        if self.reading_state["start_time"]:
            public_state["time_spent_minutes"] = (datetime.now() - self.reading_state["start_time"]).total_seconds() / 60
        
        return public_state
    
    def _generate_reading_suggestions(self) -> List[str]:
        """生成阅读建议"""
        # 根据当前阅读进度和部分生成建议
        suggestions = []
        
        if self.reading_state["progress"] < 0.2:
            suggestions.append("建议先快速浏览论文的摘要、引言和结论，了解论文的整体结构和贡献。")
        elif self.reading_state["progress"] < 0.5:
            suggestions.append("接下来可以重点阅读论文的方法部分，理解作者的技术方案。")
        elif self.reading_state["progress"] < 0.8:
            suggestions.append("建议仔细阅读实验部分，了解作者如何验证他们的方法。")
        else:
            suggestions.append("论文阅读已接近完成，建议回顾全文，总结主要贡献和发现。")
        
        # 根据当前阅读部分生成特定建议
        if self.reading_state["current_section"] == "摘要":
            suggestions.append("摘要提供了论文的概要，注意其中提到的主要贡献和结果。")
        elif self.reading_state["current_section"] == "引言":
            suggestions.append("引言介绍了研究背景和动机，关注作者提出的研究问题。")
        elif self.reading_state["current_section"] == "方法":
            suggestions.append("方法部分详细描述了作者的技术方案，注意理解其中的关键创新点。")
        elif self.reading_state["current_section"] == "实验":
            suggestions.append("实验部分展示了作者的验证结果，关注实验设计和主要发现。")
        elif self.reading_state["current_section"] == "结论":
            suggestions.append("结论总结了论文的主要贡献和局限性，注意其中提到的未来研究方向。")
        
        return suggestions
    
    def add_note(self, section: str, note: str) -> Dict:
        """
        为论文的特定部分添加笔记
        
        参数:
            section: 论文部分
            note: 笔记内容
        
        返回:
            添加笔记的结果
        """
        if not self.reading_state["is_reading"]:
            logger.error("没有正在阅读的论文，无法添加笔记")
            return {
                "success": False,
                "error": "没有正在阅读的论文，无法添加笔记",
                "message": "添加笔记失败"
            }
        
        # 初始化该部分的笔记列表（如果不存在）
        if section not in self.reading_state["notes"]:
            self.reading_state["notes"][section] = []
        
        # 添加笔记
        self.reading_state["notes"][section].append({
            "timestamp": datetime.now(),
            "note": note
        })
        
        logger.info(f"为论文部分 '{section}' 添加笔记")
        
        return {
            "success": True,
            "message": f"已为论文部分 '{section}' 添加笔记",
            "notes_count": len(self.reading_state["notes"][section])
        }

# 示例用法
if __name__ == "__main__":
    try:
        agent = PaperReadingAgent()
        # 创建一个模拟论文对象
        mock_paper = {
            "paper_id": "12345",
            "title": "大型语言模型在自然语言处理中的应用研究",
            "authors": ["张三", "李四"],
            "summary": "本文探讨了大型语言模型在自然语言处理任务中的应用，包括文本分类、情感分析、机器翻译等多个任务。实验结果表明，大型语言模型在这些任务上都取得了显著的性能提升。",
            "published": 2024,
            "published_date": "2024-01-01T00:00:00",
            "url": "https://example.com/paper/12345",
            "pdf_url": "https://example.com/paper/12345.pdf",
            "primary_category": "cs.CL",
            "categories": ["cs.CL", "cs.AI"],
            "doi": "10.1234/example.12345"
        }
        
        # 开始阅读
        start_result = agent.start_reading(mock_paper)
        print(f"开始阅读: {start_result}")
        
        # 继续阅读
        continue_result = agent.continue_reading(section="摘要", duration_minutes=5)
        print(f"继续阅读: {continue_result}")
        
        # 提问
        question_result = agent.ask_question_about_paper("这篇论文主要研究了什么？")
        print(f"提问结果: {question_result}")
        
        # 获取进度
        progress_result = agent.get_reading_progress()
        print(f"阅读进度: {progress_result}")
        
        # 停止阅读
        stop_result = agent.stop_reading()
        print(f"停止阅读: {stop_result}")
    except Exception as e:
        print(f"错误: {str(e)}")