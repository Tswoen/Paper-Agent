import asyncio
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

class WebUserProxyAgent(UserProxyAgent):
    def __init__(self, name):
        super().__init__(name)
        self.waiting_future = None  # 保存等待的future对象
    
    async def on_messages(self, messages, cancellation_token: CancellationToken):
        # 触发等待：通知前端“等待人工输入”
        self.waiting_future = asyncio.get_event_loop().create_future()
        # 等待前端输入
        user_input = await self.waiting_future
        # 收到输入后返回给AutoGen
        return TextMessage(content=user_input, source="human")

    def set_user_input(self, user_input: str):
        """外部接口：被前端调用时唤醒等待"""
        if self.waiting_future and not self.waiting_future.done():
            self.waiting_future.set_result(user_input)

userProxyAgent = WebUserProxyAgent("user_proxy")
