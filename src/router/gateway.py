from __future__ import annotations

from dataclasses import dataclass

from .protocol import BootstrapPayload, JsonObject


@dataclass(slots=True)
class GatewayConfig:
    """单机版前端网关配置。

    当前项目只运行在用户自己的电脑上，不做用户登录，也不做 API 鉴权。
    这里保留的配置只负责告诉前端“当前运行面长什么样”。
    """

    api_base: str = ""


def bootstrap_payload(config: GatewayConfig) -> JsonObject:
    """生成前端启动 payload。

    单机版仍然保留 bootstrap，是为了给前端一个稳定的启动入口和能力声明，
    但不再签发 token。
    """

    payload = BootstrapPayload(
        expires_in=0,
        api_base=config.api_base,
        runtime_capabilities={
            "fastapi_rest": True,
            "rest_management": True,
            "http_message_submit": True,
            "websocket_stream": False,
            "multi_chat_socket": False,
            "settings_snapshot": True,
            "auth_required": False,
        },
    )
    return payload.to_dict()
