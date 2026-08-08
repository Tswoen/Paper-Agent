"""生成人能直接看懂创建时间的标识符。"""

from __future__ import annotations

from datetime import datetime
import secrets
import string


# 随机部分只使用小写字母和数字，复制、输入和作为目录名时都不会出现特殊字符。
_RANDOM_CHARACTERS = string.ascii_lowercase + string.digits
_RANDOM_PART_LENGTH = 12


def create_readable_id() -> str:
    """返回“年月日时分秒_随机字符串”格式的标识符。

    前面的时间来自当前机器的本地时间，查看会话或回合编号时可以直接知道它大约何时创建。
    后面的 12 位随机字符串用于区分同一秒内创建的多条记录，避免编号重复。
    """

    created_time = datetime.now().strftime("%Y%m%d%H%M%S")
    random_part = "".join(secrets.choice(_RANDOM_CHARACTERS) for _ in range(_RANDOM_PART_LENGTH))
    return f"{created_time}_{random_part}"
