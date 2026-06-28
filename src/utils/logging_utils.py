from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator


_REQUEST_ID_CONTEXT: ContextVar[str] = ContextVar("papers_request_id", default="-")
_LOG_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("papers_log_context", default={})
_HANDLER_MARKER = "_papers_agents_managed_handler"


def _build_reserved_record_fields() -> set[str]:
    """构建 `logging.LogRecord` 的标准字段集合。

    统一日志格式需要把业务层通过 `extra=` 传入的扩展字段和 logging 自带字段区分开，
    否则在格式化阶段很容易把系统字段重复打印。这里提前生成一份保留字段表，后续提取
    `extra` 上下文时就可以稳定排除这些内建字段。
    """

    return set(logging.makeLogRecord({}).__dict__.keys()) | {
        "asctime",
        "message",
        "request_id",
        "log_context",
    }


_RESERVED_RECORD_FIELDS = _build_reserved_record_fields()


@dataclass(slots=True)
class LoggingSettings:
    """封装项目日志系统的可配置项。

    这层配置只依赖标准库，目的是让日志系统在本地开发、命令行运行、FastAPI 启动以及
    后续部署场景里都能用同一套入口完成初始化，而不把配置散落在多个模块里。
    """

    app_name: str = "papers-agents"
    root_level: int = logging.INFO
    console_level: int = logging.INFO
    file_level: int = logging.DEBUG
    log_dir: Path = Path("logs")
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True

    @classmethod
    def from_env(cls, app_name: str = "papers-agents") -> "LoggingSettings":
        """从环境变量加载日志配置。

        支持在不改代码的前提下，通过环境变量切换日志级别、目录、文件轮转大小等参数。
        这对本地排查问题和后续部署时的快速调参都比较友好。
        """

        root_level = _resolve_log_level(os.getenv("PAPERS_LOG_LEVEL"), logging.INFO)
        console_level = _resolve_log_level(os.getenv("PAPERS_LOG_CONSOLE_LEVEL"), root_level)
        file_level = _resolve_log_level(os.getenv("PAPERS_LOG_FILE_LEVEL"), logging.DEBUG)
        log_dir = Path(os.getenv("PAPERS_LOG_DIR", "logs"))
        max_bytes = _read_int_env("PAPERS_LOG_MAX_BYTES", 10 * 1024 * 1024)
        backup_count = _read_int_env("PAPERS_LOG_BACKUP_COUNT", 5)
        enable_console = _read_bool_env("PAPERS_LOG_ENABLE_CONSOLE", True)
        enable_file = _read_bool_env("PAPERS_LOG_ENABLE_FILE", True)
        return cls(
            app_name=app_name,
            root_level=root_level,
            console_level=console_level,
            file_level=file_level,
            log_dir=log_dir,
            max_bytes=max(1, max_bytes),
            backup_count=max(1, backup_count),
            enable_console=enable_console,
            enable_file=enable_file,
        )


class ContextEnricherFilter(logging.Filter):
    """把请求上下文自动注入到每一条日志记录中。

    业务代码不需要在每一次 `logger.info(...)` 时手动重复传 `request_id`；只要请求入口
    或任务入口提前把上下文写进 `ContextVar`，这里就会在日志下沉到 handler 之前自动补齐。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """为日志记录补充 `request_id` 和上下文字段。"""

        if not getattr(record, "request_id", None):
            record.request_id = get_request_id()
        current_context = get_log_context()
        record.log_context = _merge_context(current_context, getattr(record, "log_context", None))
        return True


class ContextFormatter(logging.Formatter):
    """统一格式化控制台与文件日志。

    输出中固定包含时间、级别、模块名和 `request_id`，同时会把扩展上下文压缩成一段
    JSON 文本，方便人眼阅读，也方便后续接入日志采集系统。
    """

    def format(self, record: logging.LogRecord) -> str:
        """生成最终日志文本。"""

        if not getattr(record, "request_id", None):
            record.request_id = "-"
        if not hasattr(record, "log_context"):
            record.log_context = {}
        rendered = super().format(record)
        extra_context = self._collect_extra_context(record)
        if not extra_context:
            return rendered
        serialized = json.dumps(extra_context, ensure_ascii=False, default=_json_default, sort_keys=True)
        return f"{rendered} | ctx={serialized}"

    def _collect_extra_context(self, record: logging.LogRecord) -> dict[str, Any]:
        """提取适合附加到日志尾部的扩展上下文。"""

        merged_context = _merge_context(getattr(record, "log_context", None))
        dynamic_extra: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_FIELDS or key.startswith("_"):
                continue
            dynamic_extra[key] = value
        merged_context.update(dynamic_extra)
        merged_context.pop("request_id", None)
        return merged_context


def setup_logging(app_name: str = "papers-agents") -> logging.Logger:
    """初始化项目统一日志系统。

    该函数是整个日志系统的唯一入口，具备以下能力：
    1. 同时配置控制台与文件日志；
    2. 生成 `app.log` / `error.log` 两类文件；
    3. 支持按大小轮转；
    4. 自动接管 `warnings.warn(...)` 输出；
    5. 让 uvicorn 日志并入同一套格式，避免控制台出现多种风格混杂。
    """

    settings = LoggingSettings.from_env(app_name=app_name)
    root_logger = logging.getLogger()
    if _has_managed_handlers(root_logger):
        return logging.getLogger(app_name)
    _detach_managed_handlers(root_logger)
    root_logger.setLevel(_effective_root_level(settings))

    formatter = ContextFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | req=%(request_id)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    shared_filter = ContextEnricherFilter()

    if settings.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(settings.console_level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(shared_filter)
        setattr(console_handler, _HANDLER_MARKER, True)
        root_logger.addHandler(console_handler)

    if settings.enable_file:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        app_handler = _build_rotating_file_handler(
            settings.log_dir / "app.log",
            settings.file_level,
            settings.max_bytes,
            settings.backup_count,
            formatter,
            shared_filter,
        )
        error_handler = _build_rotating_file_handler(
            settings.log_dir / "error.log",
            logging.ERROR,
            settings.max_bytes,
            settings.backup_count,
            formatter,
            shared_filter,
        )
        root_logger.addHandler(app_handler)
        root_logger.addHandler(error_handler)

    logging.captureWarnings(True)
    _configure_uvicorn_loggers()

    logger = logging.getLogger(app_name)
    logger.info(
        "日志系统初始化完成",
        extra={
            "app_name": settings.app_name,
            "log_dir": str(settings.log_dir.resolve()),
            "console_enabled": settings.enable_console,
            "file_enabled": settings.enable_file,
        },
    )
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """获取模块级 logger。

    所有业务模块都应通过这个函数获取 logger，而不是各自再手动拼接 handler 或格式，
    这样才能保证项目日志风格统一、请求上下文一致。
    """

    return logging.getLogger(name or "papers-agents")


def get_request_id() -> str:
    """读取当前协程上下文中的请求 ID。"""

    return _REQUEST_ID_CONTEXT.get()


def get_log_context() -> dict[str, Any]:
    """读取当前协程上下文中的日志扩展字段。"""

    return dict(_LOG_CONTEXT.get())


def bind_log_context(**context: Any) -> None:
    """向当前上下文追加日志字段。

    适合在单次任务、单次请求、单个工具调用开始时补充关键上下文，例如 `session_key`、
    `source`、`provider` 等。后续该上下文范围内的日志会自动继承这些字段。
    """

    merged = get_log_context()
    for key, value in context.items():
        if value is not None:
            merged[str(key)] = value
    _LOG_CONTEXT.set(merged)


def clear_log_context() -> None:
    """清空当前协程上下文中的扩展日志字段。"""

    _LOG_CONTEXT.set({})
    _REQUEST_ID_CONTEXT.set("-")


@contextmanager
def logging_context(request_id: str | None = None, **context: Any) -> Iterator[None]:
    """在一个受控作用域内临时绑定请求日志上下文。

    离开 `with` 代码块后会自动恢复之前的上下文，避免并发请求之间的上下文串线。
    """

    previous_context = get_log_context()
    previous_request_id = get_request_id()
    merged = dict(previous_context)
    for key, value in context.items():
        if value is not None:
            merged[str(key)] = value
    context_token = _LOG_CONTEXT.set(merged)
    request_token = _REQUEST_ID_CONTEXT.set(request_id or previous_request_id)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(context_token)
        _REQUEST_ID_CONTEXT.reset(request_token)


def _build_rotating_file_handler(
    file_path: Path,
    level: int,
    max_bytes: int,
    backup_count: int,
    formatter: logging.Formatter,
    context_filter: logging.Filter,
) -> RotatingFileHandler:
    """构建带轮转能力的文件 handler。"""

    handler = RotatingFileHandler(
        filename=file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler.addFilter(context_filter)
    setattr(handler, _HANDLER_MARKER, True)
    return handler


def _detach_managed_handlers(root_logger: logging.Logger) -> None:
    """移除并关闭上一轮由本项目创建的 handler。

    FastAPI 测试和热重载场景里，应用实例可能被创建多次；如果不先清理旧 handler，就会
    出现一条日志打印多次的问题。
    """

    for handler in list(root_logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root_logger.removeHandler(handler)
            handler.close()


def _has_managed_handlers(root_logger: logging.Logger) -> bool:
    """判断 root logger 当前是否已经挂载了项目自管 handler。"""

    return any(getattr(handler, _HANDLER_MARKER, False) for handler in root_logger.handlers)


def _configure_uvicorn_loggers() -> None:
    """让 uvicorn 相关 logger 走项目统一格式。

    这样做的目的是避免服务启动后控制台同时出现 uvicorn 默认格式和项目自定义格式，
    提高排查时的可读性。
    """

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True


def _effective_root_level(settings: LoggingSettings) -> int:
    """计算 root logger 应使用的最低日志级别。"""

    candidates = [settings.root_level]
    if settings.enable_console:
        candidates.append(settings.console_level)
    if settings.enable_file:
        candidates.append(settings.file_level)
        candidates.append(logging.ERROR)
    return min(candidates)


def _merge_context(*contexts: Any) -> dict[str, Any]:
    """把多个上下文字典合并成一份稳定结构。"""

    merged: dict[str, Any] = {}
    for context in contexts:
        if not isinstance(context, dict):
            continue
        for key, value in context.items():
            if value is not None:
                merged[str(key)] = value
    return merged


def _resolve_log_level(raw_level: str | None, default: int) -> int:
    """把字符串日志级别转换成 `logging` 可用的整数级别。"""

    if raw_level is None or not raw_level.strip():
        return default
    candidate = raw_level.strip().upper()
    resolved = logging.getLevelName(candidate)
    return resolved if isinstance(resolved, int) else default


def _read_bool_env(name: str, default: bool) -> bool:
    """从环境变量中读取布尔值。"""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _read_int_env(name: str, default: int) -> int:
    """从环境变量中读取整数值。"""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value.strip())
    except (TypeError, ValueError):
        return default


def _json_default(value: Any) -> str:
    """为日志中的非常规对象提供安全的 JSON 序列化兜底。"""

    return str(value)
