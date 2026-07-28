from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from src.graph.runtime_resources import WorkflowRuntimeResources
from urllib.parse import urlparse

import httpx

from src.paper_retrieval.models import PaperDocument
from src.utils.read_utils.cache import paper_cache_dir, read_cached_source_url, write_metadata


_DOWNLOAD_SEMAPHORE = asyncio.Semaphore(4)


@dataclass(slots=True)
class DownloadedPaper:
    """保存一次全文获取的结果，调用方只需根据状态继续处理。"""

    status: str
    reason: str = ""
    source_url: str | None = None
    file_path: Path | None = None
    content_type: str | None = None
    reused_cache: bool = False


def download_paper_fulltext(
    paper: PaperDocument,
    *,
    cache_dir: str | Path,
    connect_timeout_seconds: int,
    download_timeout_seconds: int,
    max_file_size_mb: int,
    runtime_resources: WorkflowRuntimeResources | None = None,
) -> DownloadedPaper:
    """兼容旧同步阅读节点的全文下载入口，主逻辑统一交给异步实现。"""

    # 中文注释：这里不再保留第二套同步下载代码，避免同步逻辑和异步逻辑以后改出差异。
    # 新代码请直接 await async_download_paper_fulltext(...)，这个入口只服务尚未 async 化的旧节点。
    awaitable = async_download_paper_fulltext(
        paper,
        cache_dir=cache_dir,
        connect_timeout_seconds=connect_timeout_seconds,
        download_timeout_seconds=download_timeout_seconds,
        max_file_size_mb=max_file_size_mb,
        runtime_resources=runtime_resources,
    )
    if runtime_resources is not None:
        return runtime_resources.run_from_sync(awaitable)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 中文注释：没有 run 级资源时，说明这只是普通同步脚本或测试调用，
        # 临时开一个 Runner 跑完异步下载即可，局部 AsyncClient 会在异步函数里自己关闭。
        with asyncio.Runner() as runner:
            return runner.run(awaitable)
    # 中文注释：如果已经处在 async 环境中，就不能再套同步桥，否则容易卡住事件循环。
    # 调用方此时应该直接 await 异步下载主入口。
    if hasattr(awaitable, "close"):
        awaitable.close()
    raise RuntimeError("同步下载兼容接口不能在已有事件循环中调用，请改用 await async_download_paper_fulltext(...)")


async def async_download_paper_fulltext(
    paper: PaperDocument,
    *,
    cache_dir: str | Path,
    connect_timeout_seconds: int,
    download_timeout_seconds: int,
    max_file_size_mb: int,
    runtime_resources: WorkflowRuntimeResources | None = None,
) -> DownloadedPaper:
    """异步下载论文全文，并优先复用当前 run 的下载资源。"""

    paper_dir = paper_cache_dir(cache_dir, paper)
    cached = _find_cached_file(paper_dir)
    if cached is not None:
        return DownloadedPaper(
            status="downloaded",
            source_url=read_cached_source_url(paper_dir),
            file_path=cached,
            content_type="application/pdf" if cached.suffix.lower() == ".pdf" else "text/html",
            reused_cache=True,
        )

    source_url = _find_fulltext_url(paper)
    if source_url is None:
        return DownloadedPaper(status="no_url", reason="论文没有可尝试的全文地址")
    if not _is_safe_http_url(source_url):
        return DownloadedPaper(status="download_failed", reason="全文地址只允许使用 http 或 https", source_url=source_url)

    maximum_bytes = max(1, max_file_size_mb) * 1024 * 1024
    timeout = httpx.Timeout(float(max(1, download_timeout_seconds)), connect=float(max(1, connect_timeout_seconds)))
    # 中文注释：优先复用当前 run 里的下载信号量和 AsyncClient；如果这次调用不是从
    # run 级工作流进来的，就退回原来的模块级信号量和局部 client 逻辑，保证旧入口不受影响。
    semaphore = runtime_resources.download_semaphore if runtime_resources is not None else _DOWNLOAD_SEMAPHORE
    client = runtime_resources.http_client if runtime_resources is not None else httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        max_redirects=5,
    )
    owns_client = runtime_resources is None
    async with semaphore:
        try:
            async with client.stream(
                "GET",
                source_url,
                headers={"Accept": "application/pdf, text/html;q=0.9"},
                timeout=timeout,
            ) as response:
                final_url = str(response.url)
                if not _is_safe_http_url(final_url):
                    return DownloadedPaper(status="download_failed", reason="跳转后的全文地址不安全", source_url=final_url)
                if response.status_code < 200 or response.status_code >= 300:
                    return DownloadedPaper(status="download_failed", reason=f"下载地址返回 HTTP {response.status_code}", source_url=final_url)
                declared_length = _safe_content_length(response.headers.get("content-length"))
                if declared_length is not None and declared_length > maximum_bytes:
                    return DownloadedPaper(status="download_failed", reason="文件超过允许大小", source_url=final_url)
                content = await _read_limited_content_async(response, maximum_bytes)
                if content is None:
                    return DownloadedPaper(status="download_failed", reason="文件超过允许大小", source_url=final_url)
                content_type = response.headers.get("content-type", "")
                content_kind = _detect_content_kind(content, content_type)
                if content_kind is None:
                    return DownloadedPaper(status="download_failed", reason="下载内容不是可读取的 PDF 或 HTML", source_url=final_url)
        except httpx.TimeoutException:
            return DownloadedPaper(status="download_failed", reason="下载全文超时", source_url=source_url)
        except httpx.HTTPError as exc:
            return DownloadedPaper(status="download_failed", reason=f"下载全文失败：{exc}", source_url=source_url)
        finally:
            if owns_client:
                await client.aclose()

    # 文件写入仍是本地阻塞操作，先放到线程里，避免在异步流程里直接卡住事件循环。
    await asyncio.to_thread(paper_dir.mkdir, parents=True, exist_ok=True)
    suffix = ".pdf" if content_kind == "pdf" else ".html"
    file_path = paper_dir / f"original{suffix}"
    await asyncio.to_thread(file_path.write_bytes, content)
    await asyncio.to_thread(write_metadata, paper_dir, paper, source_url=final_url, content_type=content_type)
    return DownloadedPaper(
        status="downloaded",
        source_url=final_url,
        file_path=file_path,
        content_type="application/pdf" if content_kind == "pdf" else "text/html",
    )


def _find_fulltext_url(paper: PaperDocument) -> str | None:
    """按可靠程度从论文对象中找出第一个可用全文地址。"""

    candidates: list[Any] = [paper.pdf_url]
    metadata = paper.metadata or {}
    candidates.extend([metadata.get("open_access_pdf"), metadata.get("openAccessPdf"), metadata.get("pdf_url")])
    candidates.append(paper.url)
    for value in candidates:
        if isinstance(value, dict):
            value = value.get("url")
        text = str(value).strip() if value is not None else ""
        if text:
            return text
    return None


def _paper_cache_name(paper: PaperDocument) -> str:
    """为论文生成不会包含非法文件名字符的固定缓存目录名。"""

    return str(paper.paperId or paper.id or paper.doi or paper.title).strip()


def _find_cached_file(paper_dir: Path) -> Path | None:
    """读取已成功保存的原始全文，避免相同论文重复下载。"""

    for name in ("original.pdf", "original.html", "source.pdf", "source.html"):
        candidate = paper_dir / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _read_cached_source_url(paper_dir: Path) -> str | None:
    """从缓存说明文件中取回原始全文地址，读取失败时返回空值。"""

    try:
        payload = json.loads((paper_dir / "source.json").read_text(encoding="utf-8"))
        return str(payload.get("source_url") or "") or None
    except (OSError, json.JSONDecodeError):
        return None


def _is_safe_http_url(url: str) -> bool:
    """只允许网络下载所需的 http 和 https 地址。"""

    parsed = urlparse(url)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _safe_content_length(value: str | None) -> int | None:
    """把响应头中的文件大小转换为整数，无法识别时不提前拦截。"""

    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


async def _read_limited_content_async(response: httpx.Response, maximum_bytes: int) -> bytes | None:
    """异步分段读取网络内容，超过限制时立刻停止并返回空值。"""

    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > maximum_bytes:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _detect_content_kind(content: bytes, content_type: str) -> str | None:
    """根据真实内容判断下载结果是 PDF、HTML 还是无效页面。"""

    if not content:
        return None
    if content.startswith(b"%PDF-"):
        return "pdf"
    lowered = content[:1024].lower()
    if "html" in content_type.lower() or b"<!doctype html" in lowered or b"<html" in lowered:
        return "html"
    return None
