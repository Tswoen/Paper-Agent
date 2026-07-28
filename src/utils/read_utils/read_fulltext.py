from __future__ import annotations

import asyncio
import html
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from src.paper_retrieval.models import PaperDocument
from src.utils.read_utils.pdf_parsers import get_pdf_parser


_UNSUPPORTED_FULLTEXT_WARNING = "暂不支持该全文文件格式"


@dataclass(slots=True)
class MarkdownConversion:
    """保存全文转成 Markdown 后的文件位置、页数和提示信息。"""

    markdown_path: Path | None = None
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)


def convert_fulltext_to_markdown(
    paper: PaperDocument,
    *,
    source_path: Path,
    source_url: str | None,
) -> MarkdownConversion:
    """兼容旧同步流程的全文解析入口，主逻辑统一交给异步实现。"""

    # 中文注释：模块四整理之后，外层主入口改成了 async。
    # 这里保留一个同步壳，只是为了让还没改成 async 的阅读节点继续可用。
    awaitable = async_convert_fulltext_to_markdown(
        paper,
        source_path=source_path,
        source_url=source_url,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 中文注释：普通同步脚本或旧节点调用这里时，临时开一个事件循环把异步逻辑跑完。
        # 这样全文解析只保留一套真正的主逻辑，后面继续维护时不会出现两份代码越改越不一致。
        with asyncio.Runner() as runner:
            return runner.run(awaitable)
    # 中文注释：如果已经在 async 环境里，就不能再走这个同步兼容壳。
    # 这里直接报错，提醒调用方改成 await 异步入口，避免把事件循环卡死。
    if hasattr(awaitable, "close"):
        awaitable.close()
    raise RuntimeError("同步全文解析兼容接口不能在已有事件循环中调用，请改用 await async_convert_fulltext_to_markdown(...)")


async def async_convert_fulltext_to_markdown(
    paper: PaperDocument,
    *,
    source_path: Path,
    source_url: str | None,
) -> MarkdownConversion:
    """异步全文转换入口，供后续 async 阅读流程直接调用。"""

    markdown_path = _markdown_output_path(source_path)
    # 中文注释：哪怕只是看缓存文件存不存在，本质上也是本地磁盘操作。
    # 这里照样放进 to_thread，避免异步阅读流程在高并发时被本地文件检查拖慢。
    cached = await asyncio.to_thread(_load_cached_markdown, markdown_path)
    if cached is not None:
        return cached

    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        # 中文注释：PDF 解析和 Markdown 写入都是最容易卡住主流程的本地重操作。
        # 这里只把真正重的那一小段丢进线程，不把整个阅读节点都包进线程里。
        return await asyncio.to_thread(_convert_pdf_with_parser, paper, source_path, source_url, markdown_path)
    if suffix in {".html", ".htm"}:
        # 中文注释：HTML 读取、正文提取、Markdown 落盘同样都是阻塞型本地操作。
        # 处理方式和 PDF 保持一致，边界清楚，后面接异步阅读流程会更稳。
        return await asyncio.to_thread(_convert_html, paper, source_path, source_url, markdown_path)
    return MarkdownConversion(warnings=[_UNSUPPORTED_FULLTEXT_WARNING])


def _markdown_output_path(source_path: Path) -> Path:
    """统一计算 Markdown 结果文件路径，避免不同入口各自拼路径。"""

    return source_path.parent / "paper.md"


def _load_cached_markdown(markdown_path: Path) -> MarkdownConversion | None:
    """读取已经生成好的 Markdown 缓存，没有缓存时返回空值。"""

    if not _has_non_empty_file(markdown_path):
        return None
    return MarkdownConversion(markdown_path=markdown_path, page_count=_read_page_count(markdown_path))


def _has_non_empty_file(path: Path) -> bool:
    """判断已有 Markdown 缓存是否存在，并且里面确实有内容。"""

    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _convert_pdf(paper: PaperDocument, source_path: Path, source_url: str | None, markdown_path: Path) -> MarkdownConversion:
    """读取 PDF 每一页的文字，并把页码写进 Markdown 便于后续定位。"""

    try:
        from pypdf import PdfReader
    except ImportError:
        return MarkdownConversion(warnings=["未安装 pypdf，暂时无法读取 PDF 正文"])
    try:
        reader = PdfReader(str(source_path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        return MarkdownConversion(warnings=[f"PDF 正文读取失败：{exc}"])
    if not any(page.strip() for page in pages):
        return MarkdownConversion(warnings=["PDF 中没有可读取的文字，可能是扫描文件"])
    body: list[str] = [_markdown_header(paper, source_url, len(pages))]
    for index, page_text in enumerate(pages, start=1):
        body.extend([f"<!-- page: {index} -->", _normalise_text(page_text)])
    markdown_path.write_text("\n\n".join(body).strip() + "\n", encoding="utf-8")
    return MarkdownConversion(markdown_path=markdown_path, page_count=len(pages))


def _convert_html(paper: PaperDocument, source_path: Path, source_url: str | None, markdown_path: Path) -> MarkdownConversion:
    """提取普通 HTML 页面中的标题和段落，生成不含页码的 Markdown 文件。"""

    parser = _ArticleHtmlParser()
    try:
        parser.feed(source_path.read_text(encoding="utf-8", errors="replace"))
        parser.close()
    except OSError as exc:
        return MarkdownConversion(warnings=[f"HTML 正文读取失败：{exc}"])
    article = parser.to_markdown()
    if not article.strip():
        return MarkdownConversion(warnings=["HTML 页面没有可读取的正文"])
    markdown_path.write_text(_markdown_header(paper, source_url, None) + "\n\n" + article.strip() + "\n", encoding="utf-8")
    return MarkdownConversion(markdown_path=markdown_path)


def _markdown_header(paper: PaperDocument, source_url: str | None, page_count: int | None) -> str:
    """生成 Markdown 开头的论文基本信息，避免正文和来源信息分散保存。"""

    header = {
        "paper_id": paper.id,
        "title": paper.title,
        "doi": paper.doi,
        "source_url": source_url or paper.url,
        "page_count": page_count,
    }
    return "---\n" + json.dumps(header, ensure_ascii=False, indent=2) + "\n---"


def _read_page_count(markdown_path: Path) -> int | None:
    """从已有 Markdown 中读出 PDF 页数，缓存文件不完整时返回空值。"""

    try:
        match = re.search(r'"page_count":\s*(\d+)', markdown_path.read_text(encoding="utf-8")[:1000])
        return int(match.group(1)) if match else None
    except OSError:
        return None


def _normalise_text(value: str) -> str:
    """整理 PDF 抽取出的多余空行，让正文更适合后续分段。"""

    return re.sub(r"\n{3,}", "\n\n", value.replace("\x00", "")).strip()


def _convert_pdf_with_parser(paper: PaperDocument, source_path: Path, source_url: str | None, markdown_path: Path) -> MarkdownConversion:
    """使用可替换的 PDF 解析器生成 Markdown。

    中文注释：阅读节点只需要 Markdown，不应该关心 PDF 到底是 pypdf、PyMuPDF
    还是其它工具解析的。这里先使用 pypdf 解析器，后续新增解析器时只需要改
    get_pdf_parser 的选择规则。
    """

    parser = get_pdf_parser("pypdf")
    parsed = parser.parse(source_path)
    if parsed.warnings:
        return MarkdownConversion(warnings=parsed.warnings)
    if not parsed.pages:
        return MarkdownConversion(warnings=["PDF 中没有可读取的正文"])
    body: list[str] = [_markdown_header(paper, source_url, len(parsed.pages))]
    for page in parsed.pages:
        body.extend([f"<!-- page: {page.page_number} -->", page.text])
    markdown_path.write_text("\n\n".join(body).strip() + "\n", encoding="utf-8")
    return MarkdownConversion(markdown_path=markdown_path, page_count=len(parsed.pages))


class _ArticleHtmlParser(HTMLParser):
    """用标准库提取常见 HTML 正文标签，避免额外引入网页解析依赖。"""

    def __init__(self) -> None:
        """初始化标签栈和已经整理出的 Markdown 片段。"""

        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._current_tag: str | None = None
        self._buffer: list[str] = []
        self._blocks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """遇到开始标签时记录正文标签，脚本和样式内容直接忽略。"""

        del attrs
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth == 0 and lowered in {"p", "h1", "h2", "h3", "h4", "li", "blockquote"}:
            self._flush_current()
            self._current_tag = lowered

    def handle_endtag(self, tag: str) -> None:
        """遇到结束标签时把已收集的段落写入结果列表。"""

        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "svg"} and self._ignored_depth > 0:
            self._ignored_depth -= 1
            return
        if self._ignored_depth == 0 and self._current_tag == lowered:
            self._flush_current()

    def handle_data(self, data: str) -> None:
        """只收集正文标签内的文字，避免把导航菜单等内容写入论文正文。"""

        if self._ignored_depth == 0 and self._current_tag is not None:
            self._buffer.append(data)

    def to_markdown(self) -> str:
        """完成最后一个未闭合段落，并返回拼接后的 Markdown 正文。"""

        self._flush_current()
        return "\n\n".join(self._blocks)

    def _flush_current(self) -> None:
        """将当前标签中的文字转换成简单 Markdown 块，并清空临时缓存。"""

        if self._current_tag is None:
            return
        text = " ".join("".join(self._buffer).split())
        tag = self._current_tag
        self._buffer = []
        self._current_tag = None
        if not text:
            return
        if tag.startswith("h") and len(tag) == 2 and tag[1].isdigit():
            self._blocks.append("#" * min(int(tag[1]), 4) + " " + html.unescape(text))
        elif tag == "li":
            self._blocks.append("- " + html.unescape(text))
        elif tag == "blockquote":
            self._blocks.append("> " + html.unescape(text))
        else:
            self._blocks.append(html.unescape(text))
