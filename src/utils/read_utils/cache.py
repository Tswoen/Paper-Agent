from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.paper_retrieval.models import PaperDocument


JsonObject = dict[str, Any]


def paper_cache_dir(base_dir: str | Path, paper: PaperDocument) -> Path:
    """返回单篇论文的缓存目录。

    中文注释：需求里希望用 paperId 作为目录名。Windows 文件名不能包含斜杠、
    冒号这类字符，所以这里只做很薄的一层清理，不再用哈希隐藏原始编号。
    """

    paper_id = str(paper.paperId or paper.id or paper.doi or paper.title).strip()
    return Path(base_dir) / safe_cache_name(paper_id)


def safe_cache_name(value: str) -> str:
    """把论文编号变成适合放进路径里的名字。

    中文注释：这里保留字母、数字、短横线、下划线和点号；其它字符统一换成
    下划线。这样目录名仍然能看出 paperId，大多数情况下也能直接复制查看。
    """

    cleaned = "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in value)
    return cleaned[:160] or "paper"


def write_metadata(cache_dir: Path, paper: PaperDocument, *, source_url: str | None, content_type: str | None) -> Path:
    """把论文元数据写入缓存目录里的 metadata.json。

    中文注释：只有下载到全文后才会调用这个函数，所以不会给下载失败的论文
    创建空缓存。metadata 里同时放论文原始信息和全文来源，后面分析节点不用
    再回头猜这篇论文是从哪里来的。
    """

    payload: JsonObject = {
        "paperId": paper.paperId or paper.id,
        "paper": paper.to_dict(),
        "fulltext": {
            "source_url": source_url,
            "content_type": content_type,
        },
    }
    path = cache_dir / "metadata.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_cached_source_url(cache_dir: Path) -> str | None:
    """从缓存里的 metadata.json 读回全文来源地址。

    中文注释：旧缓存里可能还有 source.json，所以这里顺手兼容一下。主流程新写入
    的都是 metadata.json。
    """

    for name in ("metadata.json", "source.json"):
        try:
            payload = json.loads((cache_dir / name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        fulltext = payload.get("fulltext") if isinstance(payload, dict) else None
        if isinstance(fulltext, dict) and fulltext.get("source_url"):
            return str(fulltext["source_url"])
        if isinstance(payload, dict) and payload.get("source_url"):
            return str(payload["source_url"])
    return None
