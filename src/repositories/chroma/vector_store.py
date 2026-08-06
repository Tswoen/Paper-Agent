from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb


MetadataValue = str | int | float | bool

# 中文注释：当前使用的 Chroma 本地客户端会在同一进程内共享一份底层数据库对象。
# 多个线程同时创建或关闭同一个本地数据库时，Chroma 可能提前关闭仍在使用的对象。
# 这里用一把锁把“创建客户端到关闭客户端”这段时间隔开，避免论文并发建立索引时互相影响。
_LOCAL_CHROMA_CLIENT_LOCK = threading.RLock()


@dataclass(slots=True)
class ChromaConnection:
    """保存连接本地 Chroma 数据库需要的最少信息。"""

    persist_path: str | Path
    collection_name: str


@dataclass(slots=True)
class VectorUpsertItem:
    """表示准备写入 Chroma 的一条正文切片。"""

    id: str
    embedding: list[float]
    metadata: dict[str, MetadataValue]
    document: str | None = None


@dataclass(slots=True)
class VectorQueryResult:
    """表示从 Chroma 查询出来的一条切片结果。"""

    id: str
    distance: float | None
    metadata: dict[str, Any]
    document: str | None


class ChromaVectorStore:
    """封装 Chroma 的常用操作，避免工作流节点直接依赖 Chroma 的细节。"""

    def __init__(self, connection: ChromaConnection) -> None:
        """根据本地目录和 collection 名称创建向量库操作对象。"""

        self._lock_is_held = False
        _LOCAL_CHROMA_CLIENT_LOCK.acquire()
        self._lock_is_held = True
        try:
            self.connection = connection
            self.persist_path = Path(connection.persist_path)
            self.collection_name = connection.collection_name.strip()
            if not self.collection_name:
                raise ValueError("Chroma collection 名称不能为空")

            # 中文注释：Chroma 会把数据库文件保存在 persist_path 目录下。
            # 这里提前创建目录，是为了让错误更早暴露，也方便用户直接看到向量库保存在哪里。
            self.persist_path.mkdir(parents=True, exist_ok=True)

            # 中文注释：PersistentClient 表示“本地持久保存”的 Chroma 客户端。
            # 也就是说程序退出后，写入的向量仍然会留在这个目录中。
            self._client = chromadb.PersistentClient(path=str(self.persist_path))

            # 中文注释：collection 可以理解为一张专门存论文切片的表。
            # 如果已经存在就直接复用，不存在就自动创建，避免每次运行都要手动初始化。
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
        except Exception:
            # 中文注释：如果创建客户端的中途失败，也必须归还锁；否则后续论文无法继续建立索引。
            self._release_client_lock()
            raise

    def close(self) -> None:
        """关闭 Chroma 客户端占用的本地文件句柄。"""

        # 中文注释：Windows 上如果不关闭 Chroma 客户端，临时目录或测试目录可能会因为
        # 数据库文件还被占用而无法删除。正常服务长期运行时不一定需要手动调用，
        # 但脚本验证或短生命周期任务结束前调用它会更稳。
        try:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
        finally:
            # 中文注释：无论关闭时是否报错，都要归还锁，避免后续索引任务永久等待。
            self._release_client_lock()

        # 中文注释：不能在这里调用 clear_system_cache()。这个方法会把整个程序里
        # 所有 Chroma 客户端共用的记录一起清空；当多篇论文同时建立索引时，其他还在
        # 写入的客户端会因此失效，并报出找不到 vector_store 或 bindings 的错误。

    def _release_client_lock(self) -> None:
        """归还当前对象持有的 Chroma 操作锁。"""

        if self._lock_is_held:
            self._lock_is_held = False
            _LOCAL_CHROMA_CLIENT_LOCK.release()

    def upsert(self, items: list[VectorUpsertItem]) -> int:
        """新增或更新切片；如果 id 已存在，Chroma 会用新内容覆盖旧内容。"""

        if not items:
            return 0

        # 中文注释：Chroma 的 upsert 需要把 id、正文、元数据和向量分别放成列表。
        # 这里统一转换，调用方只需要传 VectorUpsertItem，不必知道 Chroma 的参数格式。
        self._collection.upsert(
            ids=[item.id for item in items],
            embeddings=[item.embedding for item in items],
            metadatas=[item.metadata for item in items],
            documents=[item.document or "" for item in items],
        )
        return len(items)

    def delete_by_ids(self, ids: list[str]) -> int:
        """按切片 id 删除数据，返回本次请求删除的 id 数量。"""

        clean_ids = [item_id for item_id in ids if item_id]
        if not clean_ids:
            return 0
        self._collection.delete(ids=clean_ids)
        return len(clean_ids)

    def delete_by_paper_id(self, paper_id: str) -> None:
        """删除某篇论文的全部旧切片，方便重新索引时避免重复数据。"""

        clean_paper_id = paper_id.strip()
        if not clean_paper_id:
            return

        # 中文注释：重新处理同一篇论文时，切片数量可能变多也可能变少。
        # 先删掉旧切片再写入新切片，可以避免旧版本多出来的切片残留在数据库里。
        self._collection.delete(where={"paper_id": clean_paper_id})

    def get_by_ids(self, ids: list[str]) -> list[VectorQueryResult]:
        """按切片 id 读取 Chroma 中保存的正文和元数据。"""

        clean_ids = [item_id for item_id in ids if item_id]
        if not clean_ids:
            return []
        payload = self._collection.get(ids=clean_ids, include=["metadatas", "documents"])
        return _results_from_get_payload(payload)

    def query_by_embedding(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorQueryResult]:
        """用已经生成好的向量查询最相近的正文切片。"""

        if not embedding:
            return []

        # 中文注释：这个函数接收“已经算好的向量”，而不是直接接收普通文本。
        # 这样向量库模块只负责存取和查询，embedding 服务仍然由现有代码统一管理。
        payload = self._collection.query(
            query_embeddings=[embedding],
            n_results=max(1, top_k),
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        return _results_from_query_payload(payload)


def make_chroma_store(vector_store_path: str | Path, collection_name: str) -> ChromaVectorStore:
    """用配置里的目录和 collection 名称创建 Chroma 操作对象。"""

    return ChromaVectorStore(ChromaConnection(persist_path=vector_store_path, collection_name=collection_name))


def normalize_chroma_metadata(metadata: dict[str, Any]) -> dict[str, MetadataValue]:
    """把元数据整理成 Chroma 能稳定保存的简单类型。"""

    normalized: dict[str, MetadataValue] = {}
    for key, value in metadata.items():
        clean_key = str(key).strip()
        if not clean_key:
            continue

        # 中文注释：Chroma 的 metadata 只能保存字符串、数字和布尔值这类简单内容。
        # None、列表、字典等复杂内容不能直接放进去，否则写入时会报错。
        if value is None:
            if clean_key in {"page_start", "page_end", "line_start", "line_end", "chunk_index", "schema_version"}:
                normalized[clean_key] = -1
            else:
                normalized[clean_key] = ""
            continue
        if isinstance(value, bool):
            normalized[clean_key] = value
            continue
        if isinstance(value, int | float):
            normalized[clean_key] = value
            continue
        if isinstance(value, Path):
            normalized[clean_key] = str(value)
            continue
        if isinstance(value, str):
            normalized[clean_key] = value
            continue

        # 中文注释：遇到意外类型时，不把原对象直接塞进数据库。
        # 转成字符串后至少还能保留可读信息，也能避免 Chroma 写入失败。
        normalized[clean_key] = str(value)
    return normalized


def _results_from_get_payload(payload: dict[str, Any]) -> list[VectorQueryResult]:
    """把 Chroma get 返回的字典整理成项目内部更容易使用的对象。"""

    ids = list(payload.get("ids") or [])
    metadatas = list(payload.get("metadatas") or [])
    documents = list(payload.get("documents") or [])
    results: list[VectorQueryResult] = []
    for index, item_id in enumerate(ids):
        results.append(
            VectorQueryResult(
                id=str(item_id),
                distance=None,
                metadata=dict(metadatas[index] or {}) if index < len(metadatas) else {},
                document=str(documents[index]) if index < len(documents) and documents[index] is not None else None,
            )
        )
    return results


def _results_from_query_payload(payload: dict[str, Any]) -> list[VectorQueryResult]:
    """把 Chroma query 返回的嵌套列表整理成扁平结果。"""

    ids = _first_list(payload.get("ids"))
    metadatas = _first_list(payload.get("metadatas"))
    documents = _first_list(payload.get("documents"))
    distances = _first_list(payload.get("distances"))
    results: list[VectorQueryResult] = []
    for index, item_id in enumerate(ids):
        distance = distances[index] if index < len(distances) else None
        results.append(
            VectorQueryResult(
                id=str(item_id),
                distance=float(distance) if isinstance(distance, int | float) else None,
                metadata=dict(metadatas[index] or {}) if index < len(metadatas) else {},
                document=str(documents[index]) if index < len(documents) and documents[index] is not None else None,
            )
        )
    return results


def _first_list(value: Any) -> list[Any]:
    """Chroma query 会按查询向量分组，这里只取第一个查询向量对应的结果。"""

    if not isinstance(value, list) or not value:
        return []
    first = value[0]
    return list(first) if isinstance(first, list) else []
