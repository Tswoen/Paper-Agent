import unittest

from src.paper_retrieval.connectors.base import PaperSearchConnector
from src.paper_retrieval.models import PaperDocument, SearchRequest
from src.paper_retrieval.service import PaperSearchService


class _FakeConnector(PaperSearchConnector):
    """测试用 connector，用来稳定验证编排层行为。"""

    def __init__(self, source_name: str, items: list[PaperDocument]):
        self.source_name = source_name
        self.items = items
        self.seen_requests: list[SearchRequest] = []

    def search(self, request: SearchRequest) -> list[PaperDocument]:
        """记录请求并返回预设结果，避免测试依赖真实外部网络。"""

        self.seen_requests.append(request)
        return list(self.items)


class PaperSearchServiceTest(unittest.TestCase):
    def test_single_source_search_returns_standardized_response(self):
        service = PaperSearchService(
            connectors={
                "openalex": _FakeConnector(
                    "openalex",
                    [
                        PaperDocument(
                            id="oa-1",
                            title="Graph Neural Networks",
                            authors=["Alice"],
                            year=2024,
                            source="openalex",
                        )
                    ],
                )
            }
        )

        response = service.search("graph neural networks", source="openalex", limit=5)

        self.assertEqual(response.sources_used, ["openalex"])
        self.assertEqual(response.source_results["openalex"], 1)
        self.assertEqual(response.total, 1)
        self.assertEqual(response.papers[0].title, "Graph Neural Networks")

    def test_multi_source_search_deduplicates_by_doi(self):
        duplicate_a = PaperDocument(
            id="a1",
            title="Shared Paper",
            authors=["Alice"],
            doi="10.1000/shared",
            source="openalex",
        )
        duplicate_b = PaperDocument(
            id="b1",
            title="Shared Paper",
            authors=["Bob"],
            doi="10.1000/shared",
            source="semantic_scholar",
        )
        unique = PaperDocument(
            id="c1",
            title="Unique Paper",
            authors=["Carol"],
            source="arxiv",
        )
        service = PaperSearchService(
            connectors={
                "openalex": _FakeConnector("openalex", [duplicate_a]),
                "semantic_scholar": _FakeConnector("semantic_scholar", [duplicate_b]),
                "arxiv": _FakeConnector("arxiv", [unique]),
            }
        )

        response = service.search("shared query", limit=5)

        self.assertEqual(response.total, 2)
        self.assertEqual(sorted(response.source_results.keys()), ["arxiv", "openalex", "semantic_scholar"])

    def test_multi_source_search_uses_full_limit_for_each_connector(self):
        service = PaperSearchService(
            connectors={
                "openalex": _FakeConnector("openalex", []),
                "arxiv": _FakeConnector("arxiv", []),
            }
        )

        response = service.search("shared query", limit=5, truncate=False)

        self.assertEqual(response.total, 0)
        self.assertEqual(service._connectors["openalex"].seen_requests[0].limit, 5)
        self.assertEqual(service._connectors["arxiv"].seen_requests[0].limit, 5)

    def test_invalid_source_returns_error(self):
        service = PaperSearchService(connectors={})

        response = service.search("anything", source="missing", limit=3)

        self.assertIn("sources", response.errors)
        self.assertEqual(response.total, 0)


if __name__ == "__main__":
    unittest.main()
