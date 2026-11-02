"""SPA 静态回退与后端 API 404 的边界测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import create_app
from src.repositories.sessions.sqlite import SQLiteSessionRepository
from src.repositories.settings.json import SettingsRepository


class SpaFallbackTest(unittest.TestCase):
    """未知 API 路径不能被前端 index.html 吞掉。"""

    def test_unknown_api_paths_return_404_while_spa_routes_still_work(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dist_dir = root / "front" / "dist"
            dist_dir.mkdir(parents=True)
            (dist_dir / "index.html").write_text("<!doctype html><html>spa</html>", encoding="utf-8")

            os.chdir(root)
            try:
                with patch.dict(os.environ, {"PAPERS_LOG_ENABLE_FILE": "0"}, clear=False):
                    app = create_app(
                        settings_repo=SettingsRepository(initial={}),
                        sessions_repo=SQLiteSessionRepository(storage_root=root / "data"),
                    )
                client = TestClient(app)

                for path in ("/api/does-not-exist", "/api", "/webui/does-not-exist", "/webui"):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 404, path)
                    self.assertEqual(response.json()["detail"], "not found", path)

                spa_response = client.get("/some/spa/route")
                self.assertEqual(spa_response.status_code, 200)
                self.assertIn("spa", spa_response.text)

                # /apiary 只是普通前端路径，不应被当成 /api 前缀。
                self.assertEqual(client.get("/apiary").status_code, 200)
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
