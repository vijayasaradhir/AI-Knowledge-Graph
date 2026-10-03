from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smart_assist.pipeline import SmartAssistPipeline
from smart_assist.server import SmartAssistRequestHandler, SmartAssistService


class SmartAssistServerTests(unittest.TestCase):
    def setUp(self) -> None:
        pipeline = SmartAssistPipeline()
        pipeline.extractor._client = None
        pipeline._openai_client = None
        self.service = SmartAssistService(pipeline)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), SmartAssistRequestHandler)
        self.server.service = self.service  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        time.sleep(0.05)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.service.close()

    def test_ingest_multipart_and_query(self) -> None:
        body, content_type = self._build_multipart(
            fields={},
            files=[
                ("files", "collab-a.txt", b"OpenAI collaborated with Microsoft on AI research."),
                ("files", "collab-b.txt", b"Microsoft acquired GitHub in 2018."),
            ],
        )

        ingest_response = self._request(
            "/ingest",
            method="POST",
            data=body,
            content_type=content_type,
        )
        self.assertEqual(ingest_response["status"], "ok")
        self.assertEqual(ingest_response["documents_ingested"], 2)
        self.assertGreaterEqual(ingest_response["summary"]["entities"], 2)

        query_response = self._request_text(
            "/query",
            method="POST",
            data=json.dumps({"text": "Who collaborates with OpenAI?"}).encode("utf-8"),
            content_type="application/json",
        )
        self.assertIn("OpenAI", query_response)
        self.assertIn("Microsoft", query_response)

    def _request(self, path: str, method: str, data: bytes, content_type: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": content_type},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def _request_text(self, path: str, method: str, data: bytes, content_type: str) -> str:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": content_type},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read().decode("utf-8")

    def _build_multipart(
        self,
        fields: dict[str, str],
        files: list[tuple[str, str, bytes]],
    ) -> tuple[bytes, str]:
        boundary = "----SmartAssistBoundary7b1a0f"
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.append(f"--{boundary}\r\n".encode("utf-8"))
            chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
            chunks.append(value.encode("utf-8"))
            chunks.append(b"\r\n")
        for field_name, filename, payload in files:
            chunks.append(f"--{boundary}\r\n".encode("utf-8"))
            chunks.append(
                f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8")
            )
            chunks.append(b"Content-Type: application/octet-stream\r\n\r\n")
            chunks.append(payload)
            chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


if __name__ == "__main__":
    unittest.main()
