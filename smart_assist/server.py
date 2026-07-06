from __future__ import annotations

import argparse
import cgi
import json
import tempfile
import threading
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from .pipeline import SmartAssistPipeline


class SmartAssistService:
    def __init__(self, pipeline: Optional[SmartAssistPipeline] = None) -> None:
        self.pipeline = pipeline or SmartAssistPipeline()
        self._lock = threading.RLock()

    def ingest_from_documents(self, documents: list[Any]) -> dict[str, Any]:
        with self._lock:
            results = self.pipeline.ingest(documents)
            return {
                "summary": self.pipeline.export_summary(),
                "documents_ingested": len(documents),
                "results": [self._serialize_extraction(result) for result in results],
            }

    def ingest_from_path(self, source: str | Path) -> dict[str, Any]:
        documents = self.pipeline.load_documents(source)
        return self.ingest_from_documents(documents)

    def ingest_from_uploads(self, uploads: list[tuple[str, bytes]]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="smart-assist-upload-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            for index, (filename, payload) in enumerate(uploads, start=1):
                safe_name = Path(filename or f"upload-{index}").name or f"upload-{index}"
                target = temp_dir / f"{index:03d}_{safe_name}"
                target.write_bytes(payload)
            return self.ingest_from_path(temp_dir)

    def query(self, text: str, top_k: int = 5) -> dict[str, Any]:
        with self._lock:
            result = self.pipeline.reason(text, top_k=top_k)
            return self._serialize_query_result(result)

    def close(self) -> None:
        self.pipeline.close()

    def _serialize_extraction(self, result: Any) -> dict[str, Any]:
        return {
            "document": self._serialize_value(result.document),
            "entities": [self._serialize_value(entity) for entity in result.entities],
            "relations": [self._serialize_value(relation) for relation in result.relations],
        }

    def _serialize_query_result(self, result: Any) -> dict[str, Any]:
        payload = self._serialize_value(result)
        payload["matched_entities"] = [self._serialize_value(entity) for entity in result.matched_entities]
        payload["similar_entities"] = [self._serialize_value(entity) for entity in result.similar_entities]
        payload["supporting_relations"] = [self._serialize_value(relation) for relation in result.supporting_relations]
        return payload

    def _serialize_value(self, value: Any) -> Any:
        if is_dataclass(value):
            return {key: self._serialize_value(val) for key, val in asdict(value).items()}
        if isinstance(value, dict):
            return {key: self._serialize_value(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        return value


class SmartAssistRequestHandler(BaseHTTPRequestHandler):
    server_version = "SmartAssistHTTP/1.0"

    @property
    def service(self) -> SmartAssistService:
        return self.server.service  # type: ignore[attr-defined]

    def do_OPTIONS(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path in {"/", "/health"}:
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "service": "smart_assist",
                    "neo4j_enabled": self.service.pipeline.neo4j_enabled,
                },
            )
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path == "/ingest":
            self._handle_ingest()
            return
        if self.path == "/query":
            self._handle_query()
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _handle_ingest(self) -> None:
        try:
            payload = self._read_request_payload()
            if payload["kind"] == "json":
                response = self._ingest_from_json(payload["data"])
            else:
                response = self._ingest_from_form(payload["form"])
            self._send_json(HTTPStatus.OK, {"status": "ok", **response})
        except Exception as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _handle_query(self) -> None:
        try:
            payload = self._read_request_payload()
            if payload["kind"] == "json":
                response = self._query_from_json(payload["data"])
            else:
                response = self._query_from_form(payload["form"])
            self._send_text(HTTPStatus.OK, response)
        except Exception as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _read_request_payload(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        if content_type.startswith("application/json"):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            return {"kind": "json", "data": json.loads(raw.decode("utf-8") or "{}")}

        if content_type.startswith("multipart/form-data"):
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
                keep_blank_values=True,
            )
            return {"kind": "form", "form": form}

        raise ValueError("Unsupported content type. Use application/json or multipart/form-data.")

    def _ingest_from_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "documents" in payload:
            documents = self.service.pipeline.load_documents(payload["documents"])
            return self.service.ingest_from_documents(documents)
        if "input" in payload:
            return self.service.ingest_from_path(payload["input"])
        raise ValueError("Provide either 'documents' or 'input' in the request body.")

    def _ingest_from_form(self, form: cgi.FieldStorage) -> dict[str, Any]:
        uploads = self._extract_uploads(form)
        if uploads:
            return self.service.ingest_from_uploads(uploads)

        input_path = self._form_value(form, "input")
        if input_path:
            return self.service.ingest_from_path(input_path)

        documents_raw = self._form_value(form, "documents")
        if documents_raw:
            documents = self.service.pipeline.load_documents(json.loads(documents_raw))
            return self.service.ingest_from_documents(documents)

        raise ValueError("Upload one or more files or provide an 'input' field.")

    def _query_from_json(self, payload: dict[str, Any]) -> str:
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("The query endpoint requires a non-empty 'text' field.")
        top_k = int(payload.get("top_k", 5))
        if "documents" in payload:
            documents = self.service.pipeline.load_documents(payload["documents"])
            self.service.ingest_from_documents(documents)
        elif "input" in payload:
            self.service.ingest_from_path(payload["input"])
        return self.service.query(text, top_k=top_k)["answer"]

    def _query_from_form(self, form: cgi.FieldStorage) -> str:
        text = self._form_value(form, "text")
        if not text:
            raise ValueError("The query endpoint requires a non-empty 'text' field.")
        top_k_raw = self._form_value(form, "top_k")
        top_k = int(top_k_raw) if top_k_raw else 5

        uploads = self._extract_uploads(form)
        if uploads:
            self.service.ingest_from_uploads(uploads)
        else:
            input_path = self._form_value(form, "input")
            if input_path:
                self.service.ingest_from_path(input_path)
            documents_raw = self._form_value(form, "documents")
            if documents_raw:
                documents = self.service.pipeline.load_documents(json.loads(documents_raw))
                self.service.ingest_from_documents(documents)

        return self.service.query(text, top_k=top_k)["answer"]

    def _extract_uploads(self, form: cgi.FieldStorage) -> list[tuple[str, bytes]]:
        uploads: list[tuple[str, bytes]] = []
        if not getattr(form, "list", None):
            return uploads
        for item in form.list or []:
            if not getattr(item, "filename", None):
                continue
            uploads.append((item.filename, item.file.read()))
        return uploads

    def _form_value(self, form: cgi.FieldStorage, name: str) -> str:
        if name not in form:
            return ""
        field = form[name]
        if isinstance(field, list):
            field = field[0]
        return str(getattr(field, "value", "")).strip()

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_text(self, status: HTTPStatus, body: str) -> None:
        raw = body.encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"status": "error", "error": message})

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - BaseHTTPRequestHandler API
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smart-assist-server", description="Smart Assist REST service")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    service = SmartAssistService()
    server = ThreadingHTTPServer((args.host, args.port), SmartAssistRequestHandler)
    server.service = service  # type: ignore[attr-defined]
    print(f"Smart Assist REST service running on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
