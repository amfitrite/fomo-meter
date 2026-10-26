"""
Local HTTP server for the FOMO Meter dashboard.

Binds to 127.0.0.1 only. The desktop shell starts it on an ephemeral port and
loads the dashboard from the same origin, so no external web server, CDN, or
cloud API is required for the interface.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse

from sentiment_llm.config import (
    APP_NAME,
    APP_VERSION,
    ASSETS_DIR,
    HOST,
    LOG_DIR,
    WEB_DIR,
    ensure_dirs,
)
from sentiment_llm.engine import NoLiveDataError, SentimentEngine
from sentiment_llm.slm import SLMUnavailable

LOGGER = logging.getLogger("fomo_meter.web")


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = f"FOMOMeter/{APP_VERSION}"
    protocol_version = "HTTP/1.1"

    engine: SentimentEngine
    web_dir: Path
    assets_dir: Path

    # ------------------------------------------------------------------ http
    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self._common_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_get_api(parsed.path, parse_qs(parsed.query))
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self._send_json({"error": "Route tidak ditemukan."}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length > 2_000_000:
            self._send_json({"error": "Payload terlalu besar."}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "Body harus JSON yang valid."}, HTTPStatus.BAD_REQUEST)
            return
        if not isinstance(payload, dict):
            self._send_json({"error": "Body JSON harus berupa object."}, HTTPStatus.BAD_REQUEST)
            return
        self._handle_post_api(parsed.path, payload)

    # ------------------------------------------------------------------ api
    def _handle_get_api(self, path: str, query: Dict[str, list[str]]) -> None:
        try:
            if path == "/api/health":
                self._send_json({"ok": True, "app": APP_NAME, "version": APP_VERSION, "time": time.time()})
                return
            if path == "/api/status":
                self._send_json(self.engine.status())
                return
            if path == "/api/scan":
                ticker = (query.get("ticker") or ["BTCUSD"])[0]
                force = (query.get("force") or ["0"])[0].lower() in ("1", "true", "yes")
                self._send_json(self.engine.scan(ticker, force=force).model_dump(mode="json"))
                return
            if path == "/api/history":
                try:
                    limit = max(1, min(80, int((query.get("limit") or ["30"])[0])))
                except ValueError:
                    limit = 30
                entries = [entry.model_dump(mode="json") for entry in self.engine.history(limit)]
                self._send_json({"entries": entries})
                return
            if path == "/api/brief":
                ticker = (query.get("ticker") or ["BTCUSD"])[0]
                brief = self.engine.slm_brief(ticker)
                self._send_json(brief.model_dump(mode="json"))
                return
            self._send_json({"error": "Endpoint API tidak ditemukan."}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except SLMUnavailable as exc:
            self._send_json({"error": str(exc), "kind": "SLM_UNAVAILABLE"}, HTTPStatus.SERVICE_UNAVAILABLE)
        except NoLiveDataError as exc:
            self._send_json({"error": str(exc), "kind": "NO_LIVE_DATA"}, HTTPStatus.CONFLICT)
        except Exception as exc:
            LOGGER.exception("GET API failed: %s", path)
            self._send_json(
                {"error": f"Terjadi kesalahan internal ({type(exc).__name__}).", "kind": "INTERNAL"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_post_api(self, path: str, payload: Dict[str, Any]) -> None:
        try:
            if path == "/api/analyze":
                text = str(payload.get("text") or "")
                use_model = bool(payload.get("use_model", False))
                result = self.engine.analyze_text(text, use_model=use_model)
                self._send_json(result.model_dump(mode="json"))
                return
            if path == "/api/brief":
                ticker = str(payload.get("ticker") or "BTCUSD")
                brief = self.engine.slm_brief(ticker)
                self._send_json(brief.model_dump(mode="json"))
                return
            if path == "/api/scan":
                ticker = str(payload.get("ticker") or "BTCUSD")
                force = bool(payload.get("force", False))
                self._send_json(self.engine.scan(ticker, force=force).model_dump(mode="json"))
                return
            self._send_json({"error": "Endpoint API tidak ditemukan."}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except SLMUnavailable as exc:
            self._send_json({"error": str(exc), "kind": "SLM_UNAVAILABLE"}, HTTPStatus.SERVICE_UNAVAILABLE)
        except NoLiveDataError as exc:
            self._send_json({"error": str(exc), "kind": "NO_LIVE_DATA"}, HTTPStatus.CONFLICT)
        except Exception as exc:
            LOGGER.exception("POST API failed: %s", path)
            self._send_json(
                {"error": f"Terjadi kesalahan internal ({type(exc).__name__}).", "kind": "INTERNAL"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    # ----------------------------------------------------------------- static
    def _serve_static(self, request_path: str) -> None:
        decoded = unquote(request_path or "/")
        if decoded in ("", "/"):
            decoded = "/index.html"
        if decoded.startswith("/assets/"):
            base = self.assets_dir
            relative = decoded[len("/assets/"):]
        else:
            base = self.web_dir
            relative = decoded.lstrip("/")

        target = (base / relative).resolve()
        try:
            target.relative_to(base.resolve())
        except ValueError:
            self._send_json({"error": "Path tidak valid."}, HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            # SPA fallback for the dashboard root only; API 404s stay 404.
            if "." not in Path(relative).name:
                target = base / "index.html"
            else:
                self._send_json({"error": "File tidak ditemukan."}, HTTPStatus.NOT_FOUND)
                return

        mime, _encoding = mimetypes.guess_type(str(target))
        mime = mime or "application/octet-stream"
        if target.suffix == ".js":
            mime = "text/javascript; charset=utf-8"
        elif target.suffix == ".css":
            mime = "text/css; charset=utf-8"
        elif target.suffix == ".svg":
            mime = "image/svg+xml"
        elif target.suffix in (".html", ".htm"):
            mime = "text/html; charset=utf-8"
        elif target.suffix == ".json":
            mime = "application/json; charset=utf-8"

        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self._common_headers()
        self.end_headers()
        self.wfile.write(data)

    # ----------------------------------------------------------------- output
    def _common_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", (
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; font-src 'self'; object-src 'none'"
        ))

    def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._common_headers()
        self.end_headers()
        self.wfile.write(data)


class DashboardServer:
    def __init__(self, host: str = HOST, port: int = 0) -> None:
        ensure_dirs()
        self.host = host
        self.port = port
        self.engine = SentimentEngine()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> str:
        handler = type("BoundDashboardHandler", (DashboardHandler,), {
            "engine": self.engine,
            "web_dir": WEB_DIR,
            "assets_dir": ASSETS_DIR,
        })
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._server.daemon_threads = True
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="fomo-meter-http",
            daemon=True,
        )
        self._thread.start()
        LOGGER.info("Dashboard listening on http://%s:%s", self.host, self.port)
        return self.url

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        try:
            self.engine.close()
        except Exception:
            pass

    def serve_forever(self) -> None:
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def configure_logging() -> None:
    ensure_dirs()
    if LOGGER.handlers:
        return
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console)


def run_server(host: str = HOST, port: int = 0) -> None:
    configure_logging()
    server = DashboardServer(host=host, port=port)
    url = server.start()
    print(f"{APP_NAME} dashboard: {url}")
    print("Tekan Ctrl+C untuk berhenti.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
