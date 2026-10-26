"""
Capture the local FOMO Meter dashboard for visual QA.

Run:
    python tools/screenshot.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Hardware acceleration keeps the real FPS meter and radar motion smooth. It is
# also what users see when they launch the desktop shortcut.
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView

from sentiment_llm.config import SCREENSHOT_DIR, ensure_dirs
from sentiment_llm.web_dashboard import DashboardServer, configure_logging


class CapturePage(QWebEnginePage):
    pass


ANALYZE_TEXT = "Bitcoin breakout di atas resistance, volume besar, serok kawan, target next leg up"


def main() -> int:
    ensure_dirs()
    configure_logging()
    app = QApplication(sys.argv)
    server = DashboardServer()
    url = server.start()
    view = QWebEngineView()
    view.setPage(CapturePage(view))
    view.page().setBackgroundColor(QColor("#070b10"))
    view.resize(1500, 1040)
    view.setWindowTitle("FOMO Meter screenshot")
    view.setUrl(url + "#desktop")
    view.show()

    output = SCREENSHOT_DIR / "dashboard.png"
    output_bottom = SCREENSHOT_DIR / "dashboard_bottom.png"
    output_analyzer = SCREENSHOT_DIR / "analyzer.png"
    output_brief = SCREENSHOT_DIR / "brief.png"

    def save(path: Path) -> None:
        image = view.grab()
        image.save(str(path))
        print(f"Saved {path} ({image.width()}x{image.height()})")

    def capture_top() -> None:
        save(output)
        view.page().runJavaScript("window.scrollTo(0, document.body.scrollHeight);")
        QTimer.singleShot(1000, capture_bottom)

    def capture_bottom() -> None:
        save(output_bottom)
        view.page().runJavaScript(
            "document.querySelector('.analyzer-panel').scrollIntoView({block: 'center'});"
        )
        QTimer.singleShot(600, run_analyzer)

    def run_analyzer() -> None:
        script = f"""
        (() => {{
          const input = document.querySelector('#analysisInput');
          input.value = {ANALYZE_TEXT!r};
          input.dispatchEvent(new Event('input', {{bubbles: true}}));
          document.querySelector('#analyzeBtn').click();
          return true;
        }})()
        """
        view.page().runJavaScript(script)
        QTimer.singleShot(2500, capture_analyzer)

    def capture_analyzer() -> None:
        save(output_analyzer)
        view.page().runJavaScript("document.querySelector('#briefBtn').click(); true")
        QTimer.singleShot(2500, capture_brief)

    def capture_brief() -> None:
        view.page().runJavaScript(
            "document.querySelector('#briefResult').scrollIntoView({block: 'center'});"
        )
        QTimer.singleShot(700, save_and_finish)

    def save_and_finish() -> None:
        save(output_brief)
        stop()

    def stop() -> None:
        server.stop()
        app.quit()

    QTimer.singleShot(14000, capture_top)
    code = app.exec()
    try:
        server.stop()
    except Exception:
        pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())
