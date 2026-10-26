"""
Native desktop shell for FOMO Meter.

PySide6 + QtWebEngine render the local dashboard in a real desktop window. The
HTTP server is bound to 127.0.0.1 on an ephemeral port. External evidence links
open in the default browser; the app window itself never navigates away from the
local dashboard.

The window is intentionally frameless: the web top bar acts as custom chrome.
Window commands travel from the local page through a narrowly scoped console
bridge (``FOMO_WINDOW|...``). Native dragging and resizing stay in Qt, so the
web layer never gets operating-system privileges beyond those fixed commands.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Set these before importing QtWebEngine. Hardware acceleration keeps the radar
# animations and the real FPS meter smooth; Chromium falls back to software
# rendering automatically when a machine has no usable GPU. The sandbox flag
# keeps QtWebEngine stable in restricted Windows sessions.
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")

from PySide6.QtCore import QEvent, QPoint, QTimer, QUrl, Qt
from PySide6.QtGui import QColor, QCursor, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QSizeGrip
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView

from sentiment_llm.config import APP_NAME, APP_TAGLINE, APP_VERSION, ASSETS_DIR, ensure_dirs
from sentiment_llm.web_dashboard import DashboardServer, configure_logging

WINDOW_COMMAND_PREFIX = "FOMO_WINDOW|"


class AppWebPage(QWebEnginePage):
    """Open external links in the browser and receive fixed window commands."""

    def __init__(self, host: "MainWindow", parent=None) -> None:
        super().__init__(parent)
        self.host = host

    def acceptNavigationRequest(self, url: QUrl, navigation_type, is_main_frame: bool) -> bool:  # noqa: N802
        if is_main_frame and url.scheme() in ("http", "https"):
            host = (url.host() or "").lower()
            if host not in ("127.0.0.1", "localhost", "::1"):
                from PySide6.QtGui import QDesktopServices

                QDesktopServices.openUrl(url)
                return False
        return super().acceptNavigationRequest(url, navigation_type, is_main_frame)

    def javaScriptConsoleMessage(self, level, message: str, lineNumber: int, sourceID: str) -> None:  # noqa: N802
        # The command vocabulary is deliberately tiny and every command is
        # validated by MainWindow. Ignore console traffic from anything except
        # the local dashboard.
        local_source = "127.0.0.1" in sourceID or "localhost" in sourceID
        if local_source and message.startswith(WINDOW_COMMAND_PREFIX):
            self.host.handle_window_command(message[len(WINDOW_COMMAND_PREFIX):].strip())
            return
        super().javaScriptConsoleMessage(level, message, lineNumber, sourceID)


class MainWindow(QMainWindow):
    def __init__(self, url: str, server: DashboardServer) -> None:
        super().__init__()
        self.server = server
        self._dragging = False
        self._drag_offset = QPoint()
        self._native_style_applied = False

        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowTitle(f"{APP_NAME} · {APP_TAGLINE}")
        self.setMinimumSize(1120, 720)
        self.resize(1460, 920)

        icon_path = ASSETS_DIR / "fomo_meter.ico"
        if not icon_path.exists():
            icon_path = ASSETS_DIR / "fomo_meter_256.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.view = QWebEngineView(self)
        self.view.setPage(AppWebPage(self, self.view))
        self.view.page().setBackgroundColor(QColor("#070b10"))
        desktop_url = QUrl(url)
        desktop_url.setFragment("desktop")
        self.view.setUrl(desktop_url)
        self.setCentralWidget(self.view)

        # A small native grip in the bottom-right corner restores resize support
        # that the frameless window would otherwise lose on Windows.
        self.size_grip = QSizeGrip(self)
        self.size_grip.setFixedSize(18, 18)
        self.size_grip.setStyleSheet("background: transparent;")
        self.size_grip.raise_()

        QShortcut(QKeySequence("F5"), self, activated=self.reload_page)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self.reload_page)
        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)
        QShortcut(QKeySequence("F11"), self, activated=self.toggle_fullscreen)
        QShortcut(QKeySequence("Escape"), self, activated=self.exit_fullscreen)

    # ------------------------------------------------------------- window IPC
    def handle_window_command(self, action: str) -> None:
        if action == "minimize":
            self.showMinimized()
        elif action == "toggle_max":
            if self.isFullScreen() or self.isMaximized():
                self.showNormal()
            else:
                self.showMaximized()
        elif action == "close":
            self.close()
        elif action == "drag_start":
            self.begin_window_drag()
        elif action == "drag_move":
            self.move_window_drag()
        elif action == "drag_end":
            self._dragging = False

    def begin_window_drag(self) -> None:
        if self.isMaximized() or self.isFullScreen():
            self._dragging = False
            return
        handle = self.windowHandle()
        if handle is not None:
            try:
                if handle.startSystemMove():
                    self._dragging = False
                    return
            except Exception:
                pass
        # Fallback for platforms where startSystemMove is unavailable. The JS
        # layer sends throttled drag_move commands while the pointer is down.
        self._dragging = True
        self._drag_offset = QCursor.pos() - self.frameGeometry().topLeft()

    def move_window_drag(self) -> None:
        if not self._dragging:
            return
        self.move(QCursor.pos() - self._drag_offset)

    # ------------------------------------------------------------- appearance
    def apply_native_style(self) -> None:
        """Ask Windows 11 for rounded corners; harmless on other platforms."""
        if sys.platform != "win32" or self._native_style_applied:
            return
        try:
            import ctypes

            hwnd = ctypes.c_void_p(int(self.winId()))
            rounded = ctypes.c_int(2)  # DWMWCP_ROUND
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 33, ctypes.byref(rounded), ctypes.sizeof(rounded)
            )
            dark = ctypes.c_int(1)  # DWMWA_USE_IMMERSIVE_DARK_MODE
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 20, ctypes.byref(dark), ctypes.sizeof(dark)
            )
            self._native_style_applied = True
        except Exception:
            pass

    def update_size_grip(self) -> None:
        if not hasattr(self, "size_grip"):
            return
        if self.isMaximized() or self.isFullScreen():
            self.size_grip.hide()
            return
        self.size_grip.show()
        self.size_grip.move(max(0, self.width() - 18), max(0, self.height() - 18))
        self.size_grip.raise_()

    def sync_window_state(self) -> None:
        self.update_size_grip()
        if not hasattr(self, "view"):
            return
        maximized = "true" if (self.isMaximized() or self.isFullScreen()) else "false"
        try:
            self.view.page().runJavaScript(
                f"window.fomoSetMaximized && window.fomoSetMaximized({maximized});"
            )
        except Exception:
            pass

    # ------------------------------------------------------------- Qt events
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.apply_native_style()
        self.update_size_grip()
        QTimer.singleShot(0, self.sync_window_state)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.update_size_grip()

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            QTimer.singleShot(0, self.sync_window_state)

    def reload_page(self) -> None:
        self.view.reload()

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def exit_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._dragging = False
        try:
            self.server.stop()
        finally:
            super().closeEvent(event)


def run_desktop() -> int:
    ensure_dirs()
    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_NAME)
    app.setStyle("Fusion")

    server = DashboardServer()
    try:
        url = server.start()
    except Exception as exc:
        QMessageBox.critical(None, APP_NAME, f"Server lokal gagal dimulai:\n{type(exc).__name__}: {exc}")
        return 1

    window = MainWindow(url, server)
    window.show()
    # Keep the server alive for the lifetime of the event loop.
    app.aboutToQuit.connect(server.stop)
    return app.exec()


def main() -> None:
    raise SystemExit(run_desktop())


if __name__ == "__main__":
    main()
