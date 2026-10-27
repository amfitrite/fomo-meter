import sys

if sys.platform == "win32":
    try:
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:
        pass

from fomo_meter.desktop import run_desktop

if __name__ == "__main__":
    raise SystemExit(run_desktop())
