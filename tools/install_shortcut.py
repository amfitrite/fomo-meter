"""
Create a Windows desktop shortcut for FOMO Meter.

Targets pythonw.exe + run_gui.pyw so the app opens without a console window.
Uses the Windows Script Host COM object through PowerShell; no extra Python
packages are required.

Run:
    python tools/install_shortcut.py
    python tools/install_shortcut.py --start-menu
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_GUI = ROOT / "run_gui.pyw"
ICON = ROOT / "assets" / "fomo_meter.ico"


def _pythonw() -> Path:
    executable = Path(sys.executable)
    candidate = executable.with_name("pythonw.exe")
    if candidate.exists():
        return candidate
    return executable


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def create_shortcut(path: Path, target: Path, arguments: str, working_dir: Path, icon: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    script = f"""
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut({_ps_quote(str(path))})
$Shortcut.TargetPath = {_ps_quote(str(target))}
$Shortcut.Arguments = {_ps_quote(arguments)}
$Shortcut.WorkingDirectory = {_ps_quote(str(working_dir))}
$Shortcut.IconLocation = {_ps_quote(str(icon))}
$Shortcut.Description = 'FOMO Meter - Crowd Psychology Radar'
$Shortcut.Save()
"""
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"PowerShell gagal membuat shortcut: {completed.stderr.strip() or completed.stdout.strip()}"
        )


def desktop_path() -> Path:
    # USERPROFILE\Desktop is reliable even when OneDrive is enabled.
    return Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"


def start_menu_path() -> Path:
    appdata = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the FOMO Meter desktop shortcut.")
    parser.add_argument("--start-menu", action="store_true", help="Also create a Start Menu shortcut.")
    parser.add_argument("--name", default="FOMO Meter", help="Shortcut name without .lnk")
    args = parser.parse_args()

    if not RUN_GUI.exists():
        print(f"run_gui.pyw tidak ditemukan: {RUN_GUI}", file=sys.stderr)
        return 1
    if not ICON.exists():
        print(f"Icon tidak ditemukan: {ICON}. Jalankan python tools/make_icon.py dulu.", file=sys.stderr)
        return 1

    target = _pythonw()
    # Keep the path quoted because the project directory contains a space.
    arguments = f'"{RUN_GUI}"'
    shortcuts = [desktop_path() / f"{args.name}.lnk"]
    if args.start_menu:
        shortcuts.append(start_menu_path() / f"{args.name}.lnk")

    for shortcut in shortcuts:
        try:
            create_shortcut(shortcut, target, arguments, ROOT, ICON)
            print(f"Shortcut dibuat: {shortcut}")
        except Exception as exc:
            print(f"Gagal membuat {shortcut}: {exc}", file=sys.stderr)
            return 1
    print(f"Target: {target} {arguments}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
