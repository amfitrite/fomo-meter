"""
FOMO Meter command-line entry point.

Examples:
    python run.py                 # desktop GUI
    python run.py gui
    python run.py serve --port 8085
    python run.py scan --ticker BTCUSD
    python run.py analyze --text "BTC breakout, serok kawan" --model
    python run.py brief --ticker BTCUSD
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fomo_meter.config import TICKERS, ensure_dirs


def command_gui(_args: argparse.Namespace) -> int:
    from fomo_meter.desktop import run_desktop
    return run_desktop()


def command_serve(args: argparse.Namespace) -> int:
    from fomo_meter.web_dashboard import run_server
    run_server(port=args.port)
    return 0


def command_scan(args: argparse.Namespace) -> int:
    from fomo_meter.engine import SentimentEngine
    engine = SentimentEngine()
    try:
        scan = engine.scan(args.ticker, force=args.force)
        if args.json:
            print(json.dumps(scan.model_dump(mode="json"), ensure_ascii=False, indent=2))
        else:
            print(f"{scan.label} ({scan.ticker}) · {scan.consensus} · intensity {scan.intensity}")
            print(f"feed={scan.feed_status.value} · live={scan.live_sources} · analyzed={scan.analyzed_posts}/{scan.total_posts}")
            for source in scan.sources:
                print(f"  {source.name:14s} {source.status.value:10s} {source.stance.value:5s} "
                      f"conf={source.confidence:.2f} analyzed={source.analyzed_count}")
                if source.reason:
                    print(f"    {source.reason}")
            for gate in scan.gates:
                print(f"  [{gate.state.value}] {gate.id} {gate.name}: {gate.detail}")
    finally:
        engine.close()
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    from fomo_meter.engine import SentimentEngine
    engine = SentimentEngine()
    try:
        text = args.text or sys.stdin.read()
        result = engine.analyze_text(text, use_model=args.model, record=False)
        if args.json:
            print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        else:
            print(f"relevant={result.is_relevant} stance={result.stance.value} "
                  f"intensity={result.intensity} confidence={result.confidence:.2f} engine={result.engine}")
            if result.model:
                print(f"model={result.model}")
            print(result.reason)
            if result.reasoning:
                print(result.reasoning)
            if result.model_error:
                print(f"model_error={result.model_error}")
    finally:
        engine.close()
    return 0


def command_brief(args: argparse.Namespace) -> int:
    from fomo_meter.engine import SentimentEngine
    engine = SentimentEngine()
    try:
        brief = engine.slm_brief(args.ticker, record=False)
        if args.json:
            print(json.dumps(brief.model_dump(mode="json"), ensure_ascii=False, indent=2))
        else:
            print(f"{brief.ticker} · {brief.stance.value} · intensity {brief.intensity} "
                  f"confidence {brief.confidence:.2f}")
            print(brief.reasoning)
            if brief.evidence_quote:
                print(f"Evidence: {brief.evidence_quote}")
            if brief.evidence_url:
                print(f"URL: {brief.evidence_url}")
    finally:
        engine.close()
    return 0


def command_status(_args: argparse.Namespace) -> int:
    from fomo_meter.engine import SentimentEngine
    engine = SentimentEngine()
    try:
        print(json.dumps(engine.status(), ensure_ascii=False, indent=2))
    finally:
        engine.close()
    return 0


def command_learn(args: argparse.Namespace) -> int:
    from fomo_meter.engine import SentimentEngine
    engine = SentimentEngine()
    try:
        text = args.text or sys.stdin.read()
        res = engine.record_feedback(text=text, label=args.label, source="cli")
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            if res.get("ok"):
                print(f"[OK] {res.get('message')} (Pending feedbacks: {res.get('pending_count')})")
                if args.retrain:
                    print("Memulai proses auto-retrain...")
                    retrain_res = engine.auto_retrain()
                    if retrain_res.get("ok"):
                        print(f"[OK] Retrain sukses! Val Accuracy: {retrain_res.get('accuracy'):.1%} | Golden: {retrain_res.get('golden_accuracy'):.1%}")
                    else:
                        print(f"[ERROR] Retrain gagal: {retrain_res.get('error')}")
            else:
                print(f"[ERROR] Gagal: {res.get('error')}")
    finally:
        engine.close()
    return 0


def command_retrain(args: argparse.Namespace) -> int:
    from fomo_meter.engine import SentimentEngine
    engine = SentimentEngine()
    try:
        print("Memulai auto-retrain Sentiment SLM...")
        res = engine.auto_retrain()
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            if res.get("ok"):
                print(f"[OK] Berhasil: {res.get('message')}")
                print(f"  Sampel: {res.get('samples')} (termasuk {res.get('feedbacks')} feedback)")
                print(f"  Val Accuracy: {res.get('accuracy'):.1%} | Golden Benchmark: {res.get('golden_accuracy'):.1%}")
                print(f"  Waktu latih: {res.get('elapsed_s')} detik")
            else:
                print(f"[ERROR] Gagal: {res.get('error')}")
    finally:
        engine.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fomo-meter", description="FOMO Meter desktop radar")
    parser.add_argument("--version", action="version", version="FOMO Meter 2.0.0")
    sub = parser.add_subparsers(dest="command")

    gui = sub.add_parser("gui", help="Buka aplikasi desktop")
    gui.set_defaults(func=command_gui)

    serve = sub.add_parser("serve", help="Jalankan dashboard HTTP lokal")
    serve.add_argument("--port", type=int, default=0, help="Port (0 = ephemeral)")
    serve.set_defaults(func=command_serve)

    scan = sub.add_parser("scan", help="Scan satu ticker")
    scan.add_argument("--ticker", default="BTCUSD", choices=sorted(TICKERS))
    scan.add_argument("--force", action="store_true", help="Lewati cache")
    scan.add_argument("--json", action="store_true")
    scan.set_defaults(func=command_scan)

    analyze = sub.add_parser("analyze", help="Analisis satu teks")
    analyze.add_argument("--text", default="", help="Teks; kosong berarti baca stdin")
    analyze.add_argument("--model", action="store_true", help="Gunakan Sentiment SLM")
    analyze.add_argument("--json", action="store_true")
    analyze.set_defaults(func=command_analyze)

    brief = sub.add_parser("brief", help="Buat Brief Sentiment SLM dari scan live")
    brief.add_argument("--ticker", default="BTCUSD", choices=sorted(TICKERS))
    brief.add_argument("--json", action="store_true")
    brief.set_defaults(func=command_brief)

    learn = sub.add_parser("learn", help="Simpan koreksi feedback auto-learn")
    learn.add_argument("--text", required=True, help="Teks sampel finansial")
    learn.add_argument("--label", required=True, choices=["BUY", "SELL", "WAIT"], help="Label yang benar")
    learn.add_argument("--retrain", action="store_true", help="Langsung latih ulang setelah menyimpan")
    learn.add_argument("--json", action="store_true")
    learn.set_defaults(func=command_learn)

    retrain = sub.add_parser("retrain", help="Latih ulang Sentiment SLM dari dataset + feedback")
    retrain.add_argument("--json", action="store_true")
    retrain.set_defaults(func=command_retrain)

    status = sub.add_parser("status", help="Tampilkan status engine dan model")
    status.set_defaults(func=command_status)

    return parser


def main() -> int:
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        args = parser.parse_args(["gui"])
    if getattr(args, "command", None) == "gui" and sys.platform == "win32":
        try:
            import ctypes
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 0)
        except Exception:
            pass
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
