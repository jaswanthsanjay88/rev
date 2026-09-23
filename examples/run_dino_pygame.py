"""
Launches the rev Chrome Dino Autonomous Pilot in a live Pygame desktop window.
Using dhhruv/Chrome-Dino-Runner with authentic ducking and bird mechanics.

Usage:
    py examples/run_dino_pygame.py
    py examples/run_dino_pygame.py --manual
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    game_script = root / "Chrome-Dino-Runner" / "chromedino.py"

    if not game_script.exists():
        print(f"[!] Error: {game_script} not found!")
        sys.exit(1)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("  [DINO] REV // CHROME DINO AUTONOMOUS PILOT (PYGAME) [DINO]")
    print("=" * 70)
    print("Launching Pygame desktop window powered by rev System 1 Decision Model...")
    print("Features active:")
    print("  - Single, double, and triple cacti cluster physics")
    print("  - Real Dino ducking animations (DinoDuck1, DinoDuck2)")
    print("  - 3-tier flying birds: High (run), Mid (must duck!), Low (jump)")
    print("  - Sub-5ms decision cycle with live monochrome HUD telemetry")
    print("  - Controls: Press [TAB] or [M] to toggle Autopilot vs Manual Play")
    print("=" * 70 + "\n")

    cmd = [sys.executable, str(game_script)] + sys.argv[1:]
    subprocess.run(cmd, cwd=str(game_script.parent))


if __name__ == "__main__":
    main()
