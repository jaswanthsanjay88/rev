"""
Launches the rev Chrome Dino Autonomous Pilot in a live browser tab.

Usage:
    python examples/run_dino_visual.py

What this does:
1. Checks if the rev inference server is running on http://localhost:8000.
2. If not, boots the rev API server in the background.
3. Automatically opens a live browser tab at http://localhost:8000/dino.
4. The Chrome Dino game runs in real-time, with rev steering every jump and duck!
"""

import os
import sys
import time
import socket
import webbrowser
import subprocess
from pathlib import Path


def is_port_in_use(port: int = 8000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    port = 8000
    dino_url = f"http://localhost:{port}/dino"
    local_html = str(Path(__file__).parent / "dino.html")

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("  [DINO] REV // CHROME DINO AUTONOMOUS PILOT LAUNCHER [DINO]")
    print("=" * 70)

    if is_port_in_use(port):
        print(f"[+] rev server is already active on port {port}.")
        target_url = dino_url
    else:
        print(f"[*] Starting rev inference server on http://localhost:{port}...")
        python_exe = sys.executable
        # Launch rev.serve in background (fast mode)
        proc = subprocess.Popen(
            [python_exe, "-m", "rev.serve", "--port", str(port), "--mock"],
            cwd=str(Path(__file__).parent.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[+] rev server process started (PID: {})".format(proc.pid))
        
        # Wait up to 3 seconds for server to accept connections
        connected = False
        for _ in range(15):
            time.sleep(0.2)
            if is_port_in_use(port):
                connected = True
                break
        
        if connected:
            print(f"[+] Server ready! Serving visual tab at {dino_url}")
            target_url = dino_url
        else:
            print(f"[!] Server booting, falling back to direct browser canvas file...")
            target_url = f"file:///{os.path.abspath(local_html).replace('\\', '/')}"

    print(f"\n[>>>] Opening live browser tab: {target_url}\n")
    webbrowser.open(target_url)

    print("=" * 70)
    print("Controls inside the browser tab:")
    print("  - Watch rev AI autonomously jump over cacti and duck under pterodactyls.")
    print("  - Click 'Switch to Manual Play' to challenge rev with your own reflexes!")
    print("  - View the live Neural Decision Output, Probability Readout, and Sensor Radar.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
