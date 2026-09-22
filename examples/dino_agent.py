"""
Autonomous Chrome Dino Agent powered by rev (System 1 Decision Engine).

Demonstrates:
1. Ultra-low latency decision making (<15ms per frame) to react at 60 FPS.
2. Parallel typed decisions:
   - action: choice (RUN, JUMP, DUCK)
   - urgency: noul ([0.0, 1.0] continuous probability)
   - threat_level: score (safe, approaching, danger, critical)
3. Both in-process execution and HTTP (POST /v1/systemone) client support.
4. Includes a complete, zero-dependency Python Dino game simulator with:
   - Ground cacti (small and large)
   - Flying pterodactyls (low-flying requires jump, mid-height requires duck, high-flying safe)
   - Progressive speed acceleration as score advances.
"""

import os
import sys
import time
import random
from typing import Dict, Any, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import rev


def get_dino_questions() -> Dict[str, Any]:
    """Defines the typed decision questions for the Dino agent."""
    return {
        "action": {
            "type": "choice",
            "instructions": (
                "Determine the immediate physical action for the T-Rex runner. "
                "Consider the obstacle type, altitude, and time to impact."
            ),
            "criteria": {
                "RUN": "Track ahead is clear, obstacle is far away (>110px), or Dino is safely in mid-air.",
                "JUMP": "Ground obstacle (single cactus, double/triple cluster, or low-flying pterodactyl) is in immediate jump range: <=85px for single cactus, <=55px for double or triple cluster to clear trailing edge.",
                "DUCK": "Mid-height flying pterodactyl approaching overhead that would collide with a standing Dino but clears when ducking.",
            },
        },
        "urgency": {
            "type": "noul",
            "instructions": "Is an evasive maneuver (jump or duck) immediately critical to prevent a collision within 180ms?",
        },
        "threat_level": {
            "type": "score",
            "instructions": "Imminent collision threat assessment",
            "criteria": [
                "safe: no obstacle on screen or far beyond impact zone",
                "approaching: obstacle visible, prepare reaction window",
                "danger: inside trigger threshold, evasive action executing",
                "critical: immediate collision point",
            ],
        },
    }


class DinoGameSimulator:
    """
    Realistic standalone Chrome Dino physics simulator:
    - 60 FPS game tick with gravity and vertical velocities.
    - Ground obstacles (cacti) and airborne pterodactyls (low, mid, high).
    - Increasing game speed as score rises.
    """

    def __init__(self):
        self.dino_x = 50
        self.dino_y = 0.0          # 0 is ground
        self.dino_vy = 0.0
        self.gravity = 1.2
        self.jump_velocity = 16.0
        self.is_jumping = False
        self.is_ducking = False
        
        self.dino_standing_height = 45
        self.dino_ducking_height = 25
        self.dino_width = 40

        self.speed = 10.0
        self.score = 0
        self.game_over = False
        self.ticks = 0

        self.obstacles = []
        self.spawn_timer = 40

    def spawn_obstacle(self):
        # 60% cacti, 40% pterodactyls if score > 200
        if self.score > 200 and random.random() < 0.4:
            altitude_type = random.choice(["low", "mid", "high"])
            if altitude_type == "low":
                obs_y = 10   # Ground sweep -> MUST JUMP
            elif altitude_type == "mid":
                obs_y = 35   # Head height -> MUST DUCK
            else:
                obs_y = 65   # High overhead -> CAN RUN SAFELY
            obs = {
                "type": "PTERODACTYL",
                "altitude": altitude_type,
                "x": 600,
                "y": obs_y,
                "width": 36,
                "height": 24,
            }
        else:
            size = random.choice(["small", "large"])
            cluster = random.choice([1, 2, 3])
            unit_w = 17 if size == "small" else 25
            h = 35 if size == "small" else 50
            obs = {
                "type": f"CACTUS_{size.upper()}" + (f"_{cluster}X" if cluster > 1 else ""),
                "altitude": "ground",
                "x": 600,
                "y": 0,
                "width": unit_w * cluster,
                "height": h,
                "cluster": cluster,
            }
        self.obstacles.append(obs)

    def step(self, action: str):
        if self.game_over:
            return

        self.ticks += 1
        self.score += 1
        if self.ticks % 250 == 0:
            self.speed += 0.5  # Gradual acceleration

        # Process action
        if action == "JUMP" and not self.is_jumping:
            self.is_jumping = True
            self.is_ducking = False
            self.dino_vy = self.jump_velocity
        elif action == "DUCK" and not self.is_jumping:
            self.is_ducking = True
        else:
            self.is_ducking = False

        # Physics update
        if self.is_jumping:
            self.dino_y += self.dino_vy
            self.dino_vy -= self.gravity
            if self.dino_y <= 0.0:
                self.dino_y = 0.0
                self.dino_vy = 0.0
                self.is_jumping = False

        # Move obstacles
        for obs in self.obstacles:
            obs["x"] -= self.speed

        self.obstacles = [o for o in self.obstacles if o["x"] + o["width"] > 0]

        # Spawn obstacles
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self.spawn_obstacle()
            # Random gap: 45 to 80 ticks
            self.spawn_timer = random.randint(45, 85)

        # Collision detection
        d_h = self.dino_ducking_height if self.is_ducking else self.dino_standing_height
        d_top = self.dino_y + d_h
        d_bottom = self.dino_y
        d_left = self.dino_x
        d_right = self.dino_x + self.dino_width

        for obs in self.obstacles:
            o_left = obs["x"]
            o_right = obs["x"] + obs["width"]
            o_bottom = obs["y"]
            o_top = obs["y"] + obs["height"]

            # Axis-Aligned Bounding Box overlap
            overlap_x = (d_right > o_left + 4) and (d_left < o_right - 4)
            overlap_y = (d_top > o_bottom + 4) and (d_bottom < o_top - 4)
            if overlap_x and overlap_y:
                self.game_over = True
                break

    def get_telemetry(self) -> Dict[str, Any]:
        """Extracts normalized telemetry for rev."""
        # Find next incoming obstacle ahead of dino
        ahead = [o for o in self.obstacles if o["x"] + o["width"] > self.dino_x]
        ahead.sort(key=lambda o: o["x"])
        next_obs = ahead[0] if ahead else None

        if next_obs:
            distance_px = round(next_obs["x"] - (self.dino_x + self.dino_width), 1)
            time_to_impact_ms = round(max(0, distance_px) / self.speed * 16.66, 1)
            obs_info = {
                "type": next_obs["type"],
                "altitude": next_obs["altitude"],
                "distance_px": distance_px,
                "time_to_impact_ms": time_to_impact_ms,
                "obstacle_height_px": next_obs["height"],
                "obstacle_width_px": next_obs["width"],
                "cluster": next_obs.get("cluster", 1),
            }
        else:
            obs_info = {
                "type": "NONE",
                "altitude": "none",
                "distance_px": 999.0,
                "time_to_impact_ms": 9999.0,
                "obstacle_height_px": 0,
                "obstacle_width_px": 0,
                "cluster": 0,
            }

        return {
            "game_speed": round(self.speed, 2),
            "score": self.score,
            "dino": {
                "altitude_px": round(self.dino_y, 1),
                "is_jumping": self.is_jumping,
                "is_ducking": self.is_ducking,
            },
            "incoming_obstacle": obs_info,
        }


def format_state_for_rev(telemetry: Dict[str, Any]) -> str:
    """Formats telemetry dictionary into clear semantic text for the decision model."""
    dino = telemetry["dino"]
    obs = telemetry["incoming_obstacle"]
    speed = telemetry["game_speed"]

    cluster_str = f", cluster={obs.get('cluster', 1)}x (width={obs.get('obstacle_width_px', 25)}px)" if "CACTUS" in obs['type'] else ""
    return (
        f"Game Speed: {speed:.1f} px/frame. "
        f"Dino status: altitude={dino['altitude_px']:.0f}px, jumping={dino['is_jumping']}, ducking={dino['is_ducking']}. "
        f"Next Obstacle: type={obs['type']}{cluster_str}, altitude={obs['altitude']}, "
        f"distance={obs['distance_px']:.0f}px, time_to_impact={obs['time_to_impact_ms']:.0f}ms."
    )


def run_dino_agent(max_frames: int = 400, delay: float = 0.03):
    """Runs the Dino simulation steered in real time by rev."""
    game = DinoGameSimulator()
    questions = get_dino_questions()

    # Ensure utf-8 stdout on Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("[DINO] REV CHROME DINO AUTONOMOUS AGENT [DINO]")
    print("   Evaluating real-time decisions in single prefill passes (<15ms)")
    print("=" * 70 + "\n")

    frame_count = 0
    while not game.game_over and frame_count < max_frames:
        frame_count += 1
        telemetry = game.get_telemetry()
        obs = telemetry["incoming_obstacle"]

        # Only query rev when an obstacle is within 250px, otherwise default to RUN
        # This mirrors System 1 filtering and saves inference cycles
        if obs["distance_px"] <= 250 and obs["type"] != "NONE":
            state_text = format_state_for_rev(telemetry)
            
            t0 = time.perf_counter()
            # Fast in-process prediction with rev
            decision_result = rev.predict(
                state=state_text,
                questions=questions,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            ans = decision_result.get("answers", {})
            action = ans.get("action", {}).get("choice", "RUN")
            conf = ans.get("action", {}).get("confidence", 0.0)
            urgency = ans.get("urgency", {}).get("noul", 0.0)
            threat = ans.get("threat_level", {}).get("score", 0.0)
        else:
            action = "RUN"
            conf = 1.0
            urgency = 0.0
            threat = 0.0
            elapsed_ms = 0.1

        # Advance game frame
        game.step(action)

        # Render status line every frame
        dino_icon = "^JUMP^" if game.is_jumping else ("_DUCK_" if game.is_ducking else "=RUN==")
        obs_icon = "[CACTUS]" if "CACTUS" in obs["type"] else ("[BIRD]  " if obs["type"] == "PTERODACTYL" else "        ")
        print(
            f"Frame {frame_count:04d} | Score: {game.score:04d} | Speed: {game.speed:4.1f} | "
            f"Obs: {obs_icon} {obs['type']:<12} ({obs['distance_px']:5.1f}px / {obs['time_to_impact_ms']:5.0f}ms) | "
            f"rev: {dino_icon} {action:<5} (conf: {conf:.2f}, urg: {urgency:.2f}, lat: {elapsed_ms:4.1f}ms)"
        )

        if delay > 0:
            time.sleep(delay)

    print("\n" + "=" * 70)
    if game.game_over:
        print(f"[CRASH] GAME OVER at Frame {frame_count}! Final Score: {game.score}")
    else:
        print(f"[VICTORY] SURVIVED all {max_frames} frames! Final Score: {game.score}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_dino_agent(max_frames=150, delay=0.02)
