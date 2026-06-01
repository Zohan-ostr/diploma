#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SIM 1. START OFFICIAL UNITREE MUJOCO PYTHON SIMULATOR
# ============================================================
#
# Запускает официальный:
#   external/unitree_mujoco/simulate_python/unitree_mujoco.py
#
# Важно:
#   - НЕ создаёт /tmp/unitree_mujoco_h1_lab.py
#   - НЕ использует временную копию симулятора
#   - включает ENABLE_ELASTIC_BAND=True, чтобы H1 не падал до старта sender
#   - настраивает стартовую камеру прямо в official unitree_mujoco.py
# ============================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

COMPOSE_FILE="${COMPOSE_FILE:-compose/compose.home.yaml}"
SERVICE="${SERVICE:-home-dev}"
CONTAINER_NAME="${CONTAINER_NAME:-h1_mujoco_sim}"

MUJOCO_ROBOT="${MUJOCO_ROBOT:-h1}"
MUJOCO_DOMAIN_ID="${MUJOCO_DOMAIN_ID:-42}"
MUJOCO_NET_IFACE="${MUJOCO_NET_IFACE:-lo}"

# Стартовая камера: анфас, виден весь робот.
MUJOCO_CAMERA_AZIMUTH="${MUJOCO_CAMERA_AZIMUTH:-180.0}"
MUJOCO_CAMERA_DISTANCE="${MUJOCO_CAMERA_DISTANCE:-4.8}"
MUJOCO_CAMERA_ELEVATION="${MUJOCO_CAMERA_ELEVATION:--16.0}"
MUJOCO_CAMERA_LOOKAT="${MUJOCO_CAMERA_LOOKAT:-0.0,0.0,0.85}"

echo "============================================================"
echo " SIM 1: START OFFICIAL H1 MUJOCO SIMULATION"
echo "============================================================"
echo "PROJECT_DIR:              $PROJECT_DIR"
echo "COMPOSE_FILE:             $COMPOSE_FILE"
echo "SERVICE:                  $SERVICE"
echo "CONTAINER_NAME:           $CONTAINER_NAME"
echo "MUJOCO_ROBOT:             $MUJOCO_ROBOT"
echo "MUJOCO_DOMAIN_ID:         $MUJOCO_DOMAIN_ID"
echo "MUJOCO_NET_IFACE:         $MUJOCO_NET_IFACE"
echo "MUJOCO_CAMERA_AZIMUTH:    $MUJOCO_CAMERA_AZIMUTH"
echo "MUJOCO_CAMERA_DISTANCE:   $MUJOCO_CAMERA_DISTANCE"
echo "MUJOCO_CAMERA_ELEVATION:  $MUJOCO_CAMERA_ELEVATION"
echo "MUJOCO_CAMERA_LOOKAT:     $MUJOCO_CAMERA_LOOKAT"
echo "============================================================"

if [ ! -d external/unitree_mujoco/simulate_python ]; then
  echo "ERROR: external/unitree_mujoco/simulate_python not found"
  exit 1
fi

cleanup() {
  echo
  echo "==> Cleanup: stopping container $CONTAINER_NAME ..."
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

trap cleanup INT TERM EXIT

echo
echo "==> Allow Docker containers to use X11..."
xhost +local:docker >/dev/null 2>&1 || true

echo
echo "==> Build Docker image with cache..."
docker compose -f "$COMPOSE_FILE" build "$SERVICE"

echo
echo "==> Stop old container if exists..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo
echo "==> Start official Unitree MuJoCo Python simulator..."
echo "    To stop and close MuJoCo window: press Ctrl+C"
echo

docker compose -f "$COMPOSE_FILE" run --rm \
  --name "$CONTAINER_NAME" \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e QT_X11_NO_MITSHM=1 \
  -e LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}" \
  -e MUJOCO_ROBOT="$MUJOCO_ROBOT" \
  -e MUJOCO_DOMAIN_ID="$MUJOCO_DOMAIN_ID" \
  -e MUJOCO_NET_IFACE="$MUJOCO_NET_IFACE" \
  -e MUJOCO_CAMERA_AZIMUTH="$MUJOCO_CAMERA_AZIMUTH" \
  -e MUJOCO_CAMERA_DISTANCE="$MUJOCO_CAMERA_DISTANCE" \
  -e MUJOCO_CAMERA_ELEVATION="$MUJOCO_CAMERA_ELEVATION" \
  -e MUJOCO_CAMERA_LOOKAT="$MUJOCO_CAMERA_LOOKAT" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  "$SERVICE" bash -lc '
set -eo pipefail

cd /workspace/external/unitree_mujoco/simulate_python

python3 - <<PY
from pathlib import Path
import os
import re
import py_compile

config_path = Path("config.py")
s = config_path.read_text(encoding="utf-8")

robot = os.environ.get("MUJOCO_ROBOT", "h1")
domain = os.environ.get("MUJOCO_DOMAIN_ID", "42")
iface = os.environ.get("MUJOCO_NET_IFACE", "lo")

def set_var(text: str, name: str, value: str) -> str:
    pattern = rf"^{name}\\s*=\\s*.*$"
    repl = f"{name} = {value}"
    if re.search(pattern, text, flags=re.M):
        return re.sub(pattern, repl, text, flags=re.M)
    return text + "\\n" + repl + "\\n"

s = set_var(s, "ROBOT", repr(robot))
s = set_var(s, "ROBOT_SCENE", "\"../unitree_robots/\" + ROBOT + \"/scene.xml\"")
s = set_var(s, "DOMAIN_ID", str(domain))
s = set_var(s, "INTERFACE", repr(iface))

# Без геймпада.
s = set_var(s, "USE_JOYSTICK", "0")

# Важно для H1: удерживает робота от падения до запуска SDK2 sender.
s = set_var(s, "ENABLE_ELASTIC_BAND", "True")

config_path.write_text(s, encoding="utf-8")
py_compile.compile(str(config_path), doraise=True)

print("Configured Unitree MuJoCo:")
for line in config_path.read_text(encoding="utf-8").splitlines():
    if line.startswith(("ROBOT", "ROBOT_SCENE", "DOMAIN_ID", "INTERFACE", "USE_JOYSTICK", "ENABLE_ELASTIC_BAND", "SIMULATE_DT", "VIEWER_DT")):
        print("  " + line)
PY

# ------------------------------------------------------------
# Аккуратный патч стартовой камеры.
# Без временного launcher и без переписывания основного цикла.
# ------------------------------------------------------------
python3 - <<PY
from pathlib import Path

p = Path("unitree_mujoco.py")
s = p.read_text(encoding="utf-8")

marker = "# H1_LAB_CAMERA_PATCH"

if "import os" not in s.split("\\n")[:20]:
    lines = s.splitlines()
    insert_at = 0
    for i, line in enumerate(lines[:30]):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, "import os")
    s = "\\n".join(lines) + "\\n"

if marker not in s:
    lines = s.splitlines()
    out = []
    inserted = False

    in_launch = False
    launch_is_with = False
    launch_indent = ""
    paren_balance = 0

    camera_block_assignment = [
        "",
        "{indent}" + marker,
        "{indent}viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE",
        "{indent}viewer.cam.lookat[:] = [float(x) for x in os.environ.get(\"MUJOCO_CAMERA_LOOKAT\", \"0.0,0.0,0.85\").replace(\"[\", \"\").replace(\"]\", \"\").split(\",\")]",
        "{indent}viewer.cam.distance = float(os.environ.get(\"MUJOCO_CAMERA_DISTANCE\", \"4.8\"))",
        "{indent}viewer.cam.azimuth = float(os.environ.get(\"MUJOCO_CAMERA_AZIMUTH\", \"180.0\"))",
        "{indent}viewer.cam.elevation = float(os.environ.get(\"MUJOCO_CAMERA_ELEVATION\", \"-16.0\"))",
    ]

    for line in lines:
        out.append(line)

        stripped = line.lstrip()
        base_indent = line[:len(line) - len(stripped)]

        if (not inserted) and (not in_launch) and "launch_passive" in line and "viewer" in line:
            in_launch = True
            launch_is_with = stripped.startswith("with ")
            launch_indent = base_indent + ("    " if launch_is_with else "")
            paren_balance = line.count("(") - line.count(")")

            if launch_is_with and stripped.endswith(":"):
                for b in camera_block_assignment:
                    out.append(b.format(indent=launch_indent))
                inserted = True
                in_launch = False
            elif (not launch_is_with) and paren_balance <= 0:
                for b in camera_block_assignment:
                    out.append(b.format(indent=launch_indent))
                inserted = True
                in_launch = False

        elif (not inserted) and in_launch:
            paren_balance += line.count("(") - line.count(")")

            if launch_is_with:
                if stripped.endswith(":"):
                    for b in camera_block_assignment:
                        out.append(b.format(indent=launch_indent))
                    inserted = True
                    in_launch = False
            else:
                if paren_balance <= 0:
                    for b in camera_block_assignment:
                        out.append(b.format(indent=launch_indent))
                    inserted = True
                    in_launch = False

    if not inserted:
        raise SystemExit("Could not find completed mujoco.viewer.launch_passive(...) block")

    p.write_text("\\n".join(out) + "\\n", encoding="utf-8")

print("camera patch OK:", p)
PY

python3 - <<PY
import mujoco
import pygame
print("mujoco OK:", mujoco.__version__)
print("pygame OK:", pygame.version.ver)
PY

echo
echo "Running official file:"
echo "  $(pwd)/unitree_mujoco.py"
echo

python3 unitree_mujoco.py
'
