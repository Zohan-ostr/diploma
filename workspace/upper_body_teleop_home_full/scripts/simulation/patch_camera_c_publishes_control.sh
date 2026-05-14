#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo " PATCH CAMERA: C/R KEYS PUBLISH /teleop/control"
echo "============================================================"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "============================================================"

echo
echo "Searching python files with cv2.waitKey..."
mapfile -t CANDIDATES < <(
  grep -RIl "waitKey" src scripts 2>/dev/null | grep -E "\.py$" || true
)

if [ "${#CANDIDATES[@]}" -eq 0 ]; then
  echo "ERROR: no Python files with waitKey found."
  echo "Run:"
  echo "  grep -RIn \"waitKey\\|calibrate\\|ord('c')\\|ord(\\\"c\\\")\" src scripts"
  exit 1
fi

echo "Candidates:"
printf '  %s\n' "${CANDIDATES[@]}"

TARGET=""

for f in "${CANDIDATES[@]}"; do
  if grep -q "PoseLandmarks3D\|webcam\|mediapipe\|VideoCapture\|imshow" "$f"; then
    TARGET="$f"
    break
  fi
done

if [ -z "$TARGET" ]; then
  TARGET="${CANDIDATES[0]}"
fi

echo
echo "Selected target:"
echo "  $TARGET"

BACKUP="${TARGET}.bak.control.$(date +%Y%m%d_%H%M%S)"
cp "$TARGET" "$BACKUP"
echo "Backup:"
echo "  $BACKUP"

python3 - <<PY
from pathlib import Path
import re

p = Path("$TARGET")
text = p.read_text(encoding="utf-8")

# ---------- imports ----------
if "from std_msgs.msg import String" not in text:
    # Добавляем рядом с импортами ROS/messages.
    if "from upper_body_msgs.msg import" in text:
        text = re.sub(
            r"(from upper_body_msgs\.msg import[^\n]+\n)",
            r"\1from std_msgs.msg import String\n",
            text,
            count=1,
        )
    elif "import rclpy" in text:
        text = text.replace("import rclpy", "import rclpy\nfrom std_msgs.msg import String", 1)
    else:
        text = "from std_msgs.msg import String\n" + text

# ---------- publisher in Node class ----------
# Ищем место, где создаются publisher/subscription/timer.
if "teleop_control_pub" not in text:
    patterns = [
        r"(self\.[A-Za-z0-9_]+_pub\s*=\s*self\.create_publisher\([^\n]+\)\n)",
        r"(self\.[A-Za-z0-9_]+publisher\s*=\s*self\.create_publisher\([^\n]+\)\n)",
        r"(self\.create_timer\([^\n]+\)\n)",
    ]

    inserted = False
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            insert = (
                m.group(1)
                + "        self.teleop_control_pub = self.create_publisher(String, '/teleop/control', 10)\n"
            )
            text = text[:m.start()] + insert + text[m.end():]
            inserted = True
            break

    if not inserted:
        # Пытаемся вставить в __init__ после super().__init__
        text = re.sub(
            r"(super\(\).__init__\([^\n]+\)\n)",
            r"\1        self.teleop_control_pub = self.create_publisher(String, '/teleop/control', 10)\n",
            text,
            count=1,
        )

# ---------- helper method ----------
if "def publish_teleop_control" not in text:
    # Вставим метод перед main() или перед первым def main.
    helper = '''
    def publish_teleop_control(self, command: str):
        msg = String()
        msg.data = command
        self.teleop_control_pub.publish(msg)
        try:
            self.get_logger().info(f"Published /teleop/control: {command}")
        except Exception:
            pass

'''
    idx = text.find("\\ndef main(")
    if idx != -1:
        text = text[:idx+1] + helper + text[idx+1:]
    else:
        # Вставляем в конец. Если это вне класса, будет плохо, но обычно main есть.
        text += "\\n" + helper + "\\n"

# ---------- patch key handling ----------
# Случай 1: есть key = cv2.waitKey(...)
# Добавим обработку сразу после строки с waitKey.
lines = text.splitlines()
out = []
already_has_control_key_logic = "publish_teleop_control('calibrate')" in text or 'publish_teleop_control("calibrate")' in text

for line in lines:
    out.append(line)

    if not already_has_control_key_logic and "waitKey" in line and "=" in line and "cv2." in line:
        indent = line[:len(line) - len(line.lstrip())]
        # Пытаемся определить имя переменной слева от =
        var = line.split("=", 1)[0].strip()
        if var:
            out.append(f"{indent}if {var} in (ord('c'), ord('C')):")
            out.append(f"{indent}    self.publish_teleop_control('calibrate')")
            out.append(f"{indent}elif {var} in (ord('r'), ord('R')):")
            out.append(f"{indent}    self.publish_teleop_control('reset')")
            already_has_control_key_logic = True

text2 = "\\n".join(out) + "\\n"

if not already_has_control_key_logic:
    print("WARNING: could not automatically patch key handling.")
    print("File was still patched with publisher/helper, but C key logic may need manual insertion.")
else:
    print("Patched key handling for C/R.")

p.write_text(text2, encoding="utf-8")
print("patched:", p)
PY

echo
echo "Checking patch:"
grep -n "teleop_control_pub\\|publish_teleop_control\\|/teleop/control\\|ord('c')\\|ord('C')" "$TARGET" || true

echo
echo "Patch done."
echo "Now restart camera container:"
echo "  bash scripts/simulation/sim_3_start_camera.sh 0"
