#!/bin/bash
# Record real teleop episodes at the station into datasets/real/station_real (LeRobot v3, local only).
#   tools/record_real.sh 10        # record up to 10 episodes, appends if the dataset exists
#   ROOT=datasets/real/station_real_v2 tools/record_real.sh 10   # separate dataset (repo_id local/<dir name>)
# The 50-episode cap counts every dataset under datasets/real/ together.
# One recording per episode, so nothing starts before you confirm with Enter: place the cube,
# hands out of the box, Enter -> recording. During an episode (line input, any terminal):
#   Enter = episode done (save), r + Enter = redo this episode, q + Enter = stop everything.
# Videos are h264 like the sim dataset (GR00T and Cosmos decode it; the LeRobot default is AV1).
# Joint units use_degrees=false on both arms: same [-100, 100] convention as the sim dataset.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
N="${1:-10}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/tty.usbmodem5A7A0557971}"
LEADER_PORT="${LEADER_PORT:-/dev/tty.usbmodem5C821106681}"
FRONT_CAM="${FRONT_CAM:-1}"
SIDE_CAM="${SIDE_CAM:-0}"
TASK="${TASK:-pick up the blue block and place it on the flyer}"
ROOT="${ROOT:-datasets/real/station_real}"
case "$ROOT" in /*) ;; *) ROOT="$PWD/$ROOT" ;; esac
NAME="$(basename "$ROOT")"
MAX_REAL=50

count_episodes() {
    [ -f "$ROOT/meta/info.json" ] && python3 -c "import json; print(json.load(open('$ROOT/meta/info.json'))['total_episodes'])" || echo 0
}
count_all_real() {
    python3 -c "import glob, json; print(sum(json.load(open(f))['total_episodes'] for f in glob.glob('datasets/real/station_real*/meta/info.json')))"
}

for i in $(seq 1 "$N"); do
    have=$(count_episodes)
    all=$(count_all_real)
    if [ "$all" -ge "$MAX_REAL" ]; then
        echo "Schon $all echte Episoden insgesamt -- das Limit der Aufgabe ist $MAX_REAL. Stopp."
        break
    fi
    echo
    echo "=== Episode $((have + 1))  (Lauf $i von $N, in $NAME: $have, echt gesamt: $all von $MAX_REAL) ==="
    read -r -p "Würfel hinlegen, Hände raus, Arm in Ruhestellung -> Enter startet die Aufnahme (q + Enter beendet): " answer
    [ "$answer" = "q" ] && break
    resume=false
    [ -f "$ROOT/meta/info.json" ] && resume=true
    .venv-lerobot/bin/python tools/record_one.py \
        --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id=station_follower --robot.use_degrees=false \
        --robot.cameras="{front: {type: opencv, index_or_path: $FRONT_CAM, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: $SIDE_CAM, width: 640, height: 480, fps: 30}}" \
        --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id=station_leader --teleop.use_degrees=false \
        --dataset.repo_id="local/$NAME" --dataset.root="$ROOT" \
        --dataset.single_task="$TASK" --dataset.num_episodes=1 --dataset.fps=30 \
        --dataset.episode_time_s=30 --dataset.reset_time_s=0 \
        --dataset.rgb_encoder.vcodec=h264 \
        --dataset.push_to_hub=false --resume="$resume" 2>&1 \
        | grep --line-buffered -E ">>>|Recording episode|Re-record|Exiting loop|Stopping|ERROR" || true
    echo "Gespeichert: $(count_episodes) Episoden."
done
echo "Fertig. Datensatz: $ROOT ($(count_episodes) Episoden)"
