#!/bin/bash
# Record real teleop episodes at the station into datasets/real/station_real (LeRobot v3, local only).
#   tools/record_real.sh 3            # 3 episodes (first test)
#   RESUME=true tools/record_real.sh 10   # append 10 more to the same dataset
# Keys during recording: Right arrow = episode done early, Left arrow = redo episode, Esc = stop.
# Joint units use_degrees=false on both arms: same [-100, 100] convention as the sim dataset.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
N="${1:-3}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/tty.usbmodem5A7A0557971}"
LEADER_PORT="${LEADER_PORT:-/dev/tty.usbmodem5C821106681}"
FRONT_CAM="${FRONT_CAM:-1}"
SIDE_CAM="${SIDE_CAM:-0}"
TASK="${TASK:-pick up the blue block and place it on the flyer}"

exec .venv-lerobot/bin/lerobot-record \
    --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id=station_follower --robot.use_degrees=false \
    --robot.cameras="{front: {type: opencv, index_or_path: $FRONT_CAM, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: $SIDE_CAM, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id=station_leader --teleop.use_degrees=false \
    --dataset.repo_id=local/station_real --dataset.root="$PWD/datasets/real/station_real" \
    --dataset.single_task="$TASK" --dataset.num_episodes="$N" --dataset.fps=30 \
    --dataset.episode_time_s=25 --dataset.reset_time_s=10 \
    --dataset.push_to_hub=false --resume="${RESUME:-false}"
