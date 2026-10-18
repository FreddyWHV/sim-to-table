"""Run a GR00T checkpoint on the real SO-101 at the station (laptop side).

The policy server runs on the Nebius VM (``start_server.sh``, bound to localhost); reach it through
an SSH tunnel first:

    ssh -N -L 5555:localhost:5555 ubuntu@<vm-ip>

then, on the laptop with both cameras and the follower arm plugged in:

    .venv-lerobot/bin/python scripts/eval_real.py --port /dev/cu.usbmodemXXXX --front 0 --side 1

Adapted from Isaac-GR00T ``gr00t/eval/real_robot/SO100/eval_so100.py`` with our camera keys
(front + side instead of front + wrist) and the repo's dependency-light ``Gr00tClient``.
Joint units are LeRobot ``use_degrees=False`` ([-100, 100], gripper [0, 100]), the same convention
the sim dataset uses (see ``so101_pickplace/so101_units.py``). Ctrl+C stops the arm.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from so101_pickplace.gr00t_client import Gr00tClient, add_batch_time_dims  # noqa: E402
from so101_pickplace.so101_units import FEATURE_NAMES  # noqa: E402

CAMERA_KEYS = ("front", "side")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--port", required=True, help="Follower arm serial port, e.g. /dev/cu.usbmodem5A7A0557971")
parser.add_argument("--robot_id", default="station_follower", help="LeRobot calibration id of the follower")
parser.add_argument("--front", required=True, help="OpenCV index or path of the front camera")
parser.add_argument("--side", required=True, help="OpenCV index or path of the side camera")
parser.add_argument("--policy_host", default="localhost")
parser.add_argument("--policy_port", type=int, default=5555)
parser.add_argument("--action_horizon", type=int, default=8, help="Predicted steps executed per policy query.")
parser.add_argument("--task_text", default="pick up the blue block and place it on the green pad")
parser.add_argument("--timeout_s", type=float, default=120.0, help="Stop after this long (judges cap a trial at 2 min).")
parser.add_argument("--max_relative_target", type=float, default=10.0, help="Per-step joint change cap (safety).")
args = parser.parse_args()


def _cam_source(value: str):
    return int(value) if value.isdigit() else value


def main():
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    client = Gr00tClient(args.policy_host, args.policy_port)
    if not client.ping():
        raise SystemExit(f"policy server not reachable at {args.policy_host}:{args.policy_port} (SSH tunnel up?)")
    keys = client.get_modality_config()["video"]["modality_keys"]
    if list(keys) != list(CAMERA_KEYS):
        raise SystemExit(f"checkpoint expects cameras {keys}, this client sends {CAMERA_KEYS}")

    cams = {
        k: OpenCVCameraConfig(index_or_path=_cam_source(src), fps=30, width=640, height=480)
        for k, src in (("front", args.front), ("side", args.side))
    }
    robot = make_robot_from_config(
        SO101FollowerConfig(
            port=args.port, id=args.robot_id, cameras=cams, use_degrees=False, max_relative_target=args.max_relative_target
        )
    )
    robot.connect()
    client.reset()
    print(f"[eval_real] running '{args.task_text}' for up to {args.timeout_s:.0f} s -- Ctrl+C to stop")

    t_end = time.time() + args.timeout_s
    try:
        while time.time() < t_end:
            obs = robot.get_observation()
            state = np.array([obs[k] for k in FEATURE_NAMES], dtype=np.float32)
            model_obs = add_batch_time_dims(
                {
                    "video": {k: np.ascontiguousarray(obs[k]) for k in CAMERA_KEYS},
                    "state": {"single_arm": state[:5], "gripper": state[5:6]},
                    "language": {"annotation.human.task_description": args.task_text},
                }
            )
            action, _ = client.get_action(model_obs)
            chunk = np.concatenate([action["single_arm"][0], action["gripper"][0]], axis=-1)  # (H, 6)
            for step in chunk[: args.action_horizon]:
                tic = time.time()
                robot.send_action({name: float(v) for name, v in zip(FEATURE_NAMES, step)})
                time.sleep(max(0.0, 1.0 / 30 - (time.time() - tic)))
    except KeyboardInterrupt:
        print("[eval_real] stopped")
    finally:
        robot.disconnect()
        client.close()


if __name__ == "__main__":
    main()
