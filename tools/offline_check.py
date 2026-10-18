"""Open-loop check: feed recorded real frames to the policy server and compare with the recorded actions.

    .venv-lerobot/bin/python tools/offline_check.py --episode 5      # SSH tunnel to the VM on :5556 required

Separates "the model is wrong" from "the robot client is wrong": if predictions match the recorded actions here but
the arm misbehaves live, the problem is in the client / live observations.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from so101_pickplace.gr00t_client import Gr00tClient, add_batch_time_dims  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="datasets/real/station_real_v2")
parser.add_argument("--repo_id", default="local/station_real_v2")
parser.add_argument("--episode", type=int, default=5)
parser.add_argument("--every", type=int, default=45)
parser.add_argument("--horizon", type=int, default=16)
args = parser.parse_args()

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

ds = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode])
client = Gr00tClient("localhost", 5556)
n = len(ds)
actions = np.stack([ds.hf_dataset[i]["action"].numpy() for i in range(n)])
names = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
print(f"episode {args.episode}: {n} frames")
for t in range(0, n - args.horizon, args.every):
    item = ds[t]
    img = {k: (item[f"observation.images.{k}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for k in ("front", "side")}
    state = item["observation.state"].numpy().astype(np.float32)
    obs = add_batch_time_dims({
        "video": img,
        "state": {"single_arm": state[:5], "gripper": state[5:6]},
        "language": {"annotation.human.task_description": "pick up the blue block and place it on the flyer"},
    })
    act, _ = client.get_action(obs)
    pred = np.concatenate([act["single_arm"][0], act["gripper"][0]], axis=-1)[: args.horizon]
    rec = actions[t : t + args.horizon]
    err = np.abs(pred - rec).mean(0)
    print(f"t={t:4d} state={np.round(state).astype(int)} pred0={np.round(pred[0]).astype(int)} rec0={np.round(rec[0]).astype(int)} "
          f"mae={' '.join(f'{k}:{e:.0f}' for k, e in zip(names, err))}")
client.close()
