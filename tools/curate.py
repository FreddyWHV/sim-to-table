"""Copy a LeRobot v2.1 dataset keeping only the episodes Cosmos Reason accepted.

    python tools/curate.py --scores cosmos/reason_scores.json --src DATASET --dst DATASET_CURATED

Curation decision (stated for the submission): drop an episode when Cosmos 3 Reasoner says the grasp
failed, the cube did not end on the flyer, or motion_clean < MIN_CLEAN. Episodes whose answer could
not be parsed are kept. Episodes are re-indexed 0..N-1; run prepare_dataset.sh on DST afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--scores", required=True)
parser.add_argument("--src", required=True)
parser.add_argument("--dst", required=True)
parser.add_argument("--min_clean", type=int, default=3)
parser.add_argument("--drop", type=int, nargs="*", default=[], help="Episode indices dropped by hand (e.g. hardware fault during recording).")
args = parser.parse_args()

name = os.path.basename(args.src.rstrip("/"))
scores = {e["episode_index"]: e for e in json.load(open(args.scores))["episodes"] if e["dataset"] == name}


def keep(ep: int) -> bool:
    if ep in args.drop:
        return False
    s = scores.get(ep)
    if s is None or s.get("parse_error"):
        return True
    return bool(s.get("grasp_success")) and bool(s.get("placed_on_target")) and int(s.get("motion_clean", 5)) >= args.min_clean


info = json.load(open(os.path.join(args.src, "meta", "info.json")))
episodes = [json.loads(l) for l in open(os.path.join(args.src, "meta", "episodes.jsonl"))]
kept = [e for e in episodes if keep(e["episode_index"])]
print(f"[curate] {name}: keep {len(kept)}/{len(episodes)} episodes" + (f" (dropped by hand: {args.drop})" if args.drop else ""))

if os.path.exists(args.dst):
    shutil.rmtree(args.dst)
os.makedirs(os.path.join(args.dst, "meta"))
for f in os.listdir(os.path.join(args.src, "meta")):
    if f not in ("episodes.jsonl", "episodes_stats.jsonl", "info.json", "stats.json", "relative_stats.json"):
        src_f = os.path.join(args.src, "meta", f)
        if os.path.isfile(src_f):
            shutil.copy(src_f, os.path.join(args.dst, "meta", f))

chunk = info.get("chunks_size", 1000)
video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
stats_src = os.path.join(args.src, "meta", "episodes_stats.jsonl")
ep_stats = {json.loads(l)["episode_index"]: json.loads(l) for l in open(stats_src)} if os.path.exists(stats_src) else {}
new_eps, new_stats, index = [], [], 0
for new_ep, e in enumerate(kept):
    old = e["episode_index"]
    old_c, new_c = old // chunk, new_ep // chunk
    df = pd.read_parquet(os.path.join(args.src, info["data_path"].format(episode_chunk=old_c, episode_index=old)))
    df["episode_index"] = new_ep
    df["index"] = range(index, index + len(df))
    index += len(df)
    out = os.path.join(args.dst, info["data_path"].format(episode_chunk=new_c, episode_index=new_ep))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    for vk in video_keys:
        src_v = os.path.join(args.src, info["video_path"].format(episode_chunk=old_c, video_key=vk, episode_index=old))
        dst_v = os.path.join(args.dst, info["video_path"].format(episode_chunk=new_c, video_key=vk, episode_index=new_ep))
        os.makedirs(os.path.dirname(dst_v), exist_ok=True)
        shutil.copy(src_v, dst_v)
    new_eps.append({**e, "episode_index": new_ep})
    if old in ep_stats:
        new_stats.append({**ep_stats[old], "episode_index": new_ep})

with open(os.path.join(args.dst, "meta", "episodes.jsonl"), "w") as f:
    f.writelines(json.dumps(e) + "\n" for e in new_eps)
if new_stats:
    with open(os.path.join(args.dst, "meta", "episodes_stats.jsonl"), "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in new_stats)
info["total_episodes"] = len(new_eps)
info["total_frames"] = index
info["total_videos"] = len(new_eps) * len(video_keys)
info["total_chunks"] = (len(new_eps) - 1) // chunk + 1 if new_eps else 0
info["splits"] = {"train": f"0:{len(new_eps)}"}
json.dump(info, open(os.path.join(args.dst, "meta", "info.json"), "w"), indent=4)
print(f"[curate] wrote {args.dst} ({len(new_eps)} episodes, {index} frames)")
