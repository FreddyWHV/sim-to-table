"""Score episode videos with Cosmos 3 Reasoner (nvidia/Cosmos3-Nano served by vLLM) and save JSON.

Server (VM, own venv in ~/work/cosmos_reason):
    vllm serve nvidia/Cosmos3-Nano --tensor-parallel-size 1 --mm-encoder-tp-mode data --async-scheduling \
        --allowed-local-media-path $HOME/work/datasets --media-io-kwargs '{"video": {"num_frames": 32}}' \
        --gpu-memory-utilization 0.68 --max-model-len 16384 --port 8000

Client:
    python tools/cosmos_reason_score.py --out cosmos/reason_scores.json DATASET_ROOT [DATASET_ROOT ...]

Each LeRobot v2.1 dataset contributes its front-camera episodes. The curation decision applied
downstream (tools/curate.py) is: keep an episode only if grasp_success and placed_on_target are
true and motion_clean >= 3.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

PROMPT = """The video shows an SO-101 robot arm inside a white box. Its task: pick up the small blue cube and place it on the dark rectangular flyer lying on the floor of the box.
Evaluate this episode. Answer using the following format:
<think> your reasoning </think>
Then, immediately after </think>, output only JSON with these keys:
{"grasp_success": true|false, "placed_on_target": true|false, "motion_clean": 1-5, "failure_mode": "none|missed_grasp|dropped|collision|jitter|idle|other", "notes": "<one short sentence>"}"""

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("datasets", nargs="+")
parser.add_argument("--out", required=True)
parser.add_argument("--camera", default="front")
parser.add_argument("--host", default="http://localhost:8000/v1")
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

client = OpenAI(api_key="EMPTY", base_url=args.host)
model = client.models.list().data[0].id


def parse_answer(text: str) -> dict:
    tail = text.split("</think>")[-1]
    m = re.search(r"\{.*\}", tail, re.S)
    if not m:
        return {"parse_error": True}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"parse_error": True}


def score(item: tuple[str, int, str]) -> dict:
    dataset, ep, path = item
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": "file://" + os.path.abspath(path)}},
                    {"type": "text", "text": PROMPT},
                ],
            },
        ],
        max_tokens=1024,
        temperature=0.0,
    )
    text = resp.choices[0].message.content or ""
    out = {"dataset": os.path.basename(dataset.rstrip("/")), "episode_index": ep, "video": path}
    out.update(parse_answer(text))
    out["raw"] = text[-600:]
    out["seconds"] = round(time.time() - t0, 1)
    return out


items = []
for ds in args.datasets:
    for path in sorted(glob.glob(os.path.join(ds, "videos", "chunk-*", f"observation.images.{args.camera}", "episode_*.mp4"))):
        items.append((ds, int(re.search(r"episode_(\d+)", path).group(1)), path))
print(f"[reason] {len(items)} episodes, model {model}")

results = []
with ThreadPoolExecutor(args.workers) as pool:
    for i, r in enumerate(pool.map(score, items), 1):
        results.append(r)
        print(f"[reason] {i}/{len(items)} {r['dataset']} ep{r['episode_index']}: "
              f"grasp={r.get('grasp_success')} placed={r.get('placed_on_target')} clean={r.get('motion_clean')}", flush=True)
        if i % 10 == 0 or i == len(items):
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            json.dump({"model": model, "prompt": PROMPT, "frames_per_video": 32, "episodes": results}, open(args.out, "w"), indent=1)
print(f"[reason] saved {args.out}")
