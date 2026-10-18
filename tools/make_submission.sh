#!/bin/bash
# Assemble the submission folder on the VM and (optionally) upload it to Nebius Object Storage.
#   tools/make_submission.sh          # main_v2 (or main) + comparison run -> build ~/work/submission/sim-to-table
#   BUCKET=<name> REGION=eu-north1 UPLOAD=1 tools/make_submission.sh       # build + aws s3 sync
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh >/dev/null
OUT="${OUT:-$WORK/submission/sim-to-table}"
L=$DATASETS_DIR/local
last_ckpt() { ls -d "$1"/checkpoint-* | sort -t- -k2 -n | tail -1; }
# main = run with the second real batch (main_v2) when it exists; the comparison checkpoint is the earlier main.
# Override with MAIN_RUN=... CMP_RUN=... (run names under $WORK/finetune, logs in logs/train_<run>.log).
if [ -d "$WORK/finetune/main_v2" ]; then MAIN_RUN="${MAIN_RUN:-main_v2}" CMP_RUN="${CMP_RUN:-main}"
else MAIN_RUN="${MAIN_RUN:-main}" CMP_RUN="${CMP_RUN:-ablation}"; fi
MAIN_CKPT="${MAIN_CKPT:-$(last_ckpt "$WORK/finetune/$MAIN_RUN")}"
CMP_CKPT="${CMP_CKPT:-$(last_ckpt "$WORK/finetune/$CMP_RUN")}"

rm -rf "$OUT" && mkdir -p "$OUT/checkpoints" "$OUT/cosmos" "$OUT/code"
cp -al "$MAIN_CKPT" "$OUT/checkpoints/main" 2>/dev/null || cp -r "$MAIN_CKPT" "$OUT/checkpoints/main"
cp -al "$CMP_CKPT" "$OUT/checkpoints/comparison" 2>/dev/null || cp -r "$CMP_CKPT" "$OUT/checkpoints/comparison"
cp cosmos/reason_scores.json cosmos/curation.log "$OUT/cosmos/"
rsync -a --exclude logs --exclude cosmos --exclude '.venv*' --exclude __pycache__ ./ "$OUT/code/"

# Episode counts come from the dataset list each run was trained on (train.log header; repeats = sampling weight).
python3 - "$OUT" "$MAIN_RUN" "$MAIN_CKPT" "$CMP_RUN" "$CMP_CKPT" <<'EOF'
import json, os, re, sys
out, main_run, main_ckpt, cmp_run, cmp_ckpt = sys.argv[1:6]
W = os.path.expanduser("~/work")
def eps(name):
    return json.load(open(f"{W}/datasets/local/{name}/meta/info.json"))["total_episodes"]
def run_entry(run, ckpt, path):
    head = open(f"{W}/sim_station2/logs/train_{run}.log").readline()
    names = [os.path.basename(p) for p in re.search(r"Fine-tuning on (\S+)", head).group(1).split(":")]
    uniq = list(dict.fromkeys(names))
    n_sim = sum(eps(n) for n in uniq if not n.startswith("station_real"))
    n_real = sum(eps(n) for n in uniq if n.startswith("station_real"))
    return {
        "run": run, "path": path, "source_checkpoint": ckpt,
        "episodes": {"sim_isaac_raw": n_sim, "cosmos_transfer_augmented": 0, "real_teleop": n_real},
        "total_episodes": n_sim + n_real,
        "datasets": {n: {"episodes": eps(n), "sampling_weight": names.count(n)} for n in uniq},
        "curation": "Cosmos 3 Reasoner (nvidia/Cosmos3-Nano) scored every episode video; rule: drop if grasp_success or placed_on_target is false or motion_clean < 3"
                    if all(n.endswith("_curated") for n in uniq) else "none (no Cosmos)",
    }
manifest = {
    "policy": "GR00T N1.7-3B post-trained (NEW_EMBODIMENT), cameras front + side, LeRobot v2.1 data",
    "main_checkpoint": run_entry(main_run, main_ckpt, "checkpoints/main"),
    "comparison_checkpoint": run_entry(cmp_run, cmp_ckpt, "checkpoints/comparison"),
    "caps": {"real_teleop_max": 50, "total_max": 300},
}
for k in ("main_checkpoint", "comparison_checkpoint"):
    m = manifest[k]
    assert m["episodes"]["real_teleop"] <= 50 and m["total_episodes"] <= 300, (k, m["episodes"])
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=2)
for k in ("main_checkpoint", "comparison_checkpoint"):
    print(k, manifest[k]["run"], json.dumps(manifest[k]["episodes"]), "total", manifest[k]["total_episodes"])
EOF
cp "$OUT/code/docs/SUBMISSION.md" "$OUT/README.md"
du -sh "$OUT"

if [ "${UPLOAD:-0}" = 1 ]; then
    : "${BUCKET:?set BUCKET}" "${REGION:?set REGION}"
    export PATH=$HOME/.local/bin:$PATH
    aws s3 sync "$OUT" "s3://$BUCKET/${PREFIX:-sim-to-table}/" --endpoint-url "https://storage.$REGION.nebius.cloud"
    aws s3 ls "s3://$BUCKET/${PREFIX:-sim-to-table}/" --recursive --endpoint-url "https://storage.$REGION.nebius.cloud" | tail -5
fi
