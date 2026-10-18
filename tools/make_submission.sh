#!/bin/bash
# Assemble the submission folder on the VM and (optionally) upload it to Nebius Object Storage.
#   MAIN_CKPT=... ABL_CKPT=... tools/make_submission.sh                   # build ~/work/submission/sim-to-table
#   BUCKET=<name> REGION=eu-north1 UPLOAD=1 tools/make_submission.sh       # build + aws s3 sync
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh >/dev/null
OUT="${OUT:-$WORK/submission/sim-to-table}"
L=$DATASETS_DIR/local
last_ckpt() { ls -d "$1"/checkpoint-* | sort -t- -k2 -n | tail -1; }
MAIN_CKPT="${MAIN_CKPT:-$(last_ckpt "$WORK/finetune/main")}"
ABL_CKPT="${ABL_CKPT:-$(last_ckpt "$WORK/finetune/ablation")}"

rm -rf "$OUT" && mkdir -p "$OUT/checkpoints" "$OUT/cosmos" "$OUT/code"
cp -al "$MAIN_CKPT" "$OUT/checkpoints/main" 2>/dev/null || cp -r "$MAIN_CKPT" "$OUT/checkpoints/main"
cp -al "$ABL_CKPT" "$OUT/checkpoints/ablation" 2>/dev/null || cp -r "$ABL_CKPT" "$OUT/checkpoints/ablation"
cp cosmos/reason_scores.json cosmos/curation.log "$OUT/cosmos/"
rsync -a --exclude logs --exclude cosmos --exclude '.venv*' --exclude __pycache__ ./ "$OUT/code/"

python3 - "$OUT" "$MAIN_CKPT" "$ABL_CKPT" <<'EOF'
import json, os, sys
out, main_ckpt, abl_ckpt = sys.argv[1:4]
L = os.path.expanduser("~/work/datasets/local")
def eps(name):
    return json.load(open(f"{L}/{name}/meta/info.json"))["total_episodes"]
sim = eps("flyer_c_curated") + eps("flyer_d_curated")
real = eps("station_real_curated")
manifest = {
    "policy": "GR00T N1.7-3B post-trained (NEW_EMBODIMENT), cameras front + side, LeRobot v2.1 data",
    "main_checkpoint": {
        "path": "checkpoints/main", "source_checkpoint": main_ckpt,
        "episodes": {"sim_isaac_raw": sim, "cosmos_transfer_augmented": 0, "real_teleop": real},
        "total_episodes": sim + real,
        "curation": "Cosmos 3 Reasoner (nvidia/Cosmos3-Nano) scored every episode video; rule: drop if grasp_success or placed_on_target is false or motion_clean < 3",
        "sampling_note": "real dataset listed 4x in the GR00T mixture to up-weight real frames",
    },
    "ablation_checkpoint": {
        "path": "checkpoints/ablation", "source_checkpoint": abl_ckpt,
        "episodes": {"sim_isaac_raw": eps("flyer_c") + eps("flyer_d"), "cosmos_transfer_augmented": 0, "real_teleop": eps("station_real")},
        "curation": "none (no Cosmos)",
    },
    "caps": {"real_teleop_max": 50, "total_max": 300},
}
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=2)
print(json.dumps(manifest["main_checkpoint"]["episodes"]), "main total", sim + real)
EOF
cp "$OUT/code/docs/SUBMISSION.md" "$OUT/README.md"
du -sh "$OUT"

if [ "${UPLOAD:-0}" = 1 ]; then
    : "${BUCKET:?set BUCKET}" "${REGION:?set REGION}"
    export PATH=$HOME/.local/bin:$PATH
    aws s3 sync "$OUT" "s3://$BUCKET/${PREFIX:-sim-to-table}/" --endpoint-url "https://storage.$REGION.nebius.cloud"
    aws s3 ls "s3://$BUCKET/${PREFIX:-sim-to-table}/" --recursive --endpoint-url "https://storage.$REGION.nebius.cloud" | tail -5
fi
