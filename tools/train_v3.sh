#!/bin/bash
# Third run on the VM, only the rebuilt station plus the first real batch as support:
#   new sim cube40_a..f (Cosmos-curated) 1x, station_real_v2 6x, station_real 2x, no flyer_c/d -> $WORK/finetune/main_v3
#   tools/train_v3.sh            # waits for the cube40_c..f collectors, scores them, trains
# station_real_v2_curated and cube40_a/b_curated come from tools/train_v2.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh >/dev/null
STEPS="${MAX_STEPS:-1000}"
NEW=(cube40_c cube40_d cube40_e cube40_f)
L=$DATASETS_DIR/local

for s in "${NEW[@]}"; do until grep -q FERTIG "logs/collect_$s.log" 2>/dev/null; do sleep 5; done; done
until ! pgrep -f "finetune/main_v2" >/dev/null; do sleep 5; done   # one training at a time

echo "== 1. Cosmos 3 Reasoner"
if ! curl -sf localhost:8000/v1/models >/dev/null; then
    tmux kill-session -t reason_server 2>/dev/null || true
    tmux new -d -s reason_server "cd ~/work/cosmos_reason && export PATH=$HOME/work/cosmos_reason/.venv/bin:\$PATH && vllm serve nvidia/Cosmos3-Nano \
      --tensor-parallel-size 1 --mm-encoder-tp-mode data --async-scheduling --allowed-local-media-path $HOME/work/datasets \
      --media-io-kwargs '{\"video\": {\"num_frames\": 32}}' --gpu-memory-utilization 0.68 --max-model-len 16384 --port 8000 \
      > server_v3.log 2>&1"
fi
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done
~/work/cosmos_reason/.venv/bin/python tools/cosmos_reason_score.py --out cosmos/reason_cube40_cdef.json --workers 8 \
    $(for s in "${NEW[@]}"; do echo "$L/$s"; done)
tmux kill-session -t reason_server
python3 - <<'EOF'
import json
all_ = json.load(open("cosmos/reason_scores.json"))
new = json.load(open("cosmos/reason_cube40_cdef.json"))["episodes"]
names = {e["dataset"] for e in new}
all_["episodes"] = [e for e in all_["episodes"] if e["dataset"] not in names] + new
json.dump(all_, open("cosmos/reason_scores.json", "w"), indent=1)
print(f"   merged {len(new)} scores into cosmos/reason_scores.json ({len(all_['episodes'])} total)")
EOF
sleep 10

echo "== 2. curate + prep"
for s in "${NEW[@]}"; do
    "$GR00T_REPO/.venv/bin/python" tools/curate.py --scores cosmos/reason_scores.json --src "$L/$s" --dst "$L/${s}_curated" \
        | tee -a cosmos/curation.log
    ./prepare_dataset.sh "$L/${s}_curated" > "logs/prep_${s}_curated.log" 2>&1
done

echo "== 3. training main_v3"
old=$L/station_real_curated new=$L/station_real_v2_curated
ds=""
for s in cube40_a cube40_b "${NEW[@]}"; do ds="$ds$L/${s}_curated:"; done
ds="$ds$old:$old:$new:$new:$new:$new:$new:$new"
rm -rf "$WORK/finetune/main_v3"
FINETUNE_OUTPUT=$WORK/finetune/main_v3 TRAIN_DATASET="$ds" MAX_STEPS=$STEPS SAVE_STEPS=500 ./finetune.sh > logs/train_main_v3.log 2>&1
echo FERTIG >> logs/train_main_v3.log
echo "fertig: $WORK/finetune/main_v3"
