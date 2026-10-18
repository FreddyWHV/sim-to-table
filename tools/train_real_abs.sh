#!/bin/bash
# Real-only run with absolute arm actions (gr00t/so101_abs_config.py) on the VM:
#   convert station_real_v2 (v3 -> v2.1) -> Cosmos 3 Reasoner -> curate -> train $WORK/finetune/<RUN>
#   DROP_EPS="13" RUN=main_abs2 tools/train_real_abs.sh
# Mixture: station_real (first layout) 1x, station_real_v2 (rebuilt layout, 40 mm cube) 2x. No sim data: the sim
# joint convention does not match the real arm (see docs/SUBMISSION.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MODALITY_CONFIG=$PWD/gr00t/so101_abs_config.py
source env.sh >/dev/null
NAME=station_real_v2
RUN="${RUN:-main_abs2}"
STEPS="${MAX_STEPS:-1000}"
L=$DATASETS_DIR/local
CONV=$DATASETS_DIR/conv
mkdir -p cosmos logs

echo "== 1. v3 -> v2.1 ($NAME)"
rm -rf "$CONV/local/$NAME" "$CONV/local/${NAME}_v3.0" "$L/$NAME"
cp -r "$DATASETS_DIR/real_v3/$NAME" "$CONV/local/$NAME"
(cd "$GR00T_REPO/scripts/lerobot_conversion" && .venv/bin/python convert_v3_to_v2.py --repo-id "local/$NAME" --root "$CONV") \
    > "logs/convert_$NAME.log" 2>&1
mv "$CONV/local/$NAME" "$L/$NAME"
./prepare_dataset.sh "$L/$NAME" > "logs/prep_$NAME.log" 2>&1
echo "   $(python3 -c "import json; print(json.load(open('$L/$NAME/meta/info.json'))['total_episodes'])") episodes"

echo "== 2. Cosmos 3 Reasoner"
until ! pgrep -f "experiment/launch_finetun" >/dev/null; do sleep 5; done   # GPU: one training or vLLM at a time
if ! curl -sf localhost:8000/v1/models >/dev/null; then
    tmux kill-session -t reason_server 2>/dev/null || true
    tmux new -d -s reason_server "cd ~/work/cosmos_reason && export PATH=$HOME/work/cosmos_reason/.venv/bin:\$PATH && vllm serve nvidia/Cosmos3-Nano \
      --tensor-parallel-size 1 --mm-encoder-tp-mode data --async-scheduling --allowed-local-media-path $HOME/work/datasets \
      --media-io-kwargs '{\"video\": {\"num_frames\": 32}}' --gpu-memory-utilization 0.68 --max-model-len 16384 --port 8000 \
      > server_real_abs.log 2>&1"
fi
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done
~/work/cosmos_reason/.venv/bin/python tools/cosmos_reason_score.py --out "cosmos/reason_$NAME.json" --workers 8 "$L/$NAME"
tmux kill-session -t reason_server
python3 - "$NAME" <<'EOF'
import json, sys
name = sys.argv[1]
all_ = json.load(open("cosmos/reason_scores.json"))
new = json.load(open(f"cosmos/reason_{name}.json"))["episodes"]
all_["episodes"] = [e for e in all_["episodes"] if e["dataset"] != name] + new
json.dump(all_, open("cosmos/reason_scores.json", "w"), indent=1)
print(f"   merged {len(new)} scores into cosmos/reason_scores.json ({len(all_['episodes'])} total)")
EOF
sleep 10

echo "== 3. curate + prep"
read -ra drop <<< "${DROP_EPS:-}"
"$GR00T_REPO/.venv/bin/python" tools/curate.py --scores cosmos/reason_scores.json --src "$L/$NAME" --dst "$L/${NAME}_curated" \
    --drop "${drop[@]}" | tee -a cosmos/curation.log
./prepare_dataset.sh "$L/${NAME}_curated" > "logs/prep_${NAME}_curated.log" 2>&1

echo "== 4. training $RUN"
old=$L/station_real_curated new=$L/${NAME}_curated
rm -rf "$WORK/finetune/$RUN"
FINETUNE_OUTPUT=$WORK/finetune/$RUN TRAIN_DATASET="$old:$new:$new" MAX_STEPS=$STEPS SAVE_STEPS=500 ./finetune.sh > "logs/train_$RUN.log" 2>&1
echo FERTIG >> "logs/train_$RUN.log"
echo "fertig: $WORK/finetune/$RUN"
