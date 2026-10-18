#!/bin/bash
# Second real batch (grid-dot table) on the VM: convert -> GR00T prep -> Cosmos Reason -> curate -> train main_v2.
#   tools/train_v2.sh [NAME]     # NAME defaults to station_real_v2, expects the v3 dataset in $DATASETS_DIR/real_v3/NAME
# Needs the GPU free of other trainings: vLLM (Cosmos Reason) takes ~65 GB, a GR00T run ~59 GB.
# New sim datasets for the changed setup (40 mm cube, shifted flyer) come from SIM_NEW (default cube40_a cube40_b);
# they are Cosmos-scored and curated like the rest.
# DROP_EPS: episode indices of NAME to drop by hand (station_real_v2: 13 = leader cable came loose, ran into the 30 s cap).
# Mixture: old sim (flyer_c/d) once, new sim 2x, old real 4x, new real 6x -> $WORK/finetune/main_v2.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh >/dev/null
NAME="${1:-station_real_v2}"
STEPS="${MAX_STEPS:-1500}"
read -ra SIM_NEW <<< "${SIM_NEW:-cube40_a cube40_b}"
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
# reuse a server that is already up (pre-warmed while recording), otherwise start one
if ! curl -sf localhost:8000/v1/models >/dev/null; then
    tmux kill-session -t reason_server 2>/dev/null || true
    tmux new -d -s reason_server "cd ~/work/cosmos_reason && export PATH=$HOME/work/cosmos_reason/.venv/bin:\$PATH && vllm serve nvidia/Cosmos3-Nano --tensor-parallel-size 1 \
      --mm-encoder-tp-mode data --async-scheduling --allowed-local-media-path $HOME/work/datasets \
      --media-io-kwargs '{\"video\": {\"num_frames\": 32}}' --gpu-memory-utilization 0.68 --max-model-len 16384 --port 8000 \
      > server_$NAME.log 2>&1"
fi
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done
for s in "${SIM_NEW[@]}"; do   # sim collectors (tools/collect_cube40.sh) must be done
    until grep -q FERTIG "logs/collect_$s.log" 2>/dev/null; do sleep 5; done
done
~/work/cosmos_reason/.venv/bin/python tools/cosmos_reason_score.py --out "cosmos/reason_$NAME.json" --workers 6 \
    "$L/$NAME" $(for s in "${SIM_NEW[@]}"; do echo "$L/$s"; done)
tmux kill-session -t reason_server
python3 - "$NAME" <<'EOF'
import json, sys
name = sys.argv[1]
all_ = json.load(open("cosmos/reason_scores.json"))
new = json.load(open(f"cosmos/reason_{name}.json"))["episodes"]
names = {e["dataset"] for e in new}
all_["episodes"] = [e for e in all_["episodes"] if e["dataset"] not in names] + new
json.dump(all_, open("cosmos/reason_scores.json", "w"), indent=1)
print(f"   merged {len(new)} scores into cosmos/reason_scores.json ({len(all_['episodes'])} total)")
EOF
sleep 10

echo "== 3. curate + prep"
for s in "$NAME" "${SIM_NEW[@]}"; do
    drop=(); [ "$s" = "$NAME" ] && read -ra drop <<< "${DROP_EPS:-}"
    "$GR00T_REPO/.venv/bin/python" tools/curate.py --scores cosmos/reason_scores.json --src "$L/$s" --dst "$L/${s}_curated" --drop "${drop[@]}" \
        | tee -a cosmos/curation.log
    ./prepare_dataset.sh "$L/${s}_curated" > "logs/prep_${s}_curated.log" 2>&1
done

echo "== 4. training main_v2"
old=$L/station_real_curated new=$L/${NAME}_curated
ds="$L/flyer_c_curated:$L/flyer_d_curated"
for s in "${SIM_NEW[@]}"; do ds="$ds:$L/${s}_curated:$L/${s}_curated"; done
ds="$ds:$old:$old:$old:$old:$new:$new:$new:$new:$new:$new"
rm -rf "$WORK/finetune/main_v2"
FINETUNE_OUTPUT=$WORK/finetune/main_v2 TRAIN_DATASET="$ds" MAX_STEPS=$STEPS SAVE_STEPS=500 ./finetune.sh > logs/train_main_v2.log 2>&1
echo FERTIG >> logs/train_main_v2.log
echo "fertig: $WORK/finetune/main_v2"
