#!/bin/bash
# Final run on the VM: Cosmos Reason curation -> prepare -> two parallel GR00T trainings.
#   main     = sim (flyer) + real, each curated by Cosmos 3 Reasoner  -> $WORK/finetune/main
#   ablation = same sources without the Cosmos curation               -> $WORK/finetune/ablation
# The two runs go one after the other (one run at batch 16 needs ~59 GB).
# Expects the Cosmos Reason server on :8000 for the scoring step (stopped afterwards to free the GPU).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh >/dev/null
SOURCES=(flyer_c flyer_d station_real)
STEPS="${MAX_STEPS:-2000}"
L=$DATASETS_DIR/local
mkdir -p cosmos logs

echo "== 1. Cosmos 3 Reasoner scoring"
~/work/cosmos_reason/.venv/bin/python tools/cosmos_reason_score.py --out cosmos/reason_scores.json --workers 6 \
    $(for s in "${SOURCES[@]}"; do echo "$L/$s"; done)
tmux kill-session -t reason_server 2>/dev/null || true
sleep 10

echo "== 2. curation + GR00T prep"
main_paths=() abl_paths=()
for s in "${SOURCES[@]}"; do
    "$GR00T_REPO/.venv/bin/python" tools/curate.py --scores cosmos/reason_scores.json --src "$L/$s" --dst "$L/${s}_curated" | tee -a cosmos/curation.log
    ./prepare_dataset.sh "$L/$s" > "logs/prep_$s.log" 2>&1
    ./prepare_dataset.sh "$L/${s}_curated" > "logs/prep_${s}_curated.log" 2>&1
    main_paths+=("$L/${s}_curated") abl_paths+=("$L/$s")
done
join() { local IFS=:; echo "$*"; }

echo "== 3. trainings: main, then ablation (one GPU fits one run at batch 16)"
main_ds=$(join "${main_paths[@]}")
abl_ds=$(join "${abl_paths[@]}")
rm -rf "$WORK/finetune/main" "$WORK/finetune/ablation"
tmux new -d -s train_final "cd $PWD && source env.sh && \
  FINETUNE_OUTPUT=$WORK/finetune/main TRAIN_DATASET='$main_ds' MAX_STEPS=${MAIN_STEPS:-1500} SAVE_STEPS=500 ./finetune.sh > logs/train_main.log 2>&1; echo FERTIG >> logs/train_main.log; \
  FINETUNE_OUTPUT=$WORK/finetune/ablation TRAIN_DATASET='$abl_ds' MAX_STEPS=${ABL_STEPS:-1000} SAVE_STEPS=500 ./finetune.sh > logs/train_ablation.log 2>&1; echo FERTIG >> logs/train_ablation.log; sleep 86400"
echo "started: main on $main_ds"
echo "         ablation on $abl_ds"
