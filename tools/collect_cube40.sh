#!/bin/bash
# Sim batch for the changed station (40 mm cube, shifted flyer): two collectors in parallel on the VM.
#   FLYER_X=0.11 FLYER_Y=-0.235 NUM_EPISODES=15 tools/collect_cube40.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${FLYER_X:?set FLYER_X}" "${FLYER_Y:?set FLYER_Y}"
for s in cube40_a cube40_b; do
    rm -rf "$HOME/work/datasets/local/$s"
    tmux kill-session -t "$s" 2>/dev/null || true
    tmux new -d -s "$s" "cd $PWD && CUBE_SIZE_MM=40 FLYER_FLIP=${FLYER_FLIP:-1} FLYER_X=$FLYER_X FLYER_Y=$FLYER_Y DATASET_NAME=$s NUM_EPISODES=${NUM_EPISODES:-15} \
      ./collect.sh > logs/collect_$s.log 2>&1; echo FERTIG >> logs/collect_$s.log"
done
echo "gestartet: cube40_a, cube40_b (FLYER_X=$FLYER_X FLYER_Y=$FLYER_Y)"
