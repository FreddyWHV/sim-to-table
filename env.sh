#!/bin/bash
# Shared settings for the Isaac Lab SO-101 block pick-and-place -> GR00T N1.7 post-training pipeline.
#   source ~/work/so101_sim_pickplace/env.sh

# Defaults = layout created by nebius/setup_vm.sh ($HOME/work/...)
export WORK="${WORK:-$HOME/work}"
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GR00T_REPO="${GR00T_REPO:-$WORK/Isaac-GR00T}"
export ISAACLAB_DIR="${ISAACLAB_DIR:-$WORK/IsaacLab}"
export ISAACLAB_PY="${ISAACLAB_PY:-$WORK/env_isaaclab/bin/python}"

# Isaac Sim needs these (same as ~/Desktop/isaac/launch_isaacsim.sh)
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Task + dataset
export TASK="${TASK:-pick up the blue block and place it on the flyer}"
export DATASETS_DIR=$WORK/datasets
export DATASET_NAME="${DATASET_NAME:-so101_sim_pickplace}"
export DATASET_ROOT=$DATASETS_DIR/local/$DATASET_NAME
export NUM_EPISODES="${NUM_EPISODES:-50}"

# GR00T
export BASE_MODEL=$GR00T_REPO/checkpoints/GR00T-N1.7-3B
export MODALITY_CONFIG=$PROJECT_DIR/gr00t/so101_sim_config.py
export EMBODIMENT_TAG=NEW_EMBODIMENT
export FINETUNE_OUTPUT="${FINETUNE_OUTPUT:-$WORK/finetune/$DATASET_NAME}"
export CHECKPOINT="${CHECKPOINT:-}"           # set to $FINETUNE_OUTPUT/checkpoint-<N> for start_server.sh
export POLICY_HOST=localhost
export POLICY_PORT="${POLICY_PORT:-5556}"  # 5555 is taken by nv-hostengine (DCGM) on the Nebius VM

# N1.7 loads the gated nvidia/Cosmos-Reason2-2B backbone from the Hub (do not set HF_HUB_OFFLINE)
unset HF_HUB_OFFLINE
export TOKENIZERS_PARALLELISM=false

gr00t_activate() {
    cd "$GR00T_REPO"
    source .venv/bin/activate
    # DGX Spark (aarch64) only; no-op on the x86_64 Nebius VM
    if [ "$(uname -m)" = aarch64 ]; then
        source scripts/activate_spark.sh >/dev/null
        export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH"
    fi
}
