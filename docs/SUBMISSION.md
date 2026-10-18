# Sim to Table — submission

NVIDIA × Nebius AI Builders Hackathon, Berlin, 18 Oct 2026. Code: https://github.com/FreddyWHV/sim-to-table (branch `sim-station`).

## Where things are

On GitHub (this branch): the code, `submission/manifest.json`, `submission/reason_scores.json` and
`submission/curation.log`. The checkpoints (12 GB each) are too large for GitHub; `tools/make_submission.sh` assembles
the full folder below on the Nebius VM (`~/work/submission/sim-to-table`).

## What is in this folder

| Path | Content |
|---|---|
| `checkpoints/main` | Submitted checkpoint (`main_real`): GR00T N1.7-3B post-trained on the real teleop episodes of both station layouts, curated with Cosmos 3 Reasoner |
| `checkpoints/comparison` | Earlier run (`main`): Isaac Sim + real teleop, first station layout only |
| `manifest.json` | Episode counts per source and dataset sampling weights for both checkpoints |
| `cosmos/reason_scores.json` | Cosmos 3 Reasoner (nvidia/Cosmos3-Nano) output for every episode video |
| `cosmos/curation.log` | Effect of the curation rule per dataset |
| `code/` | Snapshot of the pipeline code |

## Pipeline

1. **Simulate (Isaac Sim 6.1 / Isaac Lab 3.0, Nebius RTX PRO 6000 VM).** The starter scene was rebuilt to match our
   station: white light box (dimensions of the NVIDIA SO-101 workshop light box), two fixed cameras on the top edges
   of the walls (`front` facing the arm, `side` on the arm's right) instead of a wrist camera, blue 30 mm cube, white
   arm, and the A6 flyer that serves as placement target at our station (photo texture, fixed position matched against
   the real camera frames). The scripted expert recorded 120 successful pick-and-place episodes (failed attempts are
   discarded by the collector). After the station was rebuilt in the afternoon (40 mm cube, flyer turned by 180° and
   moved ~2.5 cm, orange tape grid on the floor) a second sim batch was collected with the 40 mm cube and the new flyer
   pose (`cube40_a`, `cube40_b`). The tape grid is not modelled in sim.
2. **Cosmos 3 Reasoner (curation).** Every sim and real episode video (front camera, 32 frames) was scored by
   `nvidia/Cosmos3-Nano` served with vLLM on the same VM, returning JSON
   (`grasp_success`, `placed_on_target`, `motion_clean` 1–5, `failure_mode`, `notes`).
   **Curation decision:** drop an episode if the grasp failed, the cube did not end on the target, or
   `motion_clean < 3`. Per-dataset effect is in `cosmos/curation.log`; the JSON documents every answer.
3. **Teleop.** Real demonstrations recorded with the SO-101 leader/follower pair at our station (LeRobot 0.6.1,
   `use_degrees=false` so joint units match the sim dataset, h264 video), converted from LeRobot v3 to v2.1 for GR00T.
   Two batches: `station_real` (first layout) and `station_real_v2` (rebuilt layout); together at most 50.
4. **Train (Nebius VM).** GR00T N1.7-3B, LLM frozen, visual encoder + projector + diffusion head tuned, batch 16.
   Real data is up-weighted by listing datasets several times in the GR00T mixture: in the intermediate `main_v2` run old sim 1x,
   new sim 2x, first real batch 4x, rebuilt-layout real batch 6x (see `manifest.json`).
   A planned no-curation ablation was dropped: the reasoner rejected no episode of the first batch, so it would have
   trained on identical data.
5. **Evaluate.** `code/scripts/eval_real.py` runs the checkpoint on the real arm through the policy server.

## What the real-arm test showed

The sim + real checkpoints (`main`, and `main_v2` with the 40 mm sim batch) jittered on the real arm and drove it into
poses outside the real data (elbow ≈ −110, shoulder lift ≈ +80). Comparing the dataset statistics explains it: the sim
joint convention does not match the real arm (mean shoulder lift +15 in sim vs −33 real, elbow −3 vs +40, gripper 49 vs
11 in the [−100, 100] units), so the sim frames teach a mirrored arm. With no time left to fit a sim→real joint mapping,
the submitted checkpoint is trained on the 32 Cosmos-curated real episodes only (first batch 1x, rebuilt-layout batch 2x).
Fixing the joint mapping in `scripts/hdf5_to_lerobot.py` / the collector is the first next step; the 40 mm sim episodes
(`tools/collect_cube40.sh`, `cube40_a`, `cube40_b`) can then be reused.

## Not done

Cosmos Transfer 2.5 (sim → real look) was not run: model access was not granted in time and its ~65 GB / 15–35 min
per clip did not fit next to training within the hackathon window. `cosmos_transfer_augmented` is therefore 0 in the
manifest.
