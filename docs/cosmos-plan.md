# Cosmos-Plan (Stand 18.10.2026, 13:00)

Recherche aus den offiziellen Repos und Modellkarten. **[abgeleitet]** = nicht direkt belegt.

## Reihenfolge auf der VM

Nicht gleichzeitig auf der GPU: erst Sammeln (Isaac Sim), dann Reason, dann Transfer, dann Training.
Jedes Werkzeug in eigener venv bzw. eigenem Container, nicht in `env_isaaclab`.

## Vorher manuell (Browser, eingeloggt auf huggingface.co)

- Zugang anfordern: `nvidia/Cosmos-Transfer2.5-2B`, `nvidia/Cosmos-Guardrail1`, `nvidia/Cosmos-1.0-Guardrail`
- Prüfen, ob `nvidia/Cosmos3-Nano` gesperrt ist
- Read-Token auf der VM als `HF_TOKEN` setzen bzw. `hf auth login`
- Foto der echten Station als Referenzbild

## Cosmos 3 Reasoner – Episoden bewerten (10 Punkte)

- Modell: `nvidia/Cosmos3-Nano` (Reasoner-Teil ca. 16–17 GB), Lizenz OpenMDW-1.1
- Cosmos-Reason2 ist archiviert, nur Fallback
- Quelle: https://github.com/nvidia/cosmos → `cookbooks/cosmos3/reasoner/`

```bash
uv venv --python 3.13 --seed --managed-python && source .venv/bin/activate
uv pip install --torch-backend=cu130 "vllm>=0.23.0"
CUDA_VISIBLE_DEVICES=0 vllm serve nvidia/Cosmos3-Nano \
  --tensor-parallel-size 1 --mm-encoder-tp-mode data --async-scheduling \
  --allowed-local-media-path "$HOME/work/datasets" \
  --media-io-kwargs '{"video": {"num_frames": -1}}' --port 8000 \
  --gpu-memory-utilization 0.3
```

Anfrage per OpenAI-Client an `http://localhost:8000/v1`, Video als `video_url` (file-URI), danach der Text,
`extra_body={"media_io_kwargs": {"video": {"num_frames": -1, "fps": 4}}}`. 4 fps entspricht dem Training.

Prompt-Entwurf **[abgeleitet]**: Antwort nach `</think>` als JSON mit
`grasp_success`, `placed_on_pad`, `motion_clean` (1–5), `failure_mode`, `notes`.

Kuratierungsentscheidung (Vorschlag): Episoden mit `grasp_success=false` oder `motion_clean<3` fliegen raus,
bevor sie in Transfer oder Training gehen. Ergebnis-JSON aufheben (Abgabe).

## Cosmos Transfer 2.5 – Sim echt aussehen lassen (10 Punkte)

- Repo: https://github.com/nvidia-cosmos/cosmos-transfer2.5 (`docs/setup.md`, `docs/inference.md`)
- Modell: `nvidia/Cosmos-Transfer2.5-2B`, braucht 65,4 GB VRAM, 96 GB reichen
- Blackwell: `docker/nightly.Dockerfile` nehmen, nicht `Dockerfile`

```bash
sudo apt install git-lfs && git lfs install
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git && cd cosmos-transfer2.5 && git lfs pull
uv python install && uv sync --extra=cu130 && source .venv/bin/activate
python examples/inference.py -i <spec>.json -o outputs/edge
```

Spec pro Episode (nur Kamera `front`):

```json
{
  "name": "ep000000",
  "prompt": "<Beschreibung der echten Station: Licht, Tischoberfläche, Lampe …>",
  "video_path": ".../observation.images.front/episode_000000.mp4",
  "guidance": 3,
  "image_context_path": "<Foto der echten Station>",
  "edge": { "control_weight": 1.0 }
}
```

- Ohne `control_path` berechnet es die Kantenkarte (Canny) selbst
- Gleiche Frame-Zahl und fps wie die Eingabe, die Aktionen bleiben also frame-genau passend
- Laufzeit **[abgeleitet]**: Basis ca. 15–35 min pro 300-Frame-Clip bei 720p. Deshalb `resolution: "480"`,
  nur `front`, und erst 1 Clip testen
- Distilled-Edge ist schneller, verarbeitet aber nur 93 Frames am Stück

## Offen

- Echte Laufzeit auf der RTX PRO 6000
- Ob Cosmos3-Nano gesperrt ist
- Wie Cosmos-Episoden zurück ins LeRobot-Format kommen (Video tauschen, Parquet und Aktionen kopieren)
- Was mit der `wrist`-Kamera passiert (unverändert lassen oder ebenfalls umrechnen)
