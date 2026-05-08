# SAE Feature Explorer

Interactive Bokeh-served explorer for Sparse Autoencoder (SAE) features
trained on vision-transformer activations (DINOv2 / DINOv3 / CLIP). Click
features in a UMAP, see their top-activating images with heatmap overlays,
search by CLIP text, and label features by hand or with Gemini.

A live deployment runs at https://huggingface.co/spaces/Ramnie/sae-explorer.

## Install

The demo runs on CPU; no GPU is required.

```bash
git clone <this repo>
cd smart_init_stability_SAE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default `pip install torch` pulls the CUDA wheel (~2 GB). For CPU only:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Run locally

```bash
bash scripts/run_explorer_local.sh
```

This reads `configs/models.yaml`, downloads every listed model
(`.pt` sidecar + `_heatmaps.pt` + SAE `.pth`) from the Hugging Face
dataset repo into `./local_data/`, downloads + extracts the thumbnails
tarball into `./local_images/`, and launches `bokeh serve` on port 5006.
Re-runs skip already-downloaded files.

Open http://localhost:5006/explorer_app.

## Add a new model to the demo

The demo's model list lives entirely in `configs/models.yaml`. To add one:

1. **Train** an SAE on backbone activations:
   ```bash
   python src/main.py <shards-dir> SI-SAE --d-model 32000 ...
   ```
2. **Precompute** the explorer sidecar:
   ```bash
   python scripts/precompute_explorer_data.py --sae-path <sae>.pth ...
   ```
3. **Precompute heatmaps** (run on a GPU node — one-time):
   ```bash
   python scripts/precompute_heatmaps.py --data <out>.pt --sae-path <sae>.pth ...
   ```
4. **Upload** sidecar + heatmaps + SAE checkpoint to your HF dataset repo:
   ```bash
   bash scripts/submit_hf_upload.sh
   ```
5. **Append** one block to `configs/models.yaml`:
   ```yaml
   - id:         my_new_sae
     label:      My New SAE
     data_file:  explorer_data_my_new_sae.pt
     sae_file:   sae_my_new_sae.pth
     backbone:   dinov3
     layer:      24
     token_type: spatial
   ```
6. **Local re-run** (downloads the new files, picks them up automatically):
   ```bash
   bash scripts/run_explorer_local.sh
   ```
7. **HF Space push**:
   ```bash
   bash scripts/sync_hf_space.sh && (cd hf_space && git push)
   ```

## Add a new image dataset

The image source for a model is set during precompute, not at serve
time — pass `--image-dir /path/to/your/new/images` to
`scripts/precompute_explorer_data.py` and `scripts/precompute_heatmaps.py`,
upload the resulting `.pt` files plus a thumbnail tarball, then add a
registry block as above. The tarball repo is shared across all models in
a registry (`defaults.hf_images_repo`); change it there if you need a new
image source for everything.

## Optional features

| Feature | Enable by | What it does |
|---|---|---|
| Free-text CLIP search | (always on if `transformers` installed) | Encodes any query with CLIP and ranks features by cosine similarity to image embeds. |
| Gemini auto-interp | export `GOOGLE_API_KEY=...` | "Label with Gemini" button calls `gemini-2.5-flash` on the top-activating images and saves the returned label. |
| Persisted feature names | export `HF_TOKEN=...` and `HF_DATASET_REPO=Ramnie/sae-explorer-data` | Names typed into the demo are debounce-pushed to the HF dataset repo so they persist across sessions. |

All three are inert without the relevant environment variables.

## Project layout

```
configs/
  models.yaml                # registry — single source of truth for the demo
scripts/
  explorer_app.py            # Bokeh entry point (composes panels)
  explorer/                  # state, rendering, persistence, html_views,
                             # images, activations, runtime, registry, loaders
  explorer/panels/           # feature_list, clip_search, cross_sae,
                             # patch_explorer, summary
  bootstrap_demo.py          # registry-driven downloader + launcher
  run_explorer_local.sh      # local launcher (calls bootstrap_demo.py)
  precompute_*.py            # sidecar generation pipeline
  sync_hf_space.sh           # push canonical source to the HF Space repo
hf_space/                    # production HF Space repo (gitignored)
src/                         # training / precompute source (separate concern)
requirements.txt             # demo runtime deps
```

## Troubleshooting

- **`bokeh serve` exits with `--data-dir is not a directory`** — the
  registry's `data_file` paths are resolved relative to `--data-dir`. Run
  `bash scripts/run_explorer_local.sh` first to populate `./local_data/`.
- **Image grid shows gray placeholders** — `--image-dir` doesn't contain
  the basenames stored in the .pt file. Set `--extra-image-dir` to a
  second directory if your images live in two trees.
- **"CLIP text search unavailable for this dataset"** — the active
  dataset has no precomputed CLIP scores. Run
  `scripts/add_clip_embeddings.py` against that .pt, or switch to a
  CLIP-enabled dataset in the dropdown.
