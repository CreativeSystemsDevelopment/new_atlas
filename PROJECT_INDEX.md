# Project Index — ATLAS Annotator XL2000

Use this file as a quick orientation when returning to the repository.

## What this repo is
- Core toolchain for annotation, PDF vector preprocessing, and ML dataset/model workflows used for industrial schematic understanding.
- Includes:
  - Qt desktop annotator (`tools/annotator/annotator.py`)
  - Dataset/model utility scripts (`scripts/`)
  - Google Drive/colab artifact storage (`colab/`)
  - Historical snapshots and experiment folders (`_backup`, `_backup2`, `_zip_check`)

## At-a-glance start points
1. Install dependencies:
   - [`requirements.txt`](requirements.txt)
2. Run the desktop app:
   - [`python tools/annotator/annotator.py "path/to/schematic.pdf"`](README.md)
3. Need model download utilities:
   - [`scripts/download_atlas_runs_artifacts.py`](scripts/download_atlas_runs_artifacts.py)

## Top-level files
- [`README.md`](README.md)  
  User-facing overview and quick usage summary.
- [`requirements.txt`](requirements.txt)  
  Python dependency list.
- [`azure_agent_connection.md`](azure_agent_connection.md)  
  Azure AI Projects connection sample snippet.
- [`PROJECT_INDEX.md`](PROJECT_INDEX.md)  
  You are here.
- [`data_archive/`](data_archive/)  
  Archived CSV datasets used for inventory/downtime ingestion.
- [`.gitignore`](.gitignore)  
  Repo ignore patterns.
- `mar2022_jun2024_indirect_downtime.csv`  
  Legacy root CSV artifact currently present locally.

## Core annotation stack
### `tools/annotator/`
- [`annotator.py`](tools/annotator/annotator.py)  
  PySide6 desktop annotator (component + label + continuation + wire workflows + mask mode + fingerprint matching tools).
- [`preprocess.py`](tools/annotator/preprocess.py)  
  PDF preprocessing pipeline (images, shapes, metadata extraction).
- [`extract_mask_fingerprint.py`](tools/annotator/extract_mask_fingerprint.py)  
  Builds vector-based fingerprints from mask overlays.
- [`fingerprint_distance_search.py`](tools/annotator/fingerprint_distance_search.py)  
  Matching engine for segment-level fingerprint comparisons.
- [`search_mask_pattern.py`](tools/annotator/search_mask_pattern.py)  
  Pattern search utilities using mask signatures.
- [`data/`](tools/annotator/data/)  
  Working annotation data (`pages/`, `master/`, metadata, generated masks).
- [`annotations/`](tools/annotator/annotations/)  
  Annotation JSON output from `annotator.py`.
- `page_*.json` / `master_images.zip` / `*.pdf`  
  Local working artifacts and source PDF references.

## Scripts and automation (`scripts/`)
- `check_annotations.py`  
  Compare/inspect annotation outputs.
- `check_dims.py`  
  Validate image and geometry size assumptions.
- `check_drive.py`  
  List and inspect Google Drive folders/files.
- `check_meta.py`  
  Metadata-focused sanity checks.
- `download_atlas_runs_artifacts.py`  
  Mirror `atlas_runs` from Google Drive into `colab/colab_runs/`.
- `download_model.py`  
  Download model artifacts from Google Drive by path.
- `explore_vectors.py` and `explore_vectors2.py`  
  Vector data exploration helpers.
- `export_components.py`  
  Export component data from annotation outputs.
- `list_drive.py`  
  Lightweight Drive folder listing utility.
- `restore_annotations.py`  
  Restore annotation JSON from exported YOLO labels.
- `test_drive_api.py`  
  Drive API connectivity/permission smoke test.
- `upload_drive.py`  
  Upload dataset structure to Google Drive (`Atlas/train`).

## Colab and ML artifacts (`colab/`)
- `train_colab.ipynb`  
  Jupyter workflow placeholder for training/inference tasks.
- `colab/colab_runs/`  
  Local mirror of Google Drive `atlas_runs` artifacts.
  - `yolo26n_atlas_v1/`
  - `yolo26n_components_v1/`
  - `yolo26s_atlas_v1/`
  - `yolov8n_atlas_v1/`
  
  Each run folder includes:
  - `weights/best.pt`, `weights/last.pt`
  - training curves (`Box*.png`, `results.csv`, `results.png`)
  - sample batch images

## Dataset history and backups
- `atlas-components/`  
  Main local components dataset tree (train/valid).
- `atlas-dataset/` and `20%/`  
  Alternate exported dataset snapshots.
- `dataset_temp/`  
  Temporary export/build area.
- `_backup/`  
  YOLO base dataset backup (6-class format).
- `_backup2/`  
  YOLO expanded-label dataset backup (`atlas_dataset_v2_augmented`).
- `_zip_check/`  
  Zip-check style backup copy.

## Maintenance notes
- Model artifacts are currently large; decide if they should be versioned or treated as local-only cache before pushing.
- If you want, this index can be turned into an auto-generated manifest (scripted tree + file roles) to keep it always in sync.

