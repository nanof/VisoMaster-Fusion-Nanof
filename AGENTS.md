# Agent guide (Cursor / AI)

Operational doc: **how to work in this repo with minimal exploration and mistakes**. The detailed map is in [`docs/agent-architecture.md`](docs/agent-architecture.md).

## What this project is

**VisoMaster Fusion**: a desktop app (PySide6) for face swapping, restoration, facial editing, and video/image/VR processing, using **ONNX Runtime** / **TensorRT** / **Custom** paths (PyTorch, kernels under `custom_kernels/`).

## Entry points

| What | Where |
|------|--------|
| Main app | `main.py` → `app.ui.main_ui.MainWindow` |
| Generated Qt UI | `app/ui/core/main_window.py` (`Ui_MainWindow`); **avoid hand-editing** unless you know the regen workflow |
| Portable launcher | `app/ui/launcher/main.py` → `LauncherWindow` |
| Video pipeline / queues / recording | `app/processors/video_processor.py` → `VideoProcessor` |
| Single frame / face pipeline | `app/processors/workers/frame_worker.py` → `FrameWorker` |
| Model load, ORT sessions, VRAM | `app/processors/models_processor.py` → `ModelsProcessor` |
| Model catalog and paths | `app/processors/models_data.py`, `model_assets/` |

## UI state: `control`, parameters, markers

- `main_window.control`: **string key → value** dict (toggles, sliders, selections). Keys match IDs in `*_layout_data.py`.
- `main_window.parameters` / per face: `FacesParametersTypes` — same naming conventions.
- Per-frame markers: `MarkerTypes` (`typing_helper.py`).
- **When adding a new control**: define it in the right layout data file and, if needed, the handler in `app/ui/widgets/actions/control_actions.py` (or another action module).

## Folder layout (quick read)

- `app/ui/widgets/actions/`: UI reactions (control changes, video, save/load, jobs, …).
- `app/ui/widgets/*_layout_data.py`: control definitions and tab grouping.
- `app/processors/`: ML and video core; **`external/`** is vendored third-party code.
- `app/helpers/`: utilities (VRAM, transcoding, VR, favorites, …).
- `custom_kernels/`: per-model PyTorch implementations/benchmarks; **not the app shell** (mypy/ruff differ).
- `tests/`: `unit/` and `integration/`; markers in `pyproject.toml`.

## Very large files

`frame_worker.py`, `video_processor.py`, `main_ui.py`, and `models_processor.py` are **functional monoliths**. Before changing them:

1. Find the method via symbol or text search.
2. Respect threads, queues, and locks documented near the code.
3. Prefer localized edits; avoid broad refactors unless explicitly requested.

## Tooling and quality

- Tests: `pytest` (config in `pyproject.toml`). Markers: `slow`, `gpu`, `qt`, `integration`.
- Lint/format: `ruff`, hooks in `.pre-commit-config.yaml`.
- Mypy: `ignore_missing_imports`, `follow_imports=silent`; excludes `external/`, `custom_kernels/`, generated UI.

## Conventions for PRs / changes

- Match style and imports of the file you touch.
- Do not add user-facing markdown docs unless asked.
- Code identifiers and control keys are usually **English** even if comments or messages mix languages.

## Debugging and performance (environment variables)

Common (non-exhaustive): `VISIOMASTER_PERF_BUNDLE=1` enables bundled perf telemetry; `VISIOMASTER_PERF_STAGES`, `VISIOMASTER_PIPELINE_METRICS`, `VISIOMASTER_PREVIEW_GL`, `VISIOMASTER_GL_DEBUG`. Details in `docs/agent-architecture.md`.

## Existing human-facing docs

- [`README.md`](README.md), [`docs/quickstart.md`](docs/quickstart.md), [`docs/user_manual.md`](docs/user_manual.md) — do not duplicate the user manual here.

## When to read `docs/agent-architecture.md`

- Changes to **multi-GPU**, queues, or task **stealing**.
- New **models** or ORT/TRT providers.
- **VR**, transcoding, or OpenGL preview.
- Any bug spanning **UI → VideoProcessor → FrameWorker**.
