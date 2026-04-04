"""Persist input-face favorites under the project root (survives restarts)."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PySide6 import QtGui

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

_MANIFEST = "favorites_manifest.json"
_VERSION = 1


def favorites_dir(main_window: "MainWindow") -> Path:
    p = main_window.project_root_path / "input_face_favorites"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _manifest_path(d: Path) -> Path:
    return d / _MANIFEST


def _read_manifest(d: Path) -> dict[str, Any]:
    mp = _manifest_path(d)
    if not mp.is_file():
        return {"version": _VERSION, "items": []}
    try:
        data = json.loads(mp.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"version": _VERSION, "items": []}
        items = data.get("items")
        if not isinstance(items, list):
            items = []
        return {"version": int(data.get("version", _VERSION)), "items": items}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {"version": _VERSION, "items": []}


def _write_manifest(d: Path, data: dict[str, Any]) -> None:
    mp = _manifest_path(d)
    tmp = mp.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(mp)


def save_favorite(
    main_window: "MainWindow",
    face_id: str,
    media_path: str,
    cropped_bgr: np.ndarray,
    embedding_store: dict,
) -> None:
    d = favorites_dir(main_window)
    sub = d / str(face_id)
    sub.mkdir(parents=True, exist_ok=True)
    thumb = sub / "thumb.png"
    emb = sub / "embed.pkl"
    ok = cv2.imwrite(str(thumb), cropped_bgr)
    if not ok:
        print(f"[ERROR] Could not write favorite thumbnail: {thumb}")
        return
    try:
        with open(emb, "wb") as f:
            pickle.dump(embedding_store, f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError as e:
        print(f"[ERROR] Could not write favorite embeddings {emb}: {e}")
        try:
            thumb.unlink(missing_ok=True)
        except OSError:
            pass
        return

    manifest = _read_manifest(d)
    items = [x for x in manifest["items"] if str(x.get("id", "")) != str(face_id)]
    items.append({"id": str(face_id), "media_path": str(media_path)})
    manifest["version"] = _VERSION
    manifest["items"] = items
    try:
        _write_manifest(d, manifest)
    except OSError as e:
        print(f"[ERROR] Could not update favorites manifest: {e}")


def delete_favorite(main_window: "MainWindow", face_id: str) -> None:
    d = favorites_dir(main_window)
    sub = d / str(face_id)
    if sub.is_dir():
        try:
            shutil.rmtree(sub)
        except OSError as e:
            print(f"[WARN] Could not remove favorite folder {sub}: {e}")
    manifest = _read_manifest(d)
    items = [x for x in manifest["items"] if str(x.get("id", "")) != str(face_id)]
    manifest["items"] = items
    manifest["version"] = _VERSION
    try:
        _write_manifest(d, manifest)
    except OSError as e:
        print(f"[WARN] Could not update favorites manifest after delete: {e}")


def _prune_manifest_missing_dirs(d: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    kept = []
    for entry in manifest.get("items", []):
        fid = str(entry.get("id", ""))
        if not fid:
            continue
        sub = d / fid
        if (sub / "thumb.png").is_file() and (sub / "embed.pkl").is_file():
            kept.append(entry)
    manifest = dict(manifest)
    manifest["items"] = kept
    manifest["version"] = _VERSION
    return manifest


def load_persisted_favorites(main_window: "MainWindow") -> None:
    from app.ui.widgets import widget_components
    from app.ui.widgets.actions import list_view_actions

    d = favorites_dir(main_window)
    if not d.is_dir():
        return

    manifest = _prune_manifest_missing_dirs(d, _read_manifest(d))
    try:
        _write_manifest(d, manifest)
    except OSError:
        pass

    for entry in manifest.get("items", []):
        fid = str(entry.get("id", ""))
        mp = str(entry.get("media_path", f"Favorite ({fid})"))
        if not fid:
            continue
        sub = d / fid
        thumb_p = sub / "thumb.png"
        emb_p = sub / "embed.pkl"
        if not thumb_p.is_file() or not emb_p.is_file():
            continue
        cropped = cv2.imread(str(thumb_p), cv2.IMREAD_COLOR)
        if cropped is None:
            print(f"[WARN] Could not load favorite thumbnail: {thumb_p}")
            continue
        cropped = np.ascontiguousarray(cropped)
        try:
            with open(emb_p, "rb") as f:
                embedding_store = pickle.load(f)
        except (OSError, pickle.UnpicklingError, AttributeError, EOFError) as e:
            print(f"[WARN] Could not load favorite embeddings {emb_p}: {e}")
            continue
        if not isinstance(embedding_store, dict):
            continue

        h, w = cropped.shape[:2]
        bytes_per_line = 3 * w

        q_image = QtGui.QImage(
            cropped.data,
            w,
            h,
            bytes_per_line,
            QtGui.QImage.Format.Format_BGR888,
        ).copy()

        list_view_actions.add_media_thumbnail_button(
            main_window,
            widget_components.InputFaceCardButton,
            main_window.inputFacesFavoritesList,
            main_window.input_faces,
            q_image,
            media_path=mp,
            cropped_face=cropped,
            embedding_store=embedding_store,
            face_id=fid,
            is_favorite_clip=True,
        )

    if main_window.inputFacesFavoritesList.count():
        main_window.placeholder_update_signal.emit(
            main_window.inputFacesFavoritesList, False
        )
