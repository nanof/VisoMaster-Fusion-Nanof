from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppDisplayMetadata:
    base_title: str
    base_version: str
    short_commit_hash: str | None
    window_title: str
    about_version_text: str


def _strip_hash_suffix(title: str) -> str:
    if title.endswith(")") and " (" in title:
        prefix, _, suffix = title.rpartition(" (")
        candidate = suffix[:-1]
        if candidate and all(ch in "0123456789abcdefABCDEF" for ch in candidate):
            return prefix
    return title


def _strip_version_suffix(title: str) -> str:
    """Remove a trailing ' - X.Y.Z' semver segment, if present."""
    return re.sub(r"\s*-\s*\d+\.\d+\.\d+\s*$", "", title)


def _read_version_from_file(project_root: Path) -> str | None:
    try:
        data = json.loads((project_root / "version.json").read_text(encoding="utf-8"))
        version = data.get("version", "").strip()
        return version or None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _extract_base_version(base_title: str) -> str:
    if " - " not in base_title:
        return "Unknown"
    return base_title.rsplit(" - ", 1)[-1].strip() or "Unknown"


def _resolve_short_commit_hash(project_root_path: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(project_root_path),
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    commit_hash = result.stdout.strip()
    return commit_hash or None


def get_app_display_metadata(
    project_root_path: str | Path, base_title: str
) -> AppDisplayMetadata:
    root = Path(project_root_path)
    clean_base_title = _strip_hash_suffix(base_title.strip()) or "VisoMaster Fusion"

    # Strip any stale version suffix from the Qt title, then get version
    # from version.json (falls back to parsing the title for old installs).
    name_part = _strip_version_suffix(clean_base_title)
    base_version = _read_version_from_file(root) or _extract_base_version(
        clean_base_title
    )

    short_commit_hash = _resolve_short_commit_hash(root)
    base_title_with_version = f"{name_part} - {base_version}"

    if short_commit_hash:
        window_title = f"{base_title_with_version} ({short_commit_hash})"
        about_version_text = f"Version {base_version} ({short_commit_hash})"
    else:
        window_title = base_title_with_version
        about_version_text = f"Version {base_version}"

    return AppDisplayMetadata(
        base_title=base_title_with_version,
        base_version=base_version,
        short_commit_hash=short_commit_hash,
        window_title=window_title,
        about_version_text=about_version_text,
    )
