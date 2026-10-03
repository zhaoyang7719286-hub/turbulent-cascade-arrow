"""Resolve portable project paths without legacy-host fallbacks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_RELATIVE = Path("config") / "project_paths.json"
EXTERNAL_DATA_ENV = "JHTDB_EXTERNAL_DATA_ROOT"


class ProjectConfigurationError(RuntimeError):
    """Raised when the portable project configuration is missing or invalid."""


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the nearest project containing both config and the master manifest.

    Search begins at ``start`` (or this module's location) and then the current
    directory. No machine-specific or SciServer path is used as a fallback.
    """
    starts = [Path(start).expanduser()] if start is not None else [Path(__file__).resolve()]
    starts.append(Path.cwd())
    seen: set[Path] = set()
    for initial in starts:
        initial = initial.resolve()
        for candidate in (initial, *initial.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / CONFIG_RELATIVE).is_file() and (
                candidate / "PROJECT_MANIFEST_V1.csv"
            ).is_file():
                return candidate
    raise ProjectConfigurationError(
        "Could not locate portable project root: expected config/project_paths.json "
        "and PROJECT_MANIFEST_V1.csv in the same directory. Pass a path inside the "
        "project or set the working directory to the portable project. No legacy "
        "SciServer path fallback is attempted."
    )


@dataclass(frozen=True)
class ProjectPaths:
    """Validated project roots resolved from one portable configuration."""

    project_root: Path
    config: dict[str, Any]

    def resolve(self, key: str, *, required: bool = False, must_exist: bool = False) -> Path | None:
        """Resolve a configured project-relative path.

        ``external_data_root`` is optional and may be overridden by the configured
        environment variable. Other configured paths must be relative to the
        project root and may not escape it.
        """
        if key == "external_data_root":
            env_name = self.config.get("external_data_environment_variable", EXTERNAL_DATA_ENV)
            raw = os.environ.get(env_name) if env_name else None
            if not raw:
                raw = self.config.get(key)
            if raw in (None, ""):
                if required:
                    raise ProjectConfigurationError(
                        f"External raw data root is required but not configured. Set "
                        f"{env_name} or provide external_data_root in config/project_paths.json."
                    )
                return None
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            path = path.resolve()
            if must_exist and not path.exists():
                raise FileNotFoundError(f"Configured external data root does not exist: {path}")
            return path

        if key == "project_root":
            return self.project_root
        if key not in self.config:
            if required:
                raise ProjectConfigurationError(f"Required path key is missing from config: {key}")
            return None
        raw = self.config[key]
        if raw in (None, ""):
            if required:
                raise ProjectConfigurationError(f"Required path is not configured: {key}")
            return None
        configured = Path(str(raw)).expanduser()
        if configured.is_absolute():
            raise ProjectConfigurationError(
                f"Shared project path {key!r} must be relative, got: {raw!r}"
            )
        path = (self.project_root / configured).resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as exc:
            raise ProjectConfigurationError(
                f"Configured project path {key!r} escapes the portable project root: {raw!r}"
            ) from exc
        if must_exist and not path.exists():
            raise FileNotFoundError(f"Required configured path {key!r} does not exist: {path}")
        return path

    def require_asset(self, relative_path: str | Path) -> Path:
        """Resolve a required asset named relative to the project root."""
        rel = Path(relative_path)
        if rel.is_absolute():
            raise ProjectConfigurationError(f"Required asset must be project-relative: {relative_path}")
        path = (self.project_root / rel).resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as exc:
            raise ProjectConfigurationError(f"Required asset escapes project root: {relative_path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Required portable-core asset is missing: {path}")
        return path


def load_project_paths(
    project_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> ProjectPaths:
    """Load and validate ``project_paths.json`` for a detected project root."""
    root = Path(project_root).expanduser().resolve() if project_root else find_project_root()
    cfg = Path(config_path).expanduser() if config_path else root / CONFIG_RELATIVE
    if not cfg.is_absolute():
        cfg = root / cfg
    cfg = cfg.resolve()
    if not cfg.is_file():
        raise ProjectConfigurationError(f"Portable project configuration is missing: {cfg}")
    try:
        content = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectConfigurationError(f"Cannot read portable project config {cfg}: {exc}") from exc
    if content.get("schema_version") != 1:
        raise ProjectConfigurationError(f"Unsupported project path schema in {cfg}")
    paths = ProjectPaths(root, content)
    # Resolve every mandatory project-owned root once so invalid/absolute values fail early.
    for key in (
        "data_root", "processed_data_root", "figure_ready_root", "experiment_root",
        "result_root", "validation_root", "visualization_root", "environment_root",
    ):
        paths.resolve(key, required=True)
    return paths
