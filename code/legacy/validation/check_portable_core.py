#!/usr/bin/env python3
"""Read-only integrity diagnostic for the assembled portable research core.

The only file written is the JSON report under results/validation unless an
explicit report path is supplied. No scientific values are recomputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portable.paths import ProjectConfigurationError, load_project_paths  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, name: str, **details: Any) -> dict[str, Any]:
    return {"check": name, "status": "PASS" if condition else "FAIL", **details}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, help="Portable project root; otherwise detected from this module.")
    parser.add_argument("--report", type=Path, help="Optional report path relative to the project root.")
    args = parser.parse_args()

    try:
        paths = load_project_paths(args.project_root)
        project_root = paths.project_root
        manifest_path = paths.require_asset(paths.config["manifest_path"])
        config_path = project_root / "config/project_paths.json"
        config_text = config_path.read_text(encoding="utf-8")
        config = paths.config
        original_root = (project_root / config["original_workspace_root"]).resolve()
        checks: list[dict[str, Any]] = []
        checks.append(check(manifest_path.is_file(), "project_manifest_readable", path="PROJECT_MANIFEST_V1.csv"))

        with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required_columns = {
                "asset_id", "experiment", "category", "portable_path", "original_path", "sha256",
                "validation_status", "scientific_role", "required_for_reproduction", "public_release_candidate",
            }
            headers_ok = bool(reader.fieldnames) and required_columns.issubset(reader.fieldnames)
            rows = list(reader)
        checks.append(check(headers_ok and bool(rows), "manifest_schema_and_rows", row_count=len(rows), columns_ok=headers_ok))

        required_asset_results = []
        for rel in config.get("required_assets", []):
            try:
                resolved = paths.require_asset(rel)
                required_asset_results.append({"path": rel, "status": "PASS", "size_bytes": resolved.stat().st_size})
            except (FileNotFoundError, ProjectConfigurationError) as exc:
                required_asset_results.append({"path": rel, "status": "FAIL", "error": str(exc)})
        checks.append(check(all(x["status"] == "PASS" for x in required_asset_results), "required_assets_exist", assets=required_asset_results))

        manifest_hash_results = []
        missing_paths = []
        mismatch_paths = []
        source_checks = []
        for row in rows:
            rel = Path(row["portable_path"])
            if rel.is_absolute():
                missing_paths.append(row["portable_path"])
                continue
            staged = (project_root / rel).resolve()
            try:
                staged.relative_to(project_root)
            except ValueError:
                missing_paths.append(row["portable_path"])
                continue
            if not staged.is_file():
                missing_paths.append(row["portable_path"])
                continue
            actual = sha256(staged)
            expected = row["sha256"].strip()
            matches = bool(expected) and actual == expected
            if not matches:
                mismatch_paths.append(row["portable_path"])
            manifest_hash_results.append({"portable_path": row["portable_path"], "expected": expected, "actual": actual, "pass": matches})

            original = row["original_path"].strip()
            if original and not original.startswith("GENERATED:"):
                original_rel = Path(original)
                if not original_rel.is_absolute():
                    original_path = (original_root / original_rel).resolve()
                    original_ok = original_path.is_file() and sha256(original_path) == expected
                    source_checks.append({"original_path": original, "status": "PASS" if original_ok else "FAIL"})
                else:
                    source_checks.append({"original_path": original, "status": "FAIL", "reason": "absolute original_path in manifest"})

        checks.append(check(not missing_paths, "staged_manifest_assets_exist", checked=len(manifest_hash_results), missing=missing_paths))
        checks.append(check(not mismatch_paths, "staged_sha256_matches_manifest", checked=len(manifest_hash_results), mismatches=mismatch_paths))
        checks.append(check(bool(source_checks) and all(x["status"] == "PASS" for x in source_checks), "original_sources_and_frozen_references_match_staged_hashes", checked=len(source_checks), failures=[x for x in source_checks if x["status"] != "PASS"]))

        figure_root = paths.resolve("figure_ready_root", required=True, must_exist=True)
        index_path = project_root / "data/figure_ready/FIGURE_READY_INDEX_V1.csv"
        table_paths = sorted(
            path for path in figure_root.rglob("*.csv") if path.resolve() != index_path.resolve()
        )
        table_checks = []
        for table in table_paths:
            sidecar = table.with_suffix(".metadata.json")
            if not sidecar.is_file():
                table_checks.append({"table": table.relative_to(project_root).as_posix(), "status": "FAIL", "reason": "metadata sidecar missing"})
                continue
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                source_digest = metadata.get("source_sha256")
                table_ok = bool(source_digest) and sha256(table) == source_digest
                required_meta = all(metadata.get(k) for k in ("original_source_path", "transformation", "observable_definition", "units_and_normalization", "validation_status"))
                table_checks.append({"table": table.relative_to(project_root).as_posix(), "metadata": sidecar.relative_to(project_root).as_posix(), "status": "PASS" if table_ok and required_meta else "FAIL", "table_hash_matches_metadata": table_ok, "required_metadata_present": required_meta})
            except (OSError, json.JSONDecodeError) as exc:
                table_checks.append({"table": table.relative_to(project_root).as_posix(), "status": "FAIL", "reason": str(exc)})
        index_ok = False
        index_rows: list[dict[str, str]] = []
        index_error = None
        try:
            with index_path.open(newline="", encoding="utf-8-sig") as stream:
                index_reader = csv.DictReader(stream)
                index_rows = list(index_reader)
                expected_headers = {"table_id", "experiment", "portable_path", "metadata_path", "observable", "normalization", "role", "validation_status", "intended_use"}
                index_ok = bool(index_reader.fieldnames) and expected_headers.issubset(index_reader.fieldnames) and len(index_rows) == len(table_paths)
        except OSError as exc:
            index_error = str(exc)
        checks.append(check(bool(table_paths) and all(t["status"] == "PASS" for t in table_checks) and index_ok, "figure_ready_tables_and_metadata", table_count=len(table_paths), sidecars_ok=sum(t["status"] == "PASS" for t in table_checks), index_rows=len(index_rows), index_ok=index_ok, index_error=index_error, failures=[t for t in table_checks if t["status"] != "PASS"]))

        fig2_report_path = project_root / "results/validation/fig2_local_validation_report_V1.json"
        try:
            fig2_report = json.loads(fig2_report_path.read_text(encoding="utf-8"))
            fig2_checks = fig2_report.get("checks", [])
            fig2_pass = fig2_report.get("status") == "PASS" and bool(fig2_checks) and all(x.get("status") == "PASS" for x in fig2_checks)
            fig2_detail = {"path": fig2_report_path.relative_to(project_root).as_posix(), "report_status": fig2_report.get("status"), "check_groups": len(fig2_checks), "all_groups_pass": fig2_pass}
        except (OSError, json.JSONDecodeError) as exc:
            fig2_pass = False
            fig2_detail = {"path": fig2_report_path.relative_to(project_root).as_posix(), "report_error": str(exc)}
        checks.append(check(fig2_pass, "fig2_archived_local_validation_pass", **fig2_detail))

        absolute_path_hits = sorted(set(re.findall(r"/(?:home|scratch|mnt|SciServer)/[^\s\"']+", config_text)))
        checks.append(check(not absolute_path_hits, "shared_config_has_no_machine_specific_absolute_paths", matches=absolute_path_hits))

        external_root = paths.resolve("external_data_root", required=False)
        checks.append(check(True, "external_data_root_is_optional_and_explicit", configured=external_root is not None, environment_variable=config.get("external_data_environment_variable")))

        overall = all(item["status"] == "PASS" for item in checks)
        report = {
            "schema_version": 1,
            "diagnostic": "portable-core integrity and availability check",
            "status": "PASS" if overall else "FAIL",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": ".",
            "scientific_recomputation_performed": False,
            "figure_rendering_performed": False,
            "source_or_frozen_files_modified": False,
            "manifest_assets_checked": len(manifest_hash_results),
            "figure_ready_tables_checked": len(table_paths),
            "checks": checks,
        }
        output_rel = Path(args.report) if args.report else Path(config["validation_root"]) / "portable_core_validation_V1.json"
        if output_rel.is_absolute():
            raise ProjectConfigurationError("Diagnostic report path must be relative to the project root")
        output_path = (project_root / output_rel).resolve()
        output_path.relative_to(project_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{report['status']}: {len(checks)} checks; {len(manifest_hash_results)} manifest assets; {len(table_paths)} figure-ready tables; report={output_path.relative_to(project_root)}")
        return 0 if overall else 1
    except (ProjectConfigurationError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"PORTABLE CORE VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
