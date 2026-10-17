#!/usr/bin/env python3
"""Create a private, consistent SQLite cache archive for local response replay.

Run after the benchmark processes have exited. Database contents may include
complete prompts and responses. The default output is excluded from git.
Extract the archive into the original run directory's parent to restore caches.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error("--run-dir must be an existing benchmark directory")
    output = args.output or ROOT / ".private/artifacts" / f"{run_dir.name}-replay-caches.tar.gz"
    if output.exists():
        parser.error(f"Output exists: {output}; choose a fresh archive")
    sources = sorted(run_dir.rglob("*.sqlite"))
    if not sources:
        parser.error("No SQLite response caches found")
    suite_path = run_dir / "suite.json"
    if suite_path.exists():
        suite = json.loads(suite_path.read_text())
        if any(cell.get("status") == "running" for cell in suite.get("cells", [])):
            parser.error("Suite has a running cell; archive only after benchmark processes exit")
    manifest = {"schema_version": 1, "run_directory_name": run_dir.name,
                "restore_note": "Extract into the original run directory's parent. Keep source/configuration identical for replay.",
                "databases": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="amem-cache-backup-") as temporary:
        temporary_dir = Path(temporary)
        with tarfile.open(output, "w:gz") as archive:
            for index, source in enumerate(sources):
                target = temporary_dir / f"{index}.sqlite"
                with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
                    with sqlite3.connect(target) as backup:
                        original.backup(backup)
                        rows = backup.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
                        check = backup.execute("PRAGMA integrity_check").fetchone()[0]
                        if check != "ok":
                            raise RuntimeError(f"SQLite integrity check failed for {source.name}: {check}")
                member = str(Path(run_dir.name) / source.relative_to(run_dir))
                manifest["databases"].append({"archive_path": member, "response_count": rows,
                                              "bytes": target.stat().st_size,
                                              "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
                archive.add(target, arcname=member)
            encoded = (json.dumps(manifest, indent=2) + "\n").encode()
            info = tarfile.TarInfo(str(Path(run_dir.name) / "cache-archive-manifest.json"))
            info.size = len(encoded)
            archive.addfile(info, io.BytesIO(encoded))
    print(json.dumps({"archive": str(output), "databases": len(sources),
                      "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
