"""Refresh the integrity manifest after reviewing an intentional release edit."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def make_manifest(root=ROOT):
    exclude = {'outputs', '__pycache__', '.git', '.venv', 'venv', '.pytest_cache'}
    entries = {}
    for p in sorted(root.rglob('*')):
        relative = p.relative_to(root)
        if not p.is_file() or exclude.intersection(relative.parts) or relative.as_posix().startswith('data/external/'):
            continue
        if p.name == 'file_manifest.json' or p.suffix in {'.pyc', '.pyo'}:
            continue
        entries[relative.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    (root / 'file_manifest.json').write_text(json.dumps(entries, ensure_ascii=False, indent=2) + '\n')
    return len(entries)


if __name__ == '__main__':
    print(f'Recorded {make_manifest()} release files; review this manifest change before committing.')
