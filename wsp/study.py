"""Versioned study configuration, immutable outputs, and resumable chunk journal."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'artifacts/study_v1'
PRIVATE = ROOT / '.study_private/v1'
CONFIG = ROOT / 'plans/study_v1.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def config():
    return json.loads(CONFIG.read_text())


def atomic_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if path.exists() and immutable:
        if json.loads(path.read_text()) != value:
            raise FileExistsError(f'Refusing to replace immutable result: {path}')
        return
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def frozen_sources():
    return {str(p.relative_to(ROOT)): file_hash(p) for folder in ['wsp', 'plans']
            for p in sorted((ROOT/folder).rglob('*')) if p.is_file() and p.suffix in ('.py', '.json')
            and '__pycache__' not in p.parts}


def require_freeze():
    freeze = json.loads((PUBLIC/'preregistration.json').read_text())
    if freeze['config_sha256'] != digest(config()):
        raise ValueError('Preregistered config changed; new version required')
    for name, sha in freeze['frozen_sources'].items():
        if file_hash(ROOT/name) != sha:
            raise ValueError(f'Frozen scientific source changed: {name}')
    return freeze


class Journal:
    """Atomically replace at most 32 records; reject mismatched run provenance."""
    def __init__(self, folder, provenance, chunk_size=32):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        atomic_json(self.folder/'provenance.json', provenance, immutable=True)
        self.rows = []
        for path in sorted(self.folder.glob('chunk-*.jsonl')):
            for line in path.read_text().splitlines():
                if line.strip(): self.rows.append(json.loads(line))
        self.by_id = {r['trial_id']: r for r in self.rows}
        if len(self.by_id) != len(self.rows): raise ValueError('Duplicate journal trial IDs')

    def add(self, row):
        if row['trial_id'] in self.by_id:
            if row != self.by_id[row['trial_id']]: raise ValueError('Conflicting completed trial')
            return
        index = len(self.rows)//self.chunk_size
        rows = self.rows[index*self.chunk_size:] + [row]
        path = self.folder/f'chunk-{index:06d}.jsonl'
        temp = path.with_suffix('.tmp')
        with temp.open('w') as stream:
            for item in rows: stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
        self.rows.append(row); self.by_id[row['trial_id']] = row


def load_rows(folder):
    return [json.loads(line) for p in sorted(Path(folder).glob('chunk-*.jsonl'))
            for line in p.read_text().splitlines() if line.strip()]


def gate():
    require_freeze()
    result = json.loads((PUBLIC/'calibration_summary.json').read_text())
    if result.get('status') != 'passed':
        raise RuntimeError('STOP: observer calibration has not passed; main experiment forbidden')
    return result
