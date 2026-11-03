"""Immutable artifacts and verification; no version-1 reporting mutations."""
import json
from pathlib import Path
from wsp.study import ROOT, atomic_json, digest, file_hash, now, Journal, load_rows

PUBLIC = ROOT/'artifacts/study_v2'
PRIVATE = ROOT/'.study_private/v2'

def config():
    return json.loads((ROOT/'plans/study_v2.json').read_text())

def preserve_v1():
    dest = PUBLIC/'v1_preservation.json'
    if dest.exists():
        saved = json.loads(dest.read_text())
        bad = [p for p,h in saved['files'].items() if file_hash(ROOT/p) != h]
        if bad: raise RuntimeError(f'Version-1 evidence changed: {bad}')
        return saved
    paths = list((ROOT/'artifacts').rglob('*')) + list((ROOT/'wsp').glob('*.py'))
    paths += [ROOT/'plans/study_v1.json', ROOT/'docs/STUDY_V1.md', ROOT/'RESULTS.md']
    paths += list((ROOT/'.study_private/v1').rglob('*.json'))
    entries = {str(p.relative_to(ROOT)):file_hash(p) for p in paths if p.is_file()
               and '__pycache__' not in p.parts and not p.is_relative_to(PUBLIC)
               and '-v2-' not in p.name}
    # Hashes expose neither held-out coefficients nor hidden response labels.
    result = {'commit':config()['v1_commit'], 'files':entries, 'created_at':now()}
    atomic_json(dest,result,immutable=True)
    return result

def sources():
    paths = list((ROOT/'wsp/v2').glob('*.py')) + [ROOT/'plans/study_v2.json', ROOT/'scripts/run_study_v2.py']
    return {str(p.relative_to(ROOT)):file_hash(p) for p in paths}

def preregister():
    preserve_v1()
    dest=PUBLIC/'preregistration.json'
    if dest.exists(): return verify()
    manifest=json.loads((ROOT/'artifacts/model_manifest.json').read_text())
    if manifest['status']!='verified' or manifest['source_revision']!=config()['source_revision']:
        raise ValueError('Wrong model identity')
    for item in manifest['files']:
        path=ROOT/config()['target']/item['file']
        # Rehash the actual checkpoint, not only metadata, before the new run.
        if file_hash(path)!=item['sha256']: raise ValueError(f'Model changed: {item["file"]}')
    atomic_json(dest,{'registered_at':now(),'config':config(),'sources':sources(),
                     'model_manifest_sha256':file_hash(ROOT/'artifacts/model_manifest.json')},immutable=True)
    print('v2 preregistration and full model hashes verified',flush=True)

def verify():
    preserve_v1()
    saved=json.loads((PUBLIC/'preregistration.json').read_text())
    if saved['config']!=config(): raise ValueError('Frozen v2 config changed')
    for path,h in saved['sources'].items():
        if file_hash(ROOT/path)!=h: raise ValueError(f'Frozen v2 source changed: {path}')
    return saved

def gate():
    verify()
    p=PUBLIC/'calibration_summary.json'
    if not p.exists() or json.loads(p.read_text()).get('status')!='passed':
        raise RuntimeError('STOP: v2 independent calibration has not passed all gates')
