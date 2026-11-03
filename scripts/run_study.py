"""Run one preregistered stage; call through scripts/run_stage.py for publication."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def preregister():
    from wsp.study import PUBLIC, config, digest, file_hash, frozen_sources, atomic_json, now
    destination=PUBLIC/'preregistration.json'
    if destination.exists():
        from wsp.study import require_freeze
        require_freeze(); print('Existing preregistration verified'); return
    cfg=config(); manifest=json.loads((ROOT/'artifacts/model_manifest.json').read_text())
    if manifest['status']!='verified' or manifest['source_revision']!=cfg['source_revision']:
        raise ValueError('Wrong target model provenance')
    for f in manifest['files']:
        p=ROOT/cfg['target']/f['file']
        if p.stat().st_size!=f['size']: raise ValueError('Target file size changed')
        if p.suffix!='.safetensors' and file_hash(p)!=f['sha256']:
            raise ValueError('Target runtime file hash changed')
    value={'study_id':cfg['study_id'],'registered_at':now(),'config':cfg,'config_sha256':digest(cfg),
           'model_manifest_sha256':file_hash(ROOT/'artifacts/model_manifest.json'),
           'frozen_sources':frozen_sources(),'scope':'one target checkpoint; assistant-authored static policies',
           'timing_estimate_hours':[38,69],'timing_with_contingency_hours':[50,90],
           'clarifications':['Development selects one of three formats; separate calibration is assessed once.',
             'Common perturbation norm reference is the median over all 1820 four-neuron subsets.',
             'Bonferroni simultaneous bootstrap intervals for the two primary comparisons.',
             'Projected coefficient wire format uses 12 significant digits; truth rechecked.',
             'Scientific modules and config are frozen before dataset extraction. Test contents are not shown to the editor.',
             'Prior download verified full shard SHA256. This stage rechecks sizes and small runtime-file hashes.']}
    atomic_json(destination,value,immutable=True)
    print('PREREGISTERED',cfg['study_id'],value['config_sha256'],flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['preregister','extract','calibrate','power','prepare','preserve-bf16','preserve-fp32','main','prose','analyze','interventions','semantic','report'])
    parser.add_argument('--observer'); parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.limit is not None and args.limit<=0: parser.error('--limit must be positive')
    if args.stage=='preregister': preregister()
    elif args.stage=='extract':
        from wsp.circuits import extract
        extract()
    elif args.stage=='calibrate':
        from wsp.calibration import run
        run(args.observer)
    elif args.stage=='power':
        from wsp.statistics import power_planning
        power_planning()
    elif args.stage=='prepare':
        from wsp.preservation import prepare
        prepare()
    elif args.stage.startswith('preserve-'):
        from wsp.preservation import run
        run('bfloat16' if args.stage=='preserve-bf16' else 'float32',args.limit)
    elif args.stage in ['main','prose']:
        from wsp.evaluation import run
        run(args.stage=='prose',args.observer,args.limit)
    elif args.stage=='analyze':
        from wsp.statistics import analyze
        analyze()
    elif args.stage=='interventions':
        from wsp.interventions import run
        run(args.observer)
    elif args.stage=='semantic':
        from wsp.semantics import run
        run(args.observer)
    else:
        from wsp.reporting import build
        build()


if __name__=='__main__': main()
