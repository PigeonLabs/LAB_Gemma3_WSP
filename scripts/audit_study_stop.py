"""Audit actual stopping behavior without running any gated experiment."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wsp.study import PUBLIC, require_freeze, atomic_json, load_rows


def main():
    require_freeze()
    summary=json.loads((PUBLIC/'calibration_summary.json').read_text())
    assert summary['status']=='stopped_baseline_failure'
    rows=load_rows(PUBLIC/'calibration')
    assert len(rows)==352 and len({r['trial_id'] for r in rows})==352
    heldout=[r for r in rows if r['split']=='calibration']
    assert len(heldout)==64 and len({r['circuit_id'] for r in heldout})==32
    assert sum(r['answer']==r['truth'] for r in heldout)==summary['correct']
    assert len({r['format'] for r in heldout})==1
    gates=[]
    for stage in ['prepare','preserve-bf16','preserve-fp32','main','prose','analyze','interventions','semantic']:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/run_study.py'),stage],
                              cwd=ROOT,capture_output=True,text=True,timeout=30)
        expected=result.returncode!=0 and 'STOP: observer calibration has not passed' in result.stderr
        gates.append({'stage':stage,'refused_by_calibration_gate':expected,'exit_code':result.returncode})
        if not expected: raise AssertionError(f'Unexpected gate behavior: {stage}: {result.stderr}')
        print(f'{stage}: correctly refused before target inference',flush=True)
    forbidden=['main_status.json','prose_status.json','analysis.json','semantic_summary.json','holdout_circuits.json','frozen_plan.json']
    assert not any((PUBLIC/name).exists() for name in forbidden)
    times=[]
    for p in sorted((ROOT/'artifacts/stages').glob('*v1-*.json')):
        r=json.loads(p.read_text())
        seconds=(datetime.fromisoformat(r['ended_at'])-datetime.fromisoformat(r['started_at'])).total_seconds()
        times.append({'stage':r['stage'],'seconds':seconds,'status':r['status']})
    audit={'status':'verified','calibration_trials':352,'unique_calibration_circuits':32,
           'calibration_answers':64,'entrypoint_gates':gates,'main_results_created':False,
           'test_circuit_contents_published':False,'completed_stage_wall_seconds':times,
           'note':'Stage durations include process startup; manual implementation and GitHub transfer are not included.'}
    atomic_json(PUBLIC/'stop_audit.json',audit,immutable=True)
    print('STOP_AUDIT_PASSED',flush=True)


if __name__=='__main__': main()
