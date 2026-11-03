"""Fixed diagnostic grid, development selection, and one independent gate."""
import json
import statistics
from .state import PUBLIC, PRIVATE, config, verify, atomic_json, digest, Journal, load_rows
from .data import load, diagnostics

def summarize(rows,expected_circuits):
    n=len(expected_circuits); frames=config()['frames']; total=n*len(frames)
    lookup={(r['circuit_id'],r['frame']):r for r in rows}
    if len(lookup)!=len(rows): raise ValueError('Duplicate circuit/frame')
    byframe={str(f):sum(bool(lookup.get((c['id'],f),{}).get('correct')) for c in expected_circuits)/n for f in frames}
    joint=sum(bool(lookup.get((c['id'],f),{}).get('joint_correct')) for c in expected_circuits for f in frames)/total
    consistent=0
    for c in expected_circuits:
        answers=[lookup.get((c['id'],f),{}).get('decoded_answer','INVALID') for f in frames]
        consistent+=len(set(answers))==1 and answers[0] in ['A','B']
    acc=sum(byframe.values())/len(frames); consistency=consistent/n
    passed=(len(rows)==total and acc>=.8 and min(byframe.values())>=.8 and joint>=.8 and consistency>=.9)
    return {'status':'passed' if passed else 'failed','circuit_count':n,'denominator':total,'completed':len(rows),
            'accuracy':acc,'frame_accuracy':byframe,'joint_accuracy':joint,'consistency':consistency,
            'invalid':sum(r['decoded_answer']=='INVALID' for r in rows),
            'median_seconds':statistics.median([r['seconds'] for r in rows]) if rows else None,
            'generated_tokens':sum(r['generated_tokens'] for r in rows)}

def run_grid(name,circuits,modes,observer=None):
    freeze=verify()
    from .observer import Observer
    observer=observer or Observer()
    journal=Journal(PUBLIC/name,{'study':'v2','stage':name,'sources':freeze['sources'],
        'circuit_ids':[c['id'] for c in circuits],'modes':modes,'frames':config()['frames']},chunk_size=4)
    expected=len(circuits)*len(modes)*4
    for c in circuits:
        for mode in modes:
            for frame in config()['frames']:
                ident=f'{c["id"]}-{mode}-F{frame}'
                if ident in journal.by_id: continue
                row=observer.episode(c,frame,mode); row['trial_id']=ident; journal.add(row)
                print(json.dumps({'stage':name,'completed':len(journal.rows),'expected':expected,'id':ident,
                     'correct':row['correct'],'joint_correct':row['joint_correct'],'seconds':round(row['seconds'],2),
                     'generated_tokens':row['generated_tokens'],'errors':row['errors']}),flush=True)
    return journal.rows,observer

def diagnose():
    cs=diagnostics()
    rows,observer=run_grid('diagnostics',cs,config()['diagnostic_modes'])
    exact,observer=run_grid('diagnostics_exact',[c for c in cs if c['n']==16],['D3'],observer)
    summary={}
    for mode in ['D0','D1','D2','D3']:
        for n in config()['sizes']:
            rr=[r for r in rows+exact if r['mode']==mode and r['n']==n]
            if rr: summary[f'{mode}-N{n}']=summarize(rr,[c for c in cs if c['n']==n])
    atomic_json(PUBLIC/'diagnostic_summary.json',summary,immutable=True)
    print(json.dumps(summary),flush=True)

def develop():
    verify()
    dataset=json.loads((PUBLIC/'dataset_summary.json').read_text())
    if dataset['status']!='complete': raise RuntimeError('STOP: incomplete circuit bank')
    summaries={}; observer=None
    for n in config()['sizes']:
        cs=load('development',n)
        rows,observer=run_grid(f'development_n{n}',cs,['D2'],observer)
        summaries[str(n)]=summarize(rows,cs)
    selected=next((n for n in config()['sizes'] if summaries[str(n)]['status']=='passed'),None)
    result={'status':'selected' if selected else 'stopped_development_failure','selected_n':selected,'sizes':summaries}
    atomic_json(PUBLIC/'development_summary.json',result,immutable=True)
    print(json.dumps(result),flush=True)

def calibrate():
    verify()
    dev=json.loads((PUBLIC/'development_summary.json').read_text())
    if dev['status']!='selected': raise RuntimeError('STOP: no development configuration passed')
    cs=load('calibration',dev['selected_n'])
    rows,_=run_grid('calibration',cs,['D2'])
    result=summarize(rows,cs); result['selected_n']=dev['selected_n']
    if result['status']!='passed': result['status']='stopped_baseline_failure'
    atomic_json(PUBLIC/'calibration_summary.json',result,immutable=True)
    atomic_json(PUBLIC/'calibration_circuits.json',cs,immutable=True)
    print(json.dumps(result),flush=True)
