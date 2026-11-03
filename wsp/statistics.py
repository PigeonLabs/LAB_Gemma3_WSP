"""Layer-cluster paired estimates. Repeated prompts/seeds are not sample units."""
import json
import math
import numpy as np
from .study import PUBLIC, config, atomic_json, load_rows, gate


def arm_scores(rows):
    original={(r['circuit_id'],r['wording']):r for r in rows if r['arm']=='O'}
    outcomes={}
    for r in rows:
        if r['arm']=='O': continue
        base=original[(r['circuit_id'],r['wording'])]
        success=bool(base['answer']==r['truth'] and r['answer']==r['target'] and r['preserved'])
        outcomes.setdefault((r['layer'],r['circuit_id'],r['arm']),[]).append(int(success))
    return {key:float(np.mean(values)) for key,values in outcomes.items()}


def analyze():
    gate(); cfg=config()
    for name in ['main','prose']:
        status=json.loads((PUBLIC/f'{name}_status.json').read_text())
        if status['status']!='complete': raise RuntimeError('Analysis requires all planned outcomes, including failures')
    rows=load_rows(PUBLIC/'main'); natural=load_rows(PUBLIC/'prose')
    values=arm_scores(rows); prose_values=arm_scores(natural)
    layers=sorted({key[0] for key in values})
    if len(layers)!=24: raise ValueError('Expected all 24 heldout layers')
    matrix=np.array([[np.mean([v for (l,c,a),v in values.items() if l==layer and a==arm])
                      for arm in cfg['arms']] for layer in layers])
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    samples=matrix[rng.integers(0,len(layers),size=(cfg['bootstrap_replicates'],len(layers)))].mean(1)
    means=matrix.mean(0); comparisons={}
    for j,arm in enumerate(cfg['arms'][1:],1):
        diff=samples[:,0]-samples[:,j]
        # 97.5% per comparison => simultaneous 95% for W-S and W-M by union bound.
        q=[.0125,.9875] if arm in ['S','M'] else [.025,.975]
        interval=np.quantile(diff,q).tolist()
        prose_diff=float(np.mean([v for (l,c,a),v in prose_values.items() if a=='W'])-
                         np.mean([v for (l,c,a),v in prose_values.items() if a==arm]))
        comparisons['W-'+arm]={'difference':float(means[0]-means[j]),'interval':interval,
                               'interval_coverage':.975 if arm in ['S','M'] else .95,
                               'prose_difference':prose_diff,
                               'passes':bool(means[0]-means[j]>=.15 and interval[0]>0 and prose_diff>0)}
    success=all(comparisons['W-'+arm]['passes'] for arm in ['S','M'])
    original=[r for r in rows if r['arm']=='O']
    original_lookup={(r['circuit_id'],r['wording']):r for r in original}
    conditional={}; shifts={}
    for arm in cfg['arms']:
        eligible=[r for r in rows if r['arm']==arm and original_lookup[(r['circuit_id'],r['wording'])]['answer']==r['truth']]
        conditional[arm]={'denominator':len(eligible),
            'rate':sum(r['preserved'] and r['answer']==r['target'] for r in eligible)/len(eligible) if eligible else None}
        differences=[]
        for r in rows:
            if r['arm']!=arm or 'completion_logprob' not in r: continue
            base=original_lookup[(r['circuit_id'],r['wording'])]
            target=r['target']; other=r['truth']
            differences.append((r['completion_logprob'][target]-r['completion_logprob'][other])-
                               (base['completion_logprob'][target]-base['completion_logprob'][other]))
        shifts[arm]=float(np.mean(differences)) if differences else None
    result={'status':'complete','claim':'weight_information_advantage_supported' if success else 'not_established',
            'semantic_gate_passed':success,'unit_of_inference':'24 layers of ONE fixed Gemma checkpoint',
            'arm_success_rate':dict(zip(cfg['arms'],means.tolist())),
            'arm_cluster_95_intervals':{a:np.quantile(samples[:,i],[.025,.975]).tolist() for i,a in enumerate(cfg['arms'])},
            'conditional_success':conditional,'mean_target_logodds_shift':shifts,
            'comparisons':comparisons,'baseline_accuracy':sum(r['answer']==r['truth'] for r in original)/len(original),
            'outcome_counts':{a:{v:sum(r['arm']==a and r['answer']==v for r in rows) for v in ['A','B','UNKNOWN','INVALID','PRESERVATION_FAILED']} for a in ['O']+cfg['arms']},
            'layer_rates':{str(l):dict(zip(cfg['arms'],matrix[i].tolist())) for i,l in enumerate(layers)},
            'limitations':['no independent editor LLM repetitions','one target checkpoint','partial-circuit explanation']}
    atomic_json(PUBLIC/'analysis.json',result,immutable=True)
    print(json.dumps(result,indent=2),flush=True)


def power_planning():
    """Sensitivity analysis, NOT a promise of empirical power from four calibration layers."""
    cfg=config(); rng=np.random.default_rng(cfg['bootstrap_seed']); results=[]
    for discordance in [.2,.4,.6]:
        for icc in [0.,.1,.2]:
            for delta in [.0,.1,.15,.2,.25]:
                if delta>discordance: continue
                probabilities=[(discordance-delta)/2,1-discordance,(discordance+delta)/2]
                independent=rng.choice([-1.,0.,1.],size=(2000,24,8),p=probabilities)
                common=rng.choice([-1.,0.,1.],size=(2000,24,1),p=probabilities)
                clustered=np.where(rng.random((2000,24,1))<icc,common,independent).mean(2)
                mean=clustered.mean(1); se=clustered.std(1,ddof=1)/math.sqrt(24)
                lower=mean-2.2414027*se
                results.append({'discordance':discordance,'within_layer_correlation':icc,'true_advantage':delta,
                                'approx_detection_probability':float(np.mean(lower>0)),
                                'approx_joint_practical_threshold_probability':float(np.mean((lower>0)&(mean>=.15)))})
    atomic_json(PUBLIC/'power_planning.json',{'status':'planning_simulation','replicates':2000,
                'note':'Normal-approximation single-comparison sensitivity; actual analysis uses layer bootstrap. Does not estimate real ICC or certify joint two-comparison power.',
                'results':results},immutable=True)
