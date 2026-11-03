"""Evidence-only study summary and five generated SVG panels."""
import html
import json
from .study import ROOT, PUBLIC, load_rows, atomic_json


def read(name):
    path=PUBLIC/name
    return json.loads(path.read_text()) if path.exists() else None


def panel(name,title,items,note,intervals=None):
    height=max(180,100+len(items)*28)
    values=['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="'+str(height)+'" viewBox="0 0 920 '+str(height)+'" role="img">',
            '<title>'+html.escape(title)+'</title>',
            f'<rect width="920" height="{height}" fill="#111827" rx="12"/>',
            '<g font-family="sans-serif" fill="#e5e7eb">',
            '<text x="24" y="35" font-size="21">'+html.escape(title)+'</text>']
    for i,(label,value,extra) in enumerate(items):
        y=64+28*i
        values += [f'<text x="24" y="{y+14}" font-size="13">{html.escape(label)}</text>',
                   f'<rect x="320" y="{y}" width="{max(0,min(1,value))*360:.1f}" height="18" fill="#38bdf8"/>',
                   f'<text x="700" y="{y+14}" font-size="12">{html.escape(extra)}</text>']
        if intervals and label in intervals:
            lo,hi=intervals[label]
            values.append(f'<line x1="{320+360*lo:.1f}" x2="{320+360*hi:.1f}" y1="{y+9}" y2="{y+9}" stroke="#fbbf24" stroke-width="3"/>')
    if not items: values.append('<text x="24" y="85" font-size="17">NOT RUN / no result available</text>')
    values += [f'<text x="24" y="{height-20}" font-size="12">{html.escape(note)}</text>','</g></svg>']
    dest=ROOT/'docs/figures'/name; dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text('\n'.join(values)+'\n')


def build():
    PUBLIC.mkdir(parents=True,exist_ok=True)
    calibration=read('calibration_summary.json'); analysis=read('analysis.json')
    curves=[]
    if analysis:
        curves=[(a,r,f'{r:.1%}') for a,r in analysis['arm_success_rate'].items()]
    panel('study_attack.svg','Targeted wrong explanation / all cases',curves,
          'Yellow: marginal 95% layer-cluster intervals. Paired simultaneous comparisons are in JSON.',
          analysis.get('arm_cluster_95_intervals') if analysis else None)
    checks=load_rows(PUBLIC/'preservation_bfloat16')
    passed=sum(r['passed'] for r in checks)
    panel('study_preservation.svg','Behavior preservation',
          [('BF16',passed/len(checks),f'{passed}/{len(checks)}')] if checks else [],
          'Bitwise logits + exact greedy tokens on a finite text suite; not an all-input proof.')
    layer_items=[]
    if analysis:
        layer_items=[('Layer '+l,values['W'],f'W: {values["W"]:.1%}') for l,values in analysis['layer_rates'].items()]
    panel('study_layers.svg','Held-out layer results',layer_items,'Layers are clusters; prompts and seeds are repeated measurements.')
    intervention=read('intervention_summary.json')
    outcomes=[]
    if intervention and intervention.get('conclusions'):
        outcomes=[(r['plan_id'],float(r['replicated_on_two_new_wordings']),str(len(r['individually_necessary_and_sufficient_units']))+' single-unit candidates') for r in intervention['conclusions']]
    panel('study_reversal.svg','Edit reversal and reintroduction',outcomes,'Conditional on selected successful cases. Not a population attack success rate.')
    rows=load_rows(PUBLIC/'interventions'); defense=[]
    for d in [None,'joint','canonical','exact']:
        selected=[r for r in rows if r['defense']==d and r['mask']==15]
        if selected:
            rate=sum(r['answer']==r['target'] and r['preserved'] for r in selected)/len(selected)
            defense.append((str(d),rate,f'{rate:.1%}, n={len(selected)}'))
    panel('study_defense.svg','Defenses: targeted wrong answers',defense,'Canonicalization can make input bytes identical; that is representation invariance.')
    lines=['# Confirmatory study v1', '',
           'Protocol: [preregistered design](../../docs/STUDY_V1.md). Existing pilot results are separate.', '']
    if calibration:
        lines += [f"**Calibration: {calibration['status']}**",'',
                  f"Selected format: `{calibration['selected_format']}`. Independent accuracy: "
                  f"{calibration['correct']}/{calibration['denominator']} ({calibration['calibration_accuracy']:.1%}); required 80%.",'',
                  '| Development format | Accuracy |','|---|---:|']
        lines += [f'| {f} | {v:.1%} |' for f,v in calibration['development_accuracies'].items()]
        lines += ['',f"Median score time: {calibration['median_score_seconds']:.3f} s. "
                  f"Main A/B scoring-only projection: {calibration['estimated_main_score_hours']:.2f} h "
                  f"(p90-based {calibration['estimated_main_p90_hours']:.2f} h).",'']
        if calibration['status']!='passed':
            lines += ['**The preregistered baseline gate failed. Main attacks, neuron claims and semantic extension were not run.**',
                      'This is a failure to establish a reliable observer, not a negative causal estimate of weight information.','']
    else: lines += ['Calibration has not completed. No confirmatory result is available.','']
    if analysis: lines += [f"Main conclusion: `{analysis['claim']}`",'']
    for file,title in [('study_attack.svg','Attack'),('study_preservation.svg','Preservation'),('study_layers.svg','Layers'),('study_reversal.svg','Reversal'),('study_defense.svg','Defense')]:
        lines += [f'![{title}](../../docs/figures/{file})','']
    (PUBLIC/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return calibration
