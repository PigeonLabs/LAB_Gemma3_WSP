"""Build a GitHub-readable evidence dashboard, without plotting dependencies."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    study_summary = ''
    if root == ROOT and (root/'artifacts/study_v1/preregistration.json').exists():
        import sys
        sys.path.insert(0, str(root))
        from wsp.reporting import build as build_study
        result = build_study()
        study_summary = ('**본실험 v1:** 사전등록 완료. [프로토콜](docs/STUDY_V1.md) · '
                         '[실행 보고서](artifacts/study_v1/REPORT.md).')
        if result:
            study_summary += (f" 교정 정확도 **{result['correct']}/{result['denominator']} "
                              f"({result['calibration_accuracy']:.1%})**, 상태 `{result['status']}`.")
            if result['status'] != 'passed':
                study_summary += ' 사전등록한 80% 기준에 미달하여 본 공격 실험은 중단했습니다.'
            from render_calibration import render
            render(root, result)
            study_summary += '\n\n![교정 형식별 정확도와 사전등록 기준](docs/figures/study_calibration.svg)'
    conditions = ['original', 'structure_only', 'random_locations',
                  'weight_informed', 'location_shuffled']
    pilot_path = root / 'artifacts/circuit_forced_choice.json'
    pilot = json.loads(pilot_path.read_text()) if pilot_path.exists() else None
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="400" viewBox="0 0 900 400" role="img" aria-labelledby="title desc">',
           '<title id="title">Projected circuit observer answers</title>',
           '<desc id="desc">One circuit, two prompt wordings. Ground truth A. Cells show actual answers, not an attack success rate.</desc>',
           '<rect width="900" height="400" rx="16" fill="#111827"/>',
           '<g font-family="sans-serif" fill="#f9fafb">',
           '<text x="28" y="38" font-size="22">Same function, different observer answers</text>',
           '<text x="28" y="65" fill="#cbd5e1" font-size="14">Exploratory: ONE circuit / TWO wordings / correct answer = A</text>',
           '<text x="340" y="99">Wording 0</text><text x="580" y="99">Wording 1</text>']
    rows = []
    for i, cond in enumerate(conditions):
        answers = []
        y = 116 + i * 45
        svg.append(f'<text x="28" y="{y+26}" font-size="16">{html.escape(cond)}</text>')
        for j in range(2):
            match = [r for r in pilot['results'] if r['condition'] == cond and r['wording'] == j] if pilot else []
            answer = match[0]['answer'] if match else 'N/A'
            answers.append(answer)
            color = '#14532d' if pilot and answer == pilot['truth'] else '#78350f' if answer == 'TIE' else '#7f1d1d'
            svg.append(f'<rect x="{320+j*240}" y="{y}" width="210" height="35" rx="6" fill="{color}"/><text x="{425+j*240}" y="{y+24}" text-anchor="middle" font-size="18">{html.escape(answer)}</text>')
        rows.append(f'| {cond} | {answers[0]} | {answers[1]} |')
    svg += ['<text x="28" y="374" fill="#cbd5e1" font-size="14">Green: correct  ·  Red: wrong  ·  Amber: tied scores. No causal advantage established.</text>', '</g></svg>']
    figures = root / 'docs/figures'
    figures.mkdir(parents=True, exist_ok=True)
    (figures / 'observer_answers.svg').write_text('\n'.join(svg)+'\n')
    behavior = []
    for dtype in ['float32', 'bfloat16']:
        path = root / f'artifacts/preflight_{dtype}.json'
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        for v in r['variants']:
            behavior.append(f"| {dtype} | {v['plan']['id']} | {sum(p['bitwise_equal'] for p in v['prompts'])}/{len(v['prompts'])} | {sum(p['ids_equal'] for p in v['generations'])}/{len(v['generations'])} | {v['restored_exactly']} |")
    stages = []
    stage_dir = root / 'artifacts/stages'
    if stage_dir.exists():
        for p in sorted(stage_dir.glob('*.json')):
            r = json.loads(p.read_text())
            stages.append(f"| {r['stage']} | {r['status']} | {r['ended_at']} | [log]({r['log']}) · [record]({p.relative_to(root)}) |")
    template = (root / 'docs/README.template.md').read_text()
    for key, value in {'STUDY': study_summary, 'BEHAVIOR': '\n'.join(behavior) or '| No completed runs | | | | |',
                       'ANSWERS': '\n'.join(rows),
                       'STAGES': '\n'.join(stages) or '| 초기 실행 | 아래 원본 로그 참조 | — | — |'}.items():
        template = template.replace('{{'+key+'}}', value)
    (root / 'README.md').write_text(template)


if __name__ == '__main__':
    build()
