"""Descriptive reporting only; never changes frozen selection or evaluation."""
import json
import statistics
import html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'artifacts/study_v2'

def build():
    if not (PUBLIC/'diagnostic_summary.json').exists(): return
    summary=json.loads((PUBLIC/'diagnostic_summary.json').read_text())
    rows=[json.loads(line) for folder in ['diagnostics','diagnostics_exact']
          for p in sorted((PUBLIC/folder).glob('chunk-*.jsonl')) for line in p.read_text().splitlines()]
    lines=['# v2 진단 결과','',
      '공개 v1 회로에서 수행한 원인 분해 실험입니다. 새 독립 교정 결과가 아니며, 본 공격 가설의 성공/실패를 판정하지 않습니다.',
      '', '| 조건 | 정답 | 정답+두 수치 | 4프레임 일관성 | INVALID | 중앙 실행 시간 |',
      '|---|---:|---:|---:|---:|---:|']
    bars=[]
    for key,r in summary.items():
        mode,n=key.split('-N');rr=[v for v in rows if v['mode']==mode and v['n']==int(n)]
        lines.append(f'| {key} | {sum(v["correct"] for v in rr)}/{r["denominator"]} ({r["accuracy"]:.1%}) | '
            f'{sum(v["joint_correct"] for v in rr)}/{r["denominator"]} ({r["joint_accuracy"]:.1%}) | '
            f'{r["consistency"]:.1%} | {r["invalid"]} | {r["median_seconds"]:.2f}초 |')
        bars.append((key,r['accuracy'],r['joint_accuracy']))
    lines+=['','D0는 숫자를 생성하지 않으므로 joint 정확도는 구조적으로 0이며 D2와 동일한 생성 평가로 해석하지 않습니다. '
            'D3는 네 함수 값을 이미 제공한 진단입니다. 모든 조건은 같은 Gemma 3 체크포인트를 사용합니다.',
            '', '## 도구 접근 실패와 계산 실패','',
            '| 크기 | D2 전체 | 유효 최종응답 | 연산≥1건 수행 | protocol 오류≥1 | 토큰 소진 |',
            '|---|---:|---:|---:|---:|---:|']
    for n in [16,8,4]:
        rr=[r for r in rows if r['mode']=='D2' and r['n']==n]
        lines.append(f'| {n} | {len(rr)} | {sum(r["decoded_answer"] in ["A","B"] for r in rr)} | '
            f'{sum(bool(r["trace"]) for r in rr)} | {sum(bool(r["errors"]) for r in rr)} | {sum(r["budget_exhausted"] for r in rr)} |')
    errors={}
    for r in rows:
        if r['mode']=='D2':
            labels=set()
            for e in set(r['errors']):
                label='JSON parse' if ('delimiter' in e or 'Expecting' in e or 'Unterminated' in e or 'Extra data' in e) else e
                labels.add(label)
            for label in labels: errors[label]=errors.get(label,0)+1
    lines+=['','오류 분류는 기술적 집계이며 한 episode에 여러 오류가 겹칠 수 있습니다.','',
            '| 오류 | episode 수 |','|---|---:|']
    lines += [f'| {k.replace("|","/")} | {v} |' for k,v in sorted(errors.items(),key=lambda item:-item[1])]
    lines+=['','## 같은 회로에서의 D1/D2 비교','',
            '| 크기 | D1만 정답 | D2만 정답 | 둘 다 정답 | 둘 다 실패 |','|---|---:|---:|---:|---:|']
    for n in [16,8,4]:
        a={(r['circuit_id'],r['frame']):r for r in rows if r['mode']=='D1' and r['n']==n}
        b={(r['circuit_id'],r['frame']):r for r in rows if r['mode']=='D2' and r['n']==n}
        counts=[sum(bool(a[k]['correct'])==x and bool(b[k]['correct'])==y for k in a) for x,y in [(True,False),(False,True),(True,True),(False,False)]]
        lines.append('| '+str(n)+' | '+' | '.join(map(str,counts))+' |')
    lines+=['','표현 4개를 독립 회로로 세지 않습니다. 16개 회로의 탐색적 비교이므로 유의성이나 일반화를 주장하지 않습니다.',
        '', '## 시간 및 다음 단계','',
        f'전체 기록: {len(rows)} episode, 합산 추론 시간 {sum(r["seconds"] for r in rows)/3600:.2f}시간, 생성 {sum(r["generated_tokens"] for r in rows):,}토큰. 모델 적재·파일 기록 시간은 stage wall time에서 별도로 확인합니다.',
        '', '사용자가 반복 자동화 없이 현재 실행만 진행하도록 지정했습니다. 개발/독립 교정/공격 평가는 이 진단 완료에 의해 자동 시작되지 않습니다. '
        '현재 결과를 보고 동결한 v2 설정을 덮어쓰지 않습니다. 재개 시 사전등록한 새 개발→선택된 크기 독립 교정 순서를 지킵니다.',
        '', '원자료: [diagnostics](diagnostics/) · [exact](diagnostics_exact/) · [summary](diagnostic_summary.json).']
    (PUBLIC/'DIAGNOSTIC_REPORT.md').write_text('\n'.join(lines)+'\n')
    h=115+len(bars)*35
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="880" height="{h}" viewBox="0 0 880 {h}">',
       f'<rect width="880" height="{h}" rx="16" fill="#101b2d"/>',
       '<g font-family="sans-serif" fill="#e5edf6"><text x="28" y="35" font-size="22">Diagnostic outcomes — 16 public circuits, four frames</text>',
       '<text x="28" y="61" font-size="14">Cyan: answer · Gold: answer + numeric premises · Descriptive, not fresh calibration</text>']
    for i,(label,a,j) in enumerate(bars):
        y=84+i*35;svg.append(f'<text x="28" y="{y+17}" font-size="14">{html.escape(label)}</text>')
        for off,v,color in [(0,a,'#65d7e8'),(14,j,'#edc56a')]:
            svg.append(f'<rect x="180" y="{y+off}" width="{540*v}" height="10" fill="{color}"/><text x="{190+540*v}" y="{y+off+10}" font-size="11">{v:.1%}</text>')
    svg.append('</g></svg>');(ROOT/'docs/figures/v2_diagnostics.svg').write_text('\n'.join(svg)+'\n')

if __name__=='__main__':build()
