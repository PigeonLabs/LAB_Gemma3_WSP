"""Evidence-only v2 index. Does not invoke any version-1 report generator."""
import json
import html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'artifacts/study_v2'

def build():
    PUBLIC.mkdir(parents=True,exist_ok=True)
    summaries={}
    for name in ['dataset','diagnostic','development','calibration']:
        p=PUBLIC/(name+'_summary.json')
        if p.exists(): summaries[name]=json.loads(p.read_text())
    lines=['# Gemma 3 WSP — v2 실행','',
        '실제 가중치 대응 정보가 **행동을 보존하는 설명 조작**에 기여하는지 검증합니다. 타깃·편집자·설명자 모두 Gemma 3 4B IT입니다.',
        '', '[v2 전체 프로토콜·예상 시간](docs/STUDY_V2.md) · [설정](plans/study_v2.json) · [v1 보존 기록](artifacts/study_v2/v1_preservation.json)',
        '', '**v1 결과는 보존합니다.** 교정 33/64(51.56%)로 80% 기준에 미달하여 본 공격 평가를 실행하지 않았습니다. '
        '[v1 원래 README](docs/V1_README_ARCHIVE.md) · [v1 보고서](artifacts/study_v1/REPORT.md) · '
        '[확정 커밋](https://github.com/PigeonLabs/LAB_Gemma3_WSP/commit/f5ac320bee1dc5a9c3896525e8df57e638599122)',
        '', '```mermaid','flowchart LR','  A[기존 결과 보존] --> B[도구·편향 진단]','  B --> C[새 개발 회로: 16 / 8 / 4 뉴런]',
        '  C --> D{독립 교정 80% 이상?}','  D -->|아니오| E[중단·원인 보고]','  D -->|예| F[정보 접근 대조군 편집]',
        '  F --> G[전체 모델 보존 검사]','  G --> H[숨겨진 설명 평가·DAG 재실행]','```','',
        '현재 구현은 진단·개발·독립 교정까지입니다. 뒤 단계는 교정 통과 후 추가 코드 해시를 고정하고 실행합니다. '
        '아래 미실행 항목은 성공률 0%를 뜻하지 않습니다.','', '| 단계 | 저장된 상태 |','|---|---|']
    for name in ['dataset','diagnostic','development','calibration']:
        r=summaries.get(name)
        status=('완료 (세부 수치는 원본 JSON)' if name=='diagnostic' else r.get('status','완료')) if r else '미완료'
        lines.append(f'| {name} | '+(f'[{status}](artifacts/study_v2/{name}_summary.json)' if r else status)+' |')
    lines+=['','![교정 결과; 완료된 구성만 표시](docs/figures/v2_calibration.svg)','',
        '원자료: [v2 폴더](artifacts/study_v2). 즉시 선택·도구 계산·정확값 제공을 구분하며, 형식 실패는 정답으로 세지 않습니다.',
        '', '참고: [Neural Networks Are Already Clean](https://already-clean.goodfire.com/) · '
        '[Gemma 4 E2B neuron map](https://github.com/salixkoreensis94-spec/gemma4-e2b-neuron-map). '
        '가설의 참고 자료이며 해당 모델의 뉴런 의미나 결과를 Gemma 3에 전이하지 않습니다.',
        '', '| v2 단계 | 실행 상태 | 로그 |','|---|---|---|']
    for p in sorted((ROOT/'artifacts/stages').glob('*-v2-*.json')):
        r=json.loads(p.read_text());lines.append(f'| {r["stage"]} | {r["status"]} | [원본]({r["log"]}) |')
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
    bars=[]
    dev=summaries.get('development',{}).get('sizes',{})
    for n,r in dev.items(): bars.append((f'Development n={n}',r['accuracy'],r['joint_accuracy']))
    if 'calibration' in summaries:
        r=summaries['calibration'];bars.append(('Independent calibration',r['accuracy'],r['joint_accuracy']))
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="880" height="360" viewBox="0 0 880 360">',
         '<rect width="880" height="360" rx="16" fill="#101b2d"/>',
         '<g font-family="sans-serif" fill="#e5edf6"><text x="28" y="38" font-size="23">V2 calibration — observed results only</text>',
         '<text x="28" y="65" font-size="14">Cyan: answer accuracy · Gold: answer + numeric premises · Dashed: 80%</text>']
    if not bars: svg.append('<text x="28" y="170" font-size="20">NOT RUN / no completed calibration summary</text>')
    for i,(label,a,j) in enumerate(bars):
        y=96+i*55; svg.append(f'<text x="28" y="{y+20}" font-size="14">{html.escape(label)}</text>')
        for off,v,color in [(0,a,'#65d7e8'),(19,j,'#edc56a')]:
            svg.append(f'<rect x="255" y="{y+off}" width="{500*v}" height="14" fill="{color}"/><text x="{265+500*v}" y="{y+off+12}" font-size="12">{v:.1%}</text>')
    if bars:svg.append('<path d="M655 85 V325" stroke="#ffffff" stroke-dasharray="5 5"/>')
    svg.append('</g></svg>')
    (ROOT/'docs/figures/v2_calibration.svg').write_text('\n'.join(svg)+'\n')

if __name__=='__main__': build()
