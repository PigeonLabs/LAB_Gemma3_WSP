import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    lines = ['# Gemma 3 4B 실행 결과', '',
             '이 문서는 저장된 실행 결과에서 자동 생성한다. 예비실험과 본실험을 구분한다.', '']
    for dtype in ['float32', 'bfloat16']:
        path = ROOT / f'artifacts/preflight_{dtype}.json'
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        lines += [f'## 행동 보존 — {dtype}', '', f"상태: {r['status']}", '',
                  '| 조건 | 변경된 가중치 수 | 전체 로짓 동일 | 생성 동일 | 원복 동일 |',
                  '|---|---:|---|---|---|']
        for v in r['variants']:
            lines.append(f"| {v['plan']['id']} | {v['changed_entries']} | "
                         f"{all(p['bitwise_equal'] for p in v['prompts'])} | "
                         f"{all(p['ids_equal'] for p in v['generations'])} | {v['restored_exactly']} |")
        lines += ['', '범위: 텍스트 12개 전체 로짓, 3개 최대 32-token 생성. '
                  '전 입력/이미지에 대한 부동소수점 전수 검증은 아니다.', '']
    for filename in ['circuit_pilot.json','circuit_forced_choice.json']:
        path = ROOT / 'artifacts' / filename
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        lines += [f'## 독립 설명자 — {filename}', '',
                  f"상태: {r['status']}; 정답: {r['truth']}; "
                  f"영향 A={r['influence_A']:.9g}, B={r['influence_B']:.9g}.", '',
                  '| 조건 | 질문 표현 | 답 | 정답 여부 |', '|---|---:|---|---|']
        for v in r['results']:
            lines.append(f"| {v['condition']} | {v['wording']} | {v['answer']} | {v['correct']} |")
        lines += ['', '한 회로의 탐색적 예비실험이다. 목표 오답 조작이나 전역 행동 설명, '
                  '가중치 분석의 일반적 인과적 기여, 거짓말쟁이 뉴런 발견을 확증하지 않는다.', '']
        if filename == 'circuit_forced_choice.json':
            lines += ['강제 선택 결과는 ANSWER: A/B 완성의 조건부 log probability 비교이다. '
                      '자연어 설명 생성의 성공이나 충분한 확신을 의미하지 않는다.', '']
            lines += ['이번 실행의 해석: 원본은 질문 표현에 따라 판정이 달라져 baseline부터 '
                      '안정적이지 않았다. 가중치 기반 조건에서 오답으로 바뀐 사례가 있지만 '
                      '위치 무작위화 조건에서도 같은 오답이 발생했다. 따라서 설명자의 표현 '
                      '민감성은 관측되었으나 가중치 분석의 추가 기여나 목표 조작 성공을 '
                      '입증한 결과는 아니다.', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in ['scripts','wsp','tests','plans'] for p in (ROOT/folder).rglob('*')
               if p.is_file() and '__pycache__' not in str(p)}
    (ROOT/'artifacts/source_manifest.json').write_text(json.dumps(sources, indent=2))
    print(ROOT/'RESULTS.md')


if __name__ == '__main__': main()
