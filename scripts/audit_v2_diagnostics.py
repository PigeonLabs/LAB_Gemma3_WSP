"""Post-hoc diagnostic audit. No inference, no rescoring, no gate changes."""
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wsp.v2.state import ROOT,PUBLIC,config,verify,atomic_json,file_hash
from wsp.v2.data import diagnostics
from wsp.v2.observer import framed,decode
from wsp.v2.arithmetic import Calculator
from wsp.study import require_freeze

def first_object(text):
    text=text.strip()
    if text.startswith('```'):
        text=text.split('\n',1)[1] if '\n' in text else ''
    return json.JSONDecoder().raw_decode(text)[0]

def main():
    verify(); require_freeze()
    files=[p for folder in ['diagnostics','diagnostics_exact'] for p in sorted((PUBLIC/folder).glob('chunk-*.jsonl'))]
    before={str(p.relative_to(ROOT)):file_hash(p) for p in files}
    rows=[json.loads(line) for p in files for line in p.read_text().splitlines()]
    cs=diagnostics(); cases={c['id']:c for c in cs}
    expected={f'{c["id"]}-{mode}-F{f}' for c in cs for mode in ['D0','D1','D2'] for f in range(4)}
    expected|={f'{c["id"]}-D3-F{f}' for c in cs if c['n']==16 for f in range(4)}
    assert len(rows)==len({r['trial_id'] for r in rows})==640
    assert {r['trial_id'] for r in rows}==expected
    model_manifest=json.loads((ROOT/'artifacts/model_manifest.json').read_text())
    for item in model_manifest['files']:
        assert file_hash(ROOT/config()['target']/item['file'])==item['sha256'],item['file']
    modes={}; examples={}
    for mode in ['D0','D1','D2','D3']:
        rr=[r for r in rows if r['mode']==mode]
        rel=[]; swapped=0; negative=0; code=Counter(); axes=Counter(); truthcodes=Counter()
        for r in rr:
            c=cases[r['circuit_id']]; _,mapping=framed(c,r['frame'])
            code[r['answer']]+=1
            if r['answer'] in ['A','B']: axes['X' if r['answer']==mapping['X'] else 'Y']+=1
            else: axes['INVALID']+=1
            truthcodes[next((v for v in ['A','B'] if decode(v,r['frame'])==r['truth']),'TIE')]+=1
            target=[c['oracle']['influence_A'],c['oracle']['influence_B']]
            if r['frame']&1: target.reverse()
            if r['numeric'] is not None:
                rel.append(max(abs(a-b)/max(abs(b),1e-10) for a,b in zip(r['numeric'],target)))
                swapped+=all(math.isclose(a,b,rel_tol=.01,abs_tol=1e-10) for a,b in zip(r['numeric'],target[::-1]))
                negative+=any(v<0 for v in r['numeric'])
        modes[mode]={'planned':len(rr),'correct':sum(r['correct'] for r in rr),
            'joint_correct':sum(r['joint_correct'] for r in rr),'answer_codes':dict(code),'displayed_axis_choices':dict(axes),
            'expected_answer_codes':dict(truthcodes),'valid_numeric_count':len(rel),
            'both_numeric_within_1pct':sum(v<=.01 for v in rel),'both_numeric_within_10pct':sum(v<=.1 for v in rel),
            'median_max_relative_numeric_error':statistics.median(rel) if rel else None,
            'swapped_numeric_within_1pct':swapped,'negative_influence_outputs':negative,
            'numeric_exactly_one_one':sum(r['numeric']==[1.,1.] for r in rr),
            'episodes_with_executed_operations':sum(bool(r['trace']) for r in rr),
            'executed_operations':sum(len(r['trace']) for r in rr),
            'generation_tokens':sum(r['generated_tokens'] for r in rr),'inference_seconds':sum(r['seconds'] for r in rr)}
        if mode!='D0':
            r=rr[0]; c=cases[r['circuit_id']]
            examples[mode]={'trial_id':r['trial_id'],'selection':'first recorded episode in this mode; illustrative, not representative',
                'oracle':c['oracle'],'reported_numeric':r['numeric'],'final':r['final'],
                'first_response_excerpt':r['turns'][0]['response'][:700]}
    prefix=Counter()
    for r in [x for x in rows if x['mode']=='D2']:
        try: obj=first_object(r['turns'][0]['response'])
        except (ValueError,IndexError): prefix['first_object_not_parseable']+=1; continue
        prefix['first_object_parseable']+=1
        if not isinstance(obj,dict) or set(obj)!={'operations'} or not isinstance(obj['operations'],list):
            prefix['wrong_request_schema']+=1;continue
        if len(obj['operations'])>4: prefix['batch_exceeds_four']+=1;continue
        table,_=framed(cases[r['circuit_id']],r['frame'])
        calc=Calculator([[float(f'{v:.12g}') for v in row] for row in table])
        try: calc.apply(obj['operations'])
        except (ValueError,TypeError,IndexError,KeyError,FloatingPointError): prefix['prefix_not_executable']+=1
        else: prefix['prefix_executable']+=1
    timing={}
    for n in [16,8,4]:
        rr=[r for r in rows if r['mode']=='D2' and r['n']==n][:50]
        values=sorted(r['seconds'] for r in rr)
        timing[str(n)]={'first_episodes':len(rr),'median_seconds':statistics.median(values),
            'mean_seconds':statistics.mean(values),'p90_seconds_nearest_rank':values[math.ceil(.9*len(values))-1],
            '8448_episode_hours_at_observed_mean':statistics.mean(values)*8448/3600,
            'caveat':'Failure-dominated diagnostic workload extrapolation, NOT a valid-observer throughput benchmark or promised ETA.'}
    stage=json.loads((ROOT/'artifacts/stages/20261103T141642125117Z-v2-diagnose.json').read_text())
    duration=(datetime.fromisoformat(stage['ended_at'])-datetime.fromisoformat(stage['started_at'])).total_seconds()
    profile={str(n):{'circuits':16,'ties':sum(c['oracle']['truth']=='TIE' for c in cs if c['n']==n),
        'margin_below_point_two':sum(c['oracle']['relative_margin']<.2 for c in cs if c['n']==n),
        'median_margin':statistics.median(c['oracle']['relative_margin'] for c in cs if c['n']==n)} for n in [16,8,4]}
    assert before=={str(p.relative_to(ROOT)):file_hash(p) for p in files}
    untouched={name:not (PUBLIC/name).exists() for name in ['development_summary.json','calibration_summary.json','main_status.json','analysis.json']}
    assert all(untouched.values())
    result={'status':'verified','scope':'post-hoc descriptive diagnosis; original scores, gates, and raw data unchanged',
        'episodes':640,'raw_sha256':before,'v1_files_and_scientific_freezes_verified':True,'full_model_files_sha256_verified':True,
        'wall_seconds':duration,'modes':modes,'first_response_prefix_audit':dict(prefix),'first_50_timing_by_n':timing,
        'diagnostic_difficulty_profile':profile,'illustrative_examples':examples,'unrun_stage_files_absent':untouched}
    atomic_json(PUBLIC/'diagnostic_audit.json',result,immutable=True)
    write_report(result)
    from plot_v2_diagnostics import build
    build()
    print(json.dumps({k:result[k] for k in ['status','episodes','wall_seconds','modes','first_response_prefix_audit','diagnostic_difficulty_profile']},ensure_ascii=False),flush=True)

def write_report(r):
    m=r['modes'];p=r['first_response_prefix_audit']
    lines=['# v2 원인 분석과 보존 감사','',
        f'**640/640건 완료, 중복·누락 없음. 단계 실행 시간 {r["wall_seconds"]/3600:.2f}시간.** v1 파일/과학 코드, v2 동결 코드, 모델 전체 파일 SHA256을 다시 검증했다. 원시 응답과 최초 채점은 변경하지 않았다.',
        '', '## 관측한 실패 지점','',
        f'- 즉시 선택 D0: {m["D0"]["correct"]}/192 정답. B 코드 선택 {m["D0"]["answer_codes"].get("B",0)}/192. 정답 코드와 입력 축을 교차한 설계에서 코드 선호가 나타났다.',
        f'- 생성 계산 D1: {m["D1"]["correct"]}/192 정답. A 코드 선택 {m["D1"]["answer_codes"].get("A",0)}/192. 같은 체크포인트에서도 응답 방식에 따라 선호 코드가 달라졌다.',
        f'- 도구 D2: 유효 최종 답변 0/192. 실제 연산을 수행한 episode {m["D2"]["episodes_with_executed_operations"]}/192, 실행된 primitive 총 {m["D2"]["executed_operations"]}개. 현재 자유형 JSON 인터페이스에서는 계산 능력을 평가하기 전에 대부분 실패했다.',
        f'- 정확한 네 함수 값 제공 D3: {m["D3"]["correct"]}/64 정답. 표시된 X 선택 {m["D3"]["displayed_axis_choices"].get("X",0)}/64. 두 influence를 모두 1로 출력한 경우 {m["D3"]["numeric_exactly_one_one"]}건. 가중치 계산을 생략해도 판단·표기·수치 문제가 남았다.',
        '', '이는 **현재 모델·프롬프트·도구 인터페이스 구성의 진단**이다. 도구 자체가 무용하다거나 Gemma 3가 어떤 인터페이스에서도 계산할 수 없다고 일반화하지 않는다. v1과 모델/프롬프트가 달라졌으므로 이 결과로 v1의 원인을 단독 확정하지 않는다.',
        '', '## 반올림만의 문제인지 확인','',
        '아래 1%·10% 범위는 사후 원인 분석용이다. 사전등록한 1e-6/1e-10 수치 기준이나 80% 성공 기준을 대체하지 않는다.',
        '', '| 조건 | 숫자 응답 | 두 숫자 모두 1% 이내 | 두 숫자 모두 10% 이내 | 중앙 최대 상대오차 |',
        '|---|---:|---:|---:|---:|']
    for mode in ['D1','D3']:
        x=m[mode];lines.append(f'| {mode} | {x["valid_numeric_count"]} | {x["both_numeric_within_1pct"]} | {x["both_numeric_within_10pct"]} | {x["median_max_relative_numeric_error"]:.3g} |')
    e=r['illustrative_examples']['D3'];o=e['oracle']
    lines+=['',f'고정된 첫 D3 사례 `{e["trial_id"]}`의 실제 영향은 IX={o["influence_A"]:.10g}, IY={o["influence_B"]:.10g}인데, 최종 숫자는 {e["reported_numeric"]}였다. 이는 대표 표본이라는 주장이 아닌 원시 응답을 확인할 수 있는 예시다.',
        '', '## 출력 경계와 연산 문법','',
        '첫 응답의 첫 JSON 객체만 읽는 사후 검사를 추가했다. 원래 실행에서는 그대로 실패이며, 여기서 점수를 복구하거나 후속 모델 호출을 수행하지 않았다.',
        '', '| 첫 응답 점검 | 건수 |','|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in p.items()]
    lines+=['', '첫 객체를 읽는 것만으로도 실행되지 않는 요청은 연산 이름/인수/handle/배치 크기 문제를 포함한다. 그러므로 단순히 JSON 바깥 텍스트를 버리는 수정만으로 80%를 확보한다고 주장할 수 없다.',
        '', '## 과제 구성의 한계','',
        '진단의 4/8뉴런 조건은 기존 16뉴런 회로를 잘라 만든 것이므로 함수와 영향 차이가 함께 달라진다. 난도만 조작한 완벽한 비교가 아니다. 아래 거의 동률인 회로도 제외하지 않았다.',
        '', '| 크기 | 상대 영향 차이 < 0.2 | 동률 | 중앙 상대 영향 차이 |','|---|---:|---:|---:|']
    for n,x in r['diagnostic_difficulty_profile'].items(): lines.append(f'| {n} | {x["margin_below_point_two"]}/16 | {x["ties"]} | {x["median_margin"]:.3f} |')
    lines+=['', '## 실제 처리량과 시간 추정','',
        '각 크기에서 먼저 기록된 D2 50건을 사용했다. 아래 본평가 환산은 **실패가 대부분인 현재 작업 부하**를 그대로 반복한다는 산술 계산이다. 유효한 설명자 구성의 처리량 검증이 아니며, 배치 가속도 측정하지 않았다. 기존 본평가 20–40시간 예측을 현재 실행 속도에 그대로 적용할 근거는 없다.',
        '', '| 크기 | 평균 초/episode | 중앙 초/episode | 8,448건 평균 속도 환산 |','|---|---:|---:|---:|']
    for n,x in r['first_50_timing_by_n'].items(): lines.append(f'| {n} | {x["mean_seconds"]:.2f} | {x["median_seconds"]:.2f} | {x["8448_episode_hours_at_observed_mean"]:.1f}시간 |')
    lines+=['', '## 이어갈 때의 경계','',
        '**80% 기준은 유지한다. 독립 교정과 본 공격 실험은 아직 실행하지 않았다.** 사용자가 반복 자동화 없이 현재 실행만 진행하도록 지정했으므로 다음 GPU 단계는 시작하지 않는다.',
        '', '현재 동결 구성으로 재개한다면 개발 16/8/4 비교→선택된 한 크기의 독립 교정 순서를 지켜야 한다. 인터페이스를 개선한다면 기존 v2 결과를 보존하고 새 버전으로 등록해야 한다. 개선 후보는 답을 제공하지 않는 구조화된 도구 출력, 올바른 연산 경계, 정답과 무관한 예제에 대한 도구 사용 점검이다. 이러한 수정만으로 성능이 회복될지는 아직 실험하지 않았다.',
        '', '[전체 진단 표](DIAGNOSTIC_REPORT.md) · [감사 원자료](diagnostic_audit.json) · [계획과 선행연구](../../docs/STUDY_V2.md) · [추가 참고 자료의 적용 범위](../../docs/V2_REFERENCE_NOTES.md)',
        '', '![실측 진단 결과](../../docs/figures/v2_diagnostics_measured.svg)']
    (PUBLIC/'DIAGNOSTIC_AUDIT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
