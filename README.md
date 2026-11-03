# Gemma 3 4B · Weight-space explanation study

**행동을 보존하는 가중치 편집으로 외부 설명자의 판단을 바꿀 수 있는가?**

**본실험 v1:** 사전등록 완료. [프로토콜](docs/STUDY_V1.md) · [실행 보고서](artifacts/study_v1/REPORT.md). 교정 정확도 **33/64 (51.6%)**, 상태 `stopped_baseline_failure`. 사전등록한 80% 기준에 미달하여 본 공격 실험은 중단했습니다.

![교정 형식별 정확도와 사전등록 기준](docs/figures/study_calibration.svg)

[상세 결과](RESULTS.md) · [실험 프로토콜](docs/EXPERIMENT_PROTOCOL.md) · [원본 로그·결과](artifacts/) · [자동 기록·업로드](docs/PUBLICATION.md)

> **현재 결론 — 예비실험.** FP32/BF16에서 검사한 출력은 편집 전후 동일했다.
> 부분 회로에 대한 외부 모델의 A/B 판단은 달라졌으나, 원본 판단도 질문 표현에
> 따라 달랐다. **가중치 분석의 추가 기여와 ‘거짓말쟁이 뉴런’은 아직 입증하지 못했다.**

## 실험 구조

```mermaid
flowchart LR
    A[Gemma 3 4B 원본 가중치] --> B[가중치 통계 분석]
    B --> C[편집 계획 고정]
    C --> D[Up 8배 / Down 1/8배]
    D --> E[전체 로짓·생성 보존 검사]
    D --> F[실제 가중치의 제한된 회로 추출]
    F --> G[독립 Gemma 4 12B의 A/B 판단]
    E --> H[로그·결과·README 갱신]
    G --> H
    H --> I[GitHub 단계별 업로드]
```

| 역할 | 실제 사용 조건 |
|---|---|
| 대상 모델 | **Gemma 3 4B IT**, 4,300,079,472 parameters |
| 편집자 | 이 대화의 Codex assistant; 정적 가중치 통계로 후보 선택 |
| 독립 설명자 | 로컬 Gemma 4 12B IT, 편집자 대화 이력 없는 별도 문맥 |
| 편집 | text layer 17, 뉴런 8개, 조건당 가중치 40,960개 변경 |
| 비교 조건 | 구조만 / 무작위 위치 / 가중치 기반 / 위치 무작위화 |
| 장비 | DDR4 64 GB · RTX PRO 6000 Blackwell Max-Q 96 GB |

Gemma의 gated MLP `D[GELU(Gx) ⊙ Ux]`에서 `U′ = SU`, `D′ = DS⁻¹`, `G′ = G`로
바꾸면 실수 연산에서는 같은 함수다. 원본 체크포인트는 보존하고 메모리에서
편집·복원한다. GELU gate 자체를 스케일하는 변환은 사용하지 않는다.

## 행동 보존 결과

각 조건에서 **텍스트 12개, 전체 vocabulary logits**, **3개 입력의 최대 32-token
greedy 생성**을 비교했다. 표의 분모는 각 검사에 사용한 입력 수다.

| 정밀도 | 편집 조건 | 전체 로짓 bitwise 일치 | 생성 토큰 일치 | 정확한 원복 |
|---|---|---:|---:|---|
| float32 | structure_only_001 | 12/12 | 3/3 | True |
| float32 | random_locations_001 | 12/12 | 3/3 | True |
| float32 | weight_informed_001 | 12/12 | 3/3 | True |
| float32 | location_shuffled_001 | 12/12 | 3/3 | True |
| bfloat16 | structure_only_001 | 12/12 | 3/3 | True |
| bfloat16 | random_locations_001 | 12/12 | 3/3 | True |
| bfloat16 | weight_informed_001 | 12/12 | 3/3 | True |
| bfloat16 | location_shuffled_001 | 12/12 | 3/3 | True |

이 검사는 유한한 텍스트 입력에 대한 수치 검증이다. 모든 입력·이미지·부동소수점
실행에서의 일치를 전수 확인한 결과는 아니다.

## 설명자 판단: 변화는 관찰, 인과적 기여는 미확인

![조건별 실제 설명자 판정: 하나의 회로, 질문 표현 두 개](docs/figures/observer_answers.svg)

부분 회로의 `{0,1}²` 입력 4개를 전수 계산한 정답은 **A**다.
평균 절대 출력 영향은 A **2.99402914**, B **2.63177431**이다.

| 조건 | 질문 표현 0 | 질문 표현 1 |
|---|---|---|
| original | A | B |
| structure_only | N/A | N/A |
| random_locations | N/A | N/A |
| weight_informed | N/A | N/A |
| location_shuffled | N/A | N/A |

- 가중치 기반 조건의 오답은 위치 무작위화 조건에서도 관찰됐다. 이 결과로
  가중치 분석의 우월성을 주장할 수 없다.
- **회로 하나와 표현 두 개**이며, 두 표현을 독립 모델 표본으로 세지 않는다.
- A/B 완성의 조건부 log probability를 비교했다. 자연어 설명을 생성한 결과가 아니다.
- 최초 자유 응답은 512-token 제한에서 형식을 못 맞춘 5개 응답을 보존하고
  중단했다. INVALID를 공격 성공으로 세지 않는다. 동률은 TIE로 별도 기록한다.
- 여기서 ‘설명’은 **외부 모델의 제한된 회로 해석**이다. 대상 Gemma가 자기 행동을
  설명하는 응답이나 Gemma 전체의 의미적 행동에 대한 주장이 아니다.

## 기록과 재현

| 초기 단계 | 결과·로그 |
|---|---|
| 미러 다운로드·SHA 검증 | [manifest](artifacts/model_manifest.json) · [log](artifacts/download.log) |
| 정적 가중치 검사 | [통계](artifacts/weight_inspection.json) · [log](artifacts/inspect.log) |
| FP32 검증 | [결과](artifacts/preflight_float32.json) · [성공 log](artifacts/preflight_float32_retry.log) · [최초 실패](artifacts/preflight_float32.log) |
| BF16 검증 | [결과](artifacts/preflight_bfloat16.json) · [log](artifacts/preflight_bfloat16.log) |
| 자유 응답 중단 | [응답·중단 사유](artifacts/circuit_pilot.json) · [log](artifacts/circuit_pilot.log) |
| A/B 강제 선택 | [원본 점수·프롬프트](artifacts/circuit_forced_choice.json) · [log](artifacts/circuit_forced_choice.log) |

### 자동 기록 도입 이후 단계

| 단계 | 실행 상태 | 종료 시각 UTC | 기록 |
|---|---|---|---|
| publication-validation | complete | 2026-11-03T12:46:40.434172+00:00 | [log](artifacts/stages/20261103T124638778262Z-publication-validation.log) · [record](artifacts/stages/20261103T124638778262Z-publication-validation.json) |
| v1-implementation-tests | complete | 2026-11-03T13:22:11.050417+00:00 | [log](artifacts/stages/20261103T132206703466Z-v1-implementation-tests.log) · [record](artifacts/stages/20261103T132206703466Z-v1-implementation-tests.json) |
| v1-preregister | complete | 2026-11-03T13:24:04.305999+00:00 | [log](artifacts/stages/20261103T132403936425Z-v1-preregister.log) · [record](artifacts/stages/20261103T132403936425Z-v1-preregister.json) |
| v1-extract | complete | 2026-11-03T13:24:28.731682+00:00 | [log](artifacts/stages/20261103T132422678297Z-v1-extract.log) · [record](artifacts/stages/20261103T132422678297Z-v1-extract.json) |
| v1-readiness-tests | complete | 2026-11-03T13:25:27.801506+00:00 | [log](artifacts/stages/20261103T132524236295Z-v1-readiness-tests.log) · [record](artifacts/stages/20261103T132524236295Z-v1-readiness-tests.json) |
| v1-calibrate | complete | 2026-11-03T13:29:33.276291+00:00 | [log](artifacts/stages/20261103T132555846747Z-v1-calibrate.log) · [record](artifacts/stages/20261103T132555846747Z-v1-calibrate.json) |

실행 상태와 업로드 상태는 구분한다. GitHub의 파일과 커밋이 공개된 증거이며,
로컬 전송 대기·확인 영수증은 `.publication/`에 보관한다.

```bash
# torch/transformers 등 실행 환경의 Python 절대 경로
export WSP_PYTHON=/absolute/path/to/venv/bin/python
python3 scripts/run_stage.py unit-tests -- "$WSP_PYTHON" -m unittest discover -s tests -v
python3 scripts/run_stage.py inspect -- "$WSP_PYTHON" scripts/inspect_weights.py
python3 scripts/run_stage.py fp32 -- "$WSP_PYTHON" scripts/run_preflight.py --dtype float32
python3 scripts/run_stage.py bf16 -- "$WSP_PYTHON" scripts/run_preflight.py --dtype bfloat16
python3 scripts/run_stage.py observer -- "$WSP_PYTHON" scripts/run_circuit_pilot.py \
  --observer /absolute/path/to/observer --forced-choice
```

최초 다운로드는 `python3 scripts/run_stage.py download -- python3 scripts/download_model.py`.
원본 결과 파일이 이미 있는 검증·설명자 실험은 덮어쓰기를 거부한다. 재실험은
새 결과 경로를 구현하거나 별도 작업 디렉터리에서 수행한다.
[패키지 버전](artifacts/environment.json), [코드 해시](artifacts/source_manifest.json),
[모델 해시](artifacts/model_manifest.json), 편집 행·열의 `artifacts/*.safetensors`를 제공한다.

**업로드 방식:** 이 대화에서는 단계 종료 시 로그·README·그림을 자동 준비하고,
assistant가 연결된 GitHub 도구로 곧바로 게시·검증한다. 독립 프로세스의 무인
푸시는 로컬 Git 인증을 구성한 환경에서 `--publish git`으로 사용할 수 있다.
현재 PC에는 해당 Git 인증이 없으며, 별도 상시 업로더가 실행되는 것은 아니다.

## 모델 출처와 연구 한계

- [공개 미러](https://huggingface.co/unsloth/gemma-3-4b-it), revision
  `bf46152c47f5dd20b896357cb51abc4c03b8ee8c`.
- [공식 모델](https://huggingface.co/google/gemma-3-4b-it), 비교 revision
  `093f9f388b31de276ce2de164bdc2081324b9767`.
- 두 가중치 shard의 실제 SHA-256이 공식 LFS SHA-256과 일치한다. 일부 설정과
  tokenizer 설정·chat template는 미러 버전이므로 전체 실행 환경 동일성은 주장하지 않는다.
- 약 8.6 GB의 원본 모델은 Git에 넣지 않는다. 고정 revision의 다운로드 코드와
  해시를 제공하며, 모델과 추출 가중치에는 원 모델의 Gemma 이용 조건이 적용된다.
- 출발점: [Weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/).
  본 실험은 완성된 신규 기법 검증이 아닌 탐색적 파일럿이다.
- 다음 본실험에는 편집자 정보 접근의 독립 대조군, 사전 고정 목표 오답,
  동일 탐색 예산, baseline이 검증된 여러 회로·설명자, 숨겨진 평가셋이 필요하다.
  뉴런 후보는 제거·복원으로 설명 오류의 인과적 변화를 확인해야 한다.

README와 그림은 `docs/README.template.md` 및 저장된 JSON에서 생성한다.
연구 해석을 수정할 때는 템플릿을 고친 뒤 `python3 scripts/build_readme.py`를 실행한다.
