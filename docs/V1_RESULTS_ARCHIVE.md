# Gemma 3 4B 실행 결과

이 문서는 저장된 실행 결과에서 자동 생성한다. 예비실험과 본실험을 구분한다.

## 행동 보존 — float32

상태: complete

| 조건 | 변경된 가중치 수 | 전체 로짓 동일 | 생성 동일 | 원복 동일 |
|---|---:|---|---|---|
| structure_only_001 | 40960 | True | True | True |
| random_locations_001 | 40960 | True | True | True |
| weight_informed_001 | 40960 | True | True | True |
| location_shuffled_001 | 40960 | True | True | True |

범위: 텍스트 12개 전체 로짓, 3개 최대 32-token 생성. 전 입력/이미지에 대한 부동소수점 전수 검증은 아니다.

## 행동 보존 — bfloat16

상태: complete

| 조건 | 변경된 가중치 수 | 전체 로짓 동일 | 생성 동일 | 원복 동일 |
|---|---:|---|---|---|
| structure_only_001 | 40960 | True | True | True |
| random_locations_001 | 40960 | True | True | True |
| weight_informed_001 | 40960 | True | True | True |
| location_shuffled_001 | 40960 | True | True | True |

범위: 텍스트 12개 전체 로짓, 3개 최대 32-token 생성. 전 입력/이미지에 대한 부동소수점 전수 검증은 아니다.

## 독립 설명자 — circuit_pilot.json

상태: stopped_format_failure; 정답: A; 영향 A=2.99402914, B=2.63177431.

| 조건 | 질문 표현 | 답 | 정답 여부 |
|---|---:|---|---|
| random_locations_001 | 0 | INVALID | False |
| weight_informed_001 | 0 | INVALID | False |
| location_shuffled_001 | 0 | INVALID | False |
| weight_informed_001 | 1 | INVALID | False |
| structure_only_001 | 0 | INVALID | False |

한 회로의 탐색적 예비실험이다. 목표 오답 조작이나 전역 행동 설명, 가중치 분석의 일반적 인과적 기여, 거짓말쟁이 뉴런 발견을 확증하지 않는다.

## 독립 설명자 — circuit_forced_choice.json

상태: complete; 정답: A; 영향 A=2.99402914, B=2.63177431.

| 조건 | 질문 표현 | 답 | 정답 여부 |
|---|---:|---|---|
| random_locations_001 | 0 | A | True |
| weight_informed_001 | 0 | B | False |
| location_shuffled_001 | 0 | B | False |
| weight_informed_001 | 1 | B | False |
| structure_only_001 | 0 | A | True |
| structure_only_001 | 1 | A | True |
| random_locations_001 | 1 | TIE | False |
| original | 0 | A | True |
| original | 1 | B | False |
| location_shuffled_001 | 1 | B | False |

한 회로의 탐색적 예비실험이다. 목표 오답 조작이나 전역 행동 설명, 가중치 분석의 일반적 인과적 기여, 거짓말쟁이 뉴런 발견을 확증하지 않는다.

강제 선택 결과는 ANSWER: A/B 완성의 조건부 log probability 비교이다. 자연어 설명 생성의 성공이나 충분한 확신을 의미하지 않는다.

이번 실행의 해석: 원본은 질문 표현에 따라 판정이 달라져 baseline부터 안정적이지 않았다. 가중치 기반 조건에서 오답으로 바뀐 사례가 있지만 위치 무작위화 조건에서도 같은 오답이 발생했다. 따라서 설명자의 표현 민감성은 관측되었으나 가중치 분석의 추가 기여나 목표 조작 성공을 입증한 결과는 아니다.

