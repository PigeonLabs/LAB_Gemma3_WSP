# 실행 인계 기록 — 2026-11-03

사용자가 **반복 자동화 없이 현재 실행만 진행**을 선택했다. 반복 heartbeat 생성은 자동 승인 검토에서 거부되었고 실제 생성되지 않았다. 자동화를 재시도하거나 다른 방법으로 우회하지 않는다. 현재 `v2-diagnose` 단계만 완료·게시·보고한다. 이후 개발/교정/본평가는 자동으로 시작하지 않는다.

현재 명령:

```bash
python3 scripts/run_stage.py v2-diagnose -- /media/jeong/ExtHDD/Gemma4_neuron_LAB/.venv/bin/python scripts/run_study_v2.py diagnose
```

현재 unified exec session은 16206이다. 실행기는 stdout을 `artifacts/stages/20261103T141642125117Z-v2-diagnose.log`에 기록한다. 종료하면 동일 basename JSON record와 `.publication/` snapshot을 만든다. 스냅샷은 별도로 GitHub connector에 게시하고 원격 tree/ref를 검증해 receipt를 남겨야 한다. 이 과정을 생략하고 자동 업로드 완료라고 말하지 않는다.

완료된 단계: 23개 단위검사, v1 보존 및 전체 모델 파일 SHA256 검증, v2 사전등록, 새 회로 912개 추출. 마지막 확인된 게시 커밋 `7d550be04334ee8b1e4075bdb61c1df768ad1b80`. 원본 v1 상태는 `f5ac320bee1dc5a9c3896525e8df57e638599122`.

진단은 공개 v1 회로 16개 × 3크기 × 4프레임 × D0/D1/D2 = 576건, 이후 n16에서 D3 64건이다. 현재 이 단계의 수치만 열람해도 된다. v2 독립 교정/최종 평가 계수는 아직 공개하거나 편집자에게 보여주지 않는다. 새 과학 소스와 설정은 preregistration.json의 SHA256으로 동결돼 있다. 초반 도구 호출에서 JSON 및 연산 schema 실패가 관찰됐지만, 이것을 보고 규칙·프롬프트를 수정하지 않는다. 모든 실패를 기록해 예정한 진단을 마친다.

GPU는 escalated exec에서만 보인다. 승인된 로컬 모델만 사용하고, 과거 Gemma 4 12B를 새 실행에 쓰지 않는다. 기존 v1 모듈은 수정하지 않는다. 원시 결과를 지우거나 덮어쓰지 않는다. 별도 중단 시에는 현재 journal에서 재개하며 중복 trial ID를 만들지 않는다.

GitHub 게시 방법은 docs/PUBLICATION.md를 따른다. 로컬 .git은 사용할 수 없는 read-only stub이며 gh/터미널 자격증명이 없다. 연결된 GitHub fetch/create_tree/create_blob/create_commit/update_ref를 사용한다. expected_sha를 지정하고 force하지 않는다. update_ref가 오류를 반환해도 먼저 실제 ref를 읽어 적용 여부를 확인한다. functions store의 publish_helper가 있으면 기존 검증된 helper를 사용할 수 있다. per-file snapshot object를 읽되 단일 exec 출력 약1MB 제한을 넘기지 않는다. safetensors는 base64 blob, 텍스트는 exact content로 보낸다. GitHub tree 전체 파일 SHA를 manifest와 비교한 후 publication.py ack한다.
