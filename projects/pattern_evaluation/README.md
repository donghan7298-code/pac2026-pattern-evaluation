# 점수 기반 적재 패턴 평가

유효한 배치 후보의 미래 적재 성능을 작은 신경망으로 예측하고, 상위 후보를 실제 Rollout으로 다시 평가하는 AHEAD 구현입니다. 공유 관측 스키마 `2.0`을 사용하며 좌표는 m, 질량은 kg, 상부 하중은 N입니다.

## 실행

저장소 루트에서 Python 3.10 이상으로 실행합니다. GPU와 PyTorch는 필요하지 않습니다.

```bash
python -m pip install -r requirements.txt
python -m projects.pattern_evaluation plan --input projects/pattern_evaluation/artifacts/sample_observation.json
```

`artifacts/ranker.json`은 실행한 학습의 체크포인트이고, `artifacts/learned_config.json`은 검증 에피소드에서 선택한 가중치입니다. 기본 실행은 두 파일을 함께 읽습니다. 학습 데이터·분할·비교 결과는 [검증 보고서](VALIDATION.md)와 `artifacts/*.json`에 기록됩니다.

재학습용 `train.labels.jsonl.gz`, `validation.labels.jsonl.gz`와 864회 실행 결과 `benchmark_episodes.jsonl.gz`를 함께 제공합니다. 파일 크기와 SHA-256은 `artifacts/release_files.json`에서 확인할 수 있습니다.

## 연결

```python
from projects.pattern_evaluation import Planner

planner = Planner.released()  # 프로세스 시작 시 한 번 생성
result = planner.plan(observation, candidates=generated_candidates)
action = result["action"]
```

`candidates=None`이면 공유 코어의 유한한 모서리 후보를 생성합니다. 외부 후보는 `candidate_id`, `position_m: [x,y,z]`, `yaw_deg: 0|90`으로 전달합니다. 후보의 유효성은 평가기에서 다시 검사합니다. `observation`은 `pacdata.observe.observe()`의 결과 또는 [동일한 관측 구조](INTEGRATION.md)여야 합니다.

```bash
python -m projects.pattern_evaluation serve --port 8080
curl -X POST http://127.0.0.1:8080/plan -H 'Content-Type: application/json' --data-binary @projects/pattern_evaluation/artifacts/sample_observation.json
```

응답은 배치 제안입니다. 실제 실행 전 로봇 도달성·IK·충돌·가반하중 검증을 연결해야 합니다. 스키마와 선반 행동 소비 방법은 [연동 명세](INTEGRATION.md)를 참조하세요.

## 점수와 학습

1. 경계·높이·중복·회전·지지·상부 하중·최소 무게중심 여유를 먼저 검사합니다.
2. 지지·무게중심·하중의 가장 약한 안전 지표를 사용합니다. 안전 목표 미달 구간에서는 안전 개선을 먼저 선택하고, 목표 충족 후에는 안전 보상을 포화시켜 효율을 비교합니다.
3. 연속 바닥 여유, 단편화, 높이 편차, 남은 SKU의 바닥 배치 가능성으로 공간을 평가합니다. 현재 박스의 부피만으로 후보를 비교하지 않습니다.
4. 신경망은 현재 상태·후보·관측된 순서·미관측 재고를 보고 미래 평균 적재, 추가 부피, 하위 꼬리 성능, 차단 비율을 학습합니다. 후보 그룹 전체의 상대 순위를 배우는 listwise 손실도 사용합니다.
5. 실행 시 AI Ranking의 Top-K만 동일한 미래 표본으로 상세 평가합니다. 위험은 평균 성능과 하위 꼬리의 차이, 차단 비율로 별도 계산합니다.
6. 가중치는 여러 검증 에피소드를 끝까지 실행한 결과로 자동 탐색합니다. 적재율·성공률·하위 성능을 사용하며, 후보 점수 자체를 최적화 목표로 삼지 않습니다. 민감도와 Pareto 결과도 저장합니다.

학습된 예측은 상태에 따라 달라집니다. 가중치는 학습 실행에서 갱신되어 저장되며 운영 중 매 상자마다 무제한 변경하지 않습니다. 최소 안전 기준은 가중치 학습 대상에서 제외합니다.

## 다시 학습하거나 가중치를 조정하기

```bash
# 전체 학습 → 가중치 탐색 → 미사용 시험 그룹 비교
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m projects.pattern_evaluation.train --out projects/pattern_evaluation/runs/new_run

# 빠른 연결 검사: 작은 별도 데이터로 동일한 학습 경로 실행
OPENBLAS_NUM_THREADS=1 python -m projects.pattern_evaluation.train --quick --out projects/pattern_evaluation/runs/quick_run

# 저장된 교사 표적으로 신경망만 재학습
OPENBLAS_NUM_THREADS=1 python -m projects.pattern_evaluation.fit ranker --train projects/pattern_evaluation/artifacts/train.labels.jsonl.gz --validation projects/pattern_evaluation/artifacts/validation.labels.jsonl.gz --out projects/pattern_evaluation/runs/retrained

# 기존 체크포인트를 유지하고 검증 중단 지점부터 계속 실행
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m projects.pattern_evaluation.release --workers 4
```

새 결과를 적용하려면 `--model runs/.../ranker.json --config runs/.../learned_config.json`을 사용합니다. 기존 배포 결과를 덮어쓰지 않도록 새 출력 디렉터리를 지정합니다.

`release`는 저장된 학습 모델과 가중치를 그대로 사용합니다. 완료 결과가 있으면 재사용하고, 새 계산은 에피소드마다 기록합니다. 평가 기준·모델·가중치가 달라지면 새 `--out` 경로가 필요합니다. `--labels`를 추가하면 누락된 합성 교사 표적을 재생성합니다. `fit weights`는 재고 그룹과 시나리오에 고르게 표본을 배분하며 시험 데이터가 입력되면 거절합니다.

외부 후보의 교사 표적을 만들고 다시 학습할 수 있습니다. 같은 재고와 그 카메라 변형은 하나의 `base_group`으로 묶어야 합니다.

```bash
python -m projects.pattern_evaluation teach --input request.json --base-group inventory_001 --output label.json
OPENBLAS_NUM_THREADS=1 python -m projects.pattern_evaluation.fit ranker --train train.labels.jsonl.gz --validation validation.labels.jsonl.gz --out projects/pattern_evaluation/runs/custom_ranker
OPENBLAS_NUM_THREADS=1 python -m projects.pattern_evaluation.fit weights --episodes data/generated/validation.episodes.jsonl.gz --model projects/pattern_evaluation/runs/custom_ranker/ranker.json --out projects/pattern_evaluation/runs/custom_weights
```

`teach` 결과를 한 줄 JSON으로 모으면 `fit ranker`의 입력이 됩니다. 기존 103차원 구버전 `.states` 파일은 새 미래 출력 표적과 호환되지 않습니다. 관측을 새 `teach` 함수로 다시 평가해야 합니다.

수동 조정도 가능합니다. `config.json`의 여섯 `weights`를 변경하고 `--config`로 적용하거나, 여섯 값만 담은 JSON을 `--weights`로 전달합니다. 음수·NaN은 거절하고 합은 자동으로 1로 정규화합니다.

## 선반과 확장 범위

일반 선반은 `buffer_boxes`, `buffer_capacity`로 표현합니다. `allow_buffer=True`이면 현재 박스의 배치가 막혔을 때 보관하거나 선반의 적재 가능한 박스를 꺼내는 규칙 정책을 사용합니다. 같은 박스는 선반 방향으로 한 번만 이동하고 이동·점유 비용을 집계합니다. 손상 박스는 `ROUTE_NG`로 분리합니다.

PPO는 이번 경로의 필수 학습기가 아닙니다. `PLACE_CURRENT / BUFFER_CURRENT / RETRIEVE_BUFFER / PALLET_CLOSE`를 고르는 상위 정책으로 교체할 수 있는 행동 계약을 제공합니다. 부분 재배치와 실제 팔레트 교체 동작은 추가 제어기에서 처리합니다.

현재 기하 코어는 한 상자의 완전 지지와 가벼운 상자의 상부 배치만 허용하는 보수적인 근사입니다. 공간 지표는 6×6 그리드 대용값이고 Teacher는 제한된 미래 표본의 탐욕 탐색입니다. 실물 안정성이나 최적해를 보장하는 결과로 해석하지 않습니다.

[설계 근거](DESIGN_DECISIONS.md) · [연동 명세](INTEGRATION.md) · [검증 결과](VALIDATION.md)
