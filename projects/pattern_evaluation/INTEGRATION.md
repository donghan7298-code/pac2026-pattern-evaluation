# 관측과 행동 연동

`Planner`는 상태를 직접 변경하지 않습니다. 센서·재고·후보 생성기가 관측을 만들고, 제어기가 응답을 검증한 뒤 상태를 갱신합니다. 계획기가 계산한 `decision_ms`와 로봇 작업 시간은 구분합니다.

## 입력 계약

| 필드 | 내용 |
|---|---|
| `schema_version` | 공유 데이터 스키마 `2.0` |
| `request_id` | 프레임 또는 작업별 고유 ID. 미래 표본 재현 시드에도 사용 |
| `pallet` | `length_m`, `width_m`, `max_height_m`, `max_mass_kg` |
| `state_verified` | 센서와 실행 피드백으로 배치 상태가 확인됐을 때 `true` |
| `placed_boxes` | `box_id`, `sku`, `dimensions_m`, `mass_kg`, `top_load_capacity_N`, `allowed_yaw_deg`, `position_m`, `yaw_deg`, `upper_load_N`, `supporter_id` |
| `current_box` | 같은 물성 필드와 `measurement_valid`, `visual_damage_observed`. 완료 시 `null` |
| `observed_preview` | 순서가 확인된 연속 접두부. `track_id`, `sku`, `dimensions_m`, `nominal_mass_kg`, `top_load_capacity_N`, `allowed_yaw_deg`, `dimension_sigma_m` |
| `unordered_visible_hints` | 순서 미확인 검출. `unseen_inventory`에서 다시 빼지 않음 |
| `unseen_inventory` | 현재 박스와 순서 확인된 preview를 제외한 SKU별 수량. 선반 박스도 별도 관리 |
| `catalog` | SKU별 nominal `dimensions_m`, `mass_kg`, `top_load_capacity_N`, `allowed_yaw_deg` |
| `camera` | `dimension_sigma_m` 등 센서 품질 |
| `buffer_boxes` | 관측된 선반 박스. 측정 물성과 `buffer_moves: 1` 필요 |
| `buffer_capacity` | 설치 선반의 보관 가능 개수. 개발 기본값은 2로 교체 가능 |
| `stream_terminated` | 정상적인 종료 조건 확인 시에만 `true` |
| `distribution_status` | 외부 분포 감지기가 `OOD`로 지정하면 신경망을 건너뛰고 Full Teacher 사용 |
| `coordinate_frame`, `units` | 기본 `pallet`, `{"length":"m","mass":"kg","force":"N"}`. 다른 단위·좌표계는 먼저 변환 |
| `state_version`, `robot_scene_version` | 상태/장애물/EOAT가 바뀔 때 관리자가 갱신. 실행 검증 완료 판정에 둘 다 필요 |
| `robot_profile` | 선택 필드 `max_payload_kg`, `eoat_mass_kg`. 합산 질량 사전 검사에 사용. 로드 CoM·동역학 payload는 외부 검사 |
| `process_state` | 기본 `NORMAL`. `PALLET_CHANGE`, `REPACKING`, `HOLD` 중에는 일반 적재 중단 |
| `pattern_generation_ms` | 외부 후보 생성에 실제로 걸린 시간. 생략 시 합산 계획시간은 `null` |
| `decision_budget_ms` | 선택적 소프트 평가 예산. 안전 마스크는 생략하지 않으므로 마감시간 보장은 아님 |

지지 ID는 실제 하부 상자의 `box_id` 또는 `FLOOR`입니다. `upper_load_N`은 모든 상부 하중의 합이며 새 적재 후 지지 체인을 따라 `mass_kg × 9.81`을 갱신합니다. 서로 중복되는 배치/선반 ID는 거절합니다. `oracle`를 전달하면 오류가 발생합니다.

입고 종료가 확인되지 않은 상태에서 현재 박스가 없으면 `WAIT`입니다. 미관측 재고가 남으면 `EXPECTED_UNSEEN`으로 기다립니다. 손상·측정 오류·미확인 배치 상태는 정상 적재와 분리합니다.

## 외부 후보와 요청

```json
{
  "observation": {"...": "artifacts/sample_observation.json의 관측 구조"},
  "candidates": [
    {"candidate_id": "pattern_01", "position_m": [0.0, 0.0, 0.0], "yaw_deg": 0}
  ],
  "mode": "ahead",
  "allow_buffer": true
}
```

평가 모드는 `greedy`, `current`, `teacher`, `ranking`, `ahead`입니다. 현재 후보 수 제한은 없고 전달된 유효 후보 전체를 Ranking합니다. Top-K는 상세 Rollout의 수만 제한합니다. Teacher도 유한 후보 생성기가 놓친 위치까지 탐색하는 최적해는 아닙니다.

## 행동 소비

| 행동 | 제어기의 상태 갱신 |
|---|---|
| `PLACE_CURRENT` | 로봇 검증·작업 성공 후 현재 박스를 배치 목록에 추가하고 입고 순서를 진행 |
| `BUFFER_CURRENT` | 선반 치수·하중·빈 슬롯 확인과 작업 성공 후 선반에 추가, `buffer_moves=1`, 입고 순서 진행 |
| `RETRIEVE_BUFFER` | 지정 `box_id`를 배치하고 선반 목록에서 제거. 현재 컨베이어 박스는 소비하지 않음 |
| `PALLET_CLOSE` | 팔레트 교체 제어기 호출. 교체 완료 피드백 후 새 배치 상태로 다시 요청 |
| `REMEASURE` | 재측정 또는 복구 후 같은 박스로 재요청 |
| `ROUTE_NG` | 정상 선반과 다른 Inspection/NG 경로로 분기 |
| `HOLD` | 배치 상태·물성 확인 후 재요청 |
| `WAIT` / `DONE` | 기대 입고를 기다리거나 확인된 종료 처리 |

외부 검증이 없는 기본 응답은 `requires_robot_validation=true`, `robot_feasibility=NOT_CHECKED`입니다. 아래 계약의 6개 검사가 모두 통과하면 `EXTERNALLY_VALIDATED`를 반환합니다. 어느 경우에도 평가기가 직접 로봇 작업을 수행하지는 않습니다. 선반 슬롯의 실제 치수·가반하중·경로도 제어기가 검증합니다. 순차 재요청 시 버퍼 이동 횟수를 보존해야 한 번 제한이 유지됩니다.

기존 `PackingEnv`에서 버퍼 없는 정책을 실행할 때는 `to_packing_env_action(result)`로 `PLACE / REMEASURE / ROUTE_DAMAGE / STOP`에 변환합니다. 이 환경이 지원하지 않는 선반 행동은 변환 함수가 오류로 알립니다.

## 반환 점수

`static`에는 공간·안전·하중·균형·시간 대용 지표가, `future`에는 평균 적재 비율·추가 부피·CVaR·최악 표본·차단 비율이 포함됩니다. `diagnostics`는 후보 수, 상세 Rollout 수, 마스크 거절, 모델 사용 상태와 OOD 상태를 제공합니다. `ranking` 모드는 상세 Rollout을 하지 않으므로 `future=null`입니다.

응답 버전은 `ahead-action-2.1`입니다. 기존 호출 방식은 유지하고 다음 필드를 추가했습니다.

| 필드 | 내용 |
|---|---|
| `score_contributions` | 여섯 가중치별 실제 기여도. 합은 `score`. Greedy는 다른 휴리스틱이므로 `null` |
| `metrics.before/after` | 체적률·바닥 점유율·CoG m/정규화·박스별 하중·바닥 하중 분포·접촉면적과 지지 여유 |
| `metrics.sampled_expected_volume_utilization` | 배치 후 체적률 + 미래 샘플의 평균 추가 체적률. 확정 패턴이 아님 |
| `diagnostics.ranking` | 전체 유효 후보의 Ranking 점수, 상세 평가 점수, 선택 여부, 로봇 검사와 요청용 토큰 |
| `diagnostics.rejected[].stage` | `PACKING`, `SAFETY`, `ROBOT`. 로봇 거절과 적재 공간 불가능을 구분 |
| `timing` | 후보 생성/평가/합산 시간 ms, 로봇 예상시간 s/대용값 s, 예산 초과 여부 |
| `execution_ready` | 현재 상태·장면·위치에 묶인 외부 6개 검사가 모두 통과했는지. 자동 실행 명령은 아님 |

`metrics.after.stability.collapse_probability=null`은 실제 낙하 확률을 추정하지 않았다는 뜻입니다. `rigid_tipping_acceleration_proxy_m_s2 = g × 접촉면 가장자리 여유 / 접촉면 위 상부 구조 CoG 높이`는 정적 강체 가정의 비교값이며 로봇 가속도 허용치로 사용하지 않습니다. 바닥 4분할 하중은 바닥 박스에 전달되는 하중을 각 footprint에 균등 분포시킨 대용 지도입니다.

## 로봇 검증 왕복 계약

1. 인식 상태에 `state_version`, 로봇 장면에 `robot_scene_version`을 부여하고 `execution_mode="proposal"`로 호출합니다.
2. 반환된 후보들에 대해 외부 로봇 계획기가 검사를 수행합니다. Packing 검사를 통과해도 경로가 가능하다는 뜻은 아닙니다.
3. 검사 결과를 해당 후보의 `robot_validation`으로 붙여 같은 관측과 함께 `execution_mode="validated"`로 다시 요청합니다.
4. `PLACE_CURRENT` 또는 `RETRIEVE_BUFFER`이면서 `execution_ready=true`일 때 제어기가 상태/장면 버전을 다시 대조합니다. 작업 성공·센서 확인 후에만 상태를 갱신합니다.

```json
{
  "candidate_id": "C17",
  "position_m": [0.4, 0.0, 0.0],
  "yaw_deg": 90,
  "grasp_id": "top-vacuum",
  "approach_id": "vertical-01",
  "robot_validation": {
    "state_token": "반환된 robot.state_token",
    "placement_token": "반환된 robot.placement_token",
    "box_id": "현재 box_id",
    "validator": "경로 계획기 버전",
    "checks": {
      "reachability": true,
      "ik": true,
      "collision_free": true,
      "payload": true,
      "grasp_direction": true,
      "approach_pose": true
    },
    "estimated_cycle_seconds": 7.4
  }
}
```

위 true 값은 형식 예시입니다. 실제 검사 결과로 채워야 합니다. `grasp_id`/`approach_id`는 첫 제안 시부터 후보에 포함시켜야 위치 토큰이 일치합니다. 토큰은 상태 일치 확인용 해시이며 검사기의 진실성이나 신뢰를 인증하지 않습니다. 입력은 신뢰되는 로봇 계획기에서 받습니다.

검사 중 하나라도 false이면 기본 proposal 모드에서도 후보를 제외합니다. 결과가 누락/불완전/오래됐으면 validated 모드에서 선택하지 않습니다. 로봇 검사 때문에 후보가 사라진 경우에는 `PALLET_CLOSE` 대신 `HOLD / ROBOT_VALIDATION_OR_REPLAN_REQUIRED`로 재계획을 요청합니다. 모든 검사와 장면 버전이 있으면 응답은 `robot_feasibility=EXTERNALLY_VALIDATED`입니다.

올바른 상태에 묶인 `estimated_cycle_seconds`는 `handling` 비용에 반영합니다. 계산식은 `min(1, seconds/20)`입니다. 20초는 학습 당시의 정규화 기준이므로 실제 사이클 규모가 크게 달라지면 별도 검증·보정해야 합니다. 예상시간이 없으면 기존 좌표 대용값을 사용하며 `handling_source`로 구분합니다. 로봇 검사시간은 외부 호출에 걸린 시간이라 이 평가기의 `decision_ms`에 포함되지 않습니다.

선반 회수 후보는 요청의 `buffer_candidates={"box_id": [후보들]}`로 전달합니다. `contracts.buffer_observation(prepare_observation(obs), box_id)`가 회수 후보 평가용 상태를 만듭니다. 회수할 박스는 현재 평가 대상으로 옮기고, 기다리는 컨베이어 박스는 확인된 미래 접두부와 `pending_conveyor_box`로 유지합니다. 토큰은 이 평가 상태를 기준으로 합니다. 물리적 선반→팔레트 경로를 검사해야 하며 컨베이어 pick 경로 결과를 재사용하지 않습니다.

## 여러 박스 순서 검증

`evaluate_sequence()` 또는 `POST /evaluate-sequence`에 `{observation, steps}`를 보냅니다. 각 step은 `{box_id, candidate}`입니다. 현재 박스와 확인된 preview는 입력 순서를 지키고, 실제 선반 박스는 그 사이에 선택할 수 있습니다. 미확인 미입고 SKU를 임의의 실제 박스로 만들어 실행 순서에 넣을 수는 없습니다.

```python
from projects.pattern_evaluation import evaluate_sequence

review = evaluate_sequence(observation, [
    {"box_id": "B01", "candidate": {"position_m": [0, 0, 0], "yaw_deg": 0}},
    {"box_id": "B02", "candidate": {"position_m": [0, 0, 0.2], "yaw_deg": 0}}
])
```

각 단계에서 앞 단계의 지지·하중 변화를 적용하고 검사합니다. 성공하면 `packing_valid=true`와 최종 KPI를 반환하지만 로봇 검사가 없으면 `PACKING_VALID_REQUIRES_ROBOT`입니다. 로봇 검사에서 중단되면 `ROBOT_REPLAN_REQUIRED`, `packing_valid=null`로 남은 순서의 packing 검사가 미완료임을 표시합니다. 이 경로의 `mean_static_score`는 미래 Rollout 점수가 아닙니다.

두 번째 이후의 가상 상태 버전은 `원래state_version/planned:1`, `...:2`입니다. 가상 계획을 로봇 시뮬레이션으로 검사할 수 있으나 실제 실행에서는 매 박스 후 센서 관측과 실제 버전으로 다시 검증해야 합니다. `requires_observation_refresh_each_step=true`는 그 계약입니다. 예시: [sequence_request.json](examples/sequence_request.json).

## 참고 코드와 좌표 변환

참고 `palletizing_core`의 후보는 `i/j` 격자, `z` mm, `o=0/1` 방향 번호를 사용합니다. `reference_candidate(candidate, grid_mm)`는 `[i×grid/1000, j×grid/1000, z/1000]` m와 yaw 0/90으로 변환합니다. `reference_box(box)`는 크기만 mm→m로 변환하고 kg와 N은 유지합니다. 이미 m인 값에 이 변환기를 적용하면 안 됩니다.

참고 구현의 2–5 방향, 다중 지지 구조, 기하적으로 재구성할 수 없는 지지 ID를 그대로 이식하는 어댑터는 아닙니다. 외부에서 관측·재고·지지 체인을 계약에 맞춰 만들고, 이 코어의 보수적인 제약을 별도로 적용해야 합니다. 로봇 도달 검사로 미리 제거된 후보만 넘기면 Teacher의 범위도 그 후보 집합으로 한정됩니다. 학습용 packing 후보 전체와 runtime 로봇 검증을 분리하는 회의 원칙을 권장합니다.

## 표시와 계산 예산

`ahead-planner visualize --input request.json --output preview.svg`는 Top view와 등각 3D 보기를 생성합니다. 외부 시뮬레이터에는 `visualize.scene_data(observation, result)`의 팔레트/박스 3D 형상을 전달할 수 있습니다. 로봇 궤적이나 동역학 장면은 포함하지 않습니다.

`decision_budget_ms`가 지나면 다음 상세 후보 Rollout을 시작하지 않습니다. 완료된 후보 중 선택하고, 아직 완료된 Rollout이 없으면 Ranking/정적 점수로 선택한 사실을 `selection_source`에 표시합니다. 진행 중인 한 Rollout과 전체 안전 검사는 끝내므로 소프트 예산입니다. 학습용 `teacher` 모드는 모든 유효 후보를 평가하기 위해 이 중단을 적용하지 않습니다. Top-K·표본 수·Horizon은 기존 설정으로도 조절할 수 있습니다.

상부 하중 기준이나 선반의 실제 용량을 확보하기 전에는 개발 기본값을 실제 설비 규격으로 간주하지 마세요. 설정은 JSON으로 전달하고 안전 기준의 변경은 기계·물성 검증과 함께 관리합니다.
