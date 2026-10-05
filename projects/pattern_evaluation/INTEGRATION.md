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

응답의 `requires_robot_validation=true`, `robot_feasibility=NOT_CHECKED`는 이 평가기가 로봇 작업을 수행하지 않았다는 뜻입니다. 선반 슬롯의 실제 치수·가반하중·경로도 제어기가 검증합니다. 순차 재요청 시 버퍼 이동 횟수를 보존해야 한 번 제한이 유지됩니다.

기존 `PackingEnv`에서 버퍼 없는 정책을 실행할 때는 `to_packing_env_action(result)`로 `PLACE / REMEASURE / ROUTE_DAMAGE / STOP`에 변환합니다. 이 환경이 지원하지 않는 선반 행동은 변환 함수가 오류로 알립니다.

## 반환 점수

`static`에는 공간·안전·하중·균형·시간 대용 지표가, `future`에는 평균 적재 비율·추가 부피·CVaR·최악 표본·차단 비율이 포함됩니다. `diagnostics`는 후보 수, 상세 Rollout 수, 마스크 거절, 모델 사용 상태와 OOD 상태를 제공합니다. `ranking` 모드는 상세 Rollout을 하지 않으므로 `future=null`입니다.

상부 하중 기준이나 선반의 실제 용량을 확보하기 전에는 개발 기본값을 실제 설비 규격으로 간주하지 마세요. 설정은 JSON으로 전달하고 안전 기준의 변경은 기계·물성 검증과 함께 관리합니다.
