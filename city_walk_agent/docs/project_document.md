# CityWalkAgent — Project Documentation

Version: 0.2.0 | 2026-03-03

---

## 1. 프로젝트 개요

CityWalkAgent(StreetAgent)는 VLM 기반 도시 보행 경험 분석 플랫폼.
Google Street View 이미지를 5~10미터 간격으로 연속 처리하여 보행자 관점의 도시 환경 품질을 정량화한다.

### 핵심 기여

- **연속 경험 분석**: 지점 평균이 아닌 경로 전체 흐름 분석. 동일 평균 점수도 변동성/전환 구간에 따라 다른 경험
- **Dual-System 인지 아키텍처**: System 1(ContinuousAnalyzer) 전수 빠른 평가 + System 2(PersonaReasoner) 선택적 깊은 추론
- **Persona-aware 평가**: 사용자 유형(부모, 러너, 사진작가 등)에 따라 동일 공간도 다른 평가
- **MIT Place Pulse 2.0 검증**: CLIP + K-NN으로 인간 지각 데이터와 Spearman ρ 0.57~0.85 달성

---

## 2. 아키텍처 개요

```
Waypoint Image
      │
      ├── CognitiveController ── pHash, visual change 감지
      │
      ├── ContinuousAnalyzer ── System 1 (모든 waypoint, 빠른 VLM 평가)
      │   ├── Text-Context VLM (single image, 기본)
      │   └── Multi-Image VLM (visual change 시)
      │
      ├── PersonaReasoner ──── System 2 (선택적 트리거, 깊은 추론)
      │   ├── Interpret: 왜 이 점수인가?
      │   ├── Decide: 어떤 행동이 필요한가?
      │   ├── Plan: 이후 경로 계획
      │   └── Report: 사용자 메시지
      │
      └── MemoryManager
          ├── ShortTermMemory (최근 5~10 waypoint)
          └── LongTermMemory (Procedure / Semantic / Episodic / Disposition)
```

**핵심 수정 (Phase 2+3 인터리브)**:

```
for each waypoint:
  1. CognitiveController → visual change 감지
  2. ContinuousAnalyzer → VLM 평가 (STM 읽기)
  3. MemoryManager      → STM 즉시 쓰기  ← 핵심
  4. PersonaReasoner    → 추론 (트리거 시)
```

→ 이전: Phase 2 전수 완료 후 Phase 3. STM이 항상 비어있었음.
→ 이후: 각 waypoint 직후 STM 갱신. waypoint 1부터 context 전달.

---

## 3. 컴포넌트 상세

### 3.1 WalkingAgent (`src/agent/walking_agent.py`)

파이프라인 오케스트레이터.

메서드:

- `run_with_memory(start, end, ...)` — 좌표로 분석
- `run_with_memory_from_folder(route_folder, ...)` — 기존 이미지 폴더로 분석
- `from_preset(preset_name, framework_id)` — 프리셋으로 에이전트 생성

속성 (lazy-load):

- `continuous_analyzer`, `persona_reasoner`, `memory_manager`, `cognitive`

---

### 3.2 ContinuousAnalyzer (`src/analysis/continuous_analyzer.py`)

System 1. 모든 waypoint VLM 평가.

평가 모드:
| 모드 | 조건 | 비율 |
|------|------|------|
| Single-image + text context | 기본 | ~85-90% |
| Multi-image comparison | pHash 트리거 | ~10-15% |

출력: `WaypointAnalysis`

```python
scores: Dict[str, float]          # objective
persona_scores: Dict[str, float]  # persona-aware
visual_change_detected: bool
phash_distance: float
```

---

### 3.3 PersonaReasoner (`src/agent/capabilities/persona_reasoner.py`)

System 2. 선택적 트리거, 깊은 추론.
**현재 상태: Skeleton (heuristic placeholder, TODO: LLM 구현)**

트리거 조건 (`TriggerReason`):

- `VISUAL_CHANGE`, `SCORE_VOLATILITY`, `DISTANCE_MILESTONE`
- `EXCEPTIONAL_MOMENT`, `DECISION_POINT` (예정)

주요 메서드: `reason()` → `_interpret()` → `_decide()` → `_plan()` → `_report()`

출력: `ReasoningResult`

```python
interpretation: str          # Interpret 결과
significance: str            # "high"|"medium"|"low"
avoid_recommendation: bool   # 우회 권장
prediction: Optional[str]    # 이후 예측
recommendation: Optional[str] # 사용자 메시지
confidence: float
```

Note: **System 1 scores는 Final. PersonaReasoner는 점수를 수정하지 않음.**

---

### 3.4 MemoryManager (`src/agent/capabilities/memory_manager.py`)

STM/LTM 통합 허브.

주요 메서드:

- `process_waypoint(analysis, triggered, trigger_reason)` → STM + context 반환
- `update_with_system2_result(waypoint_id, reasoning_result)` → STM 갱신
- `complete_route(...)` → LTM consolidation

Attention Gate: STM 진입 전 ~50% 필터링

---

### 3.5 ShortTermMemory / LongTermMemory

STM: 슬라이딩 윈도우 (기본 5개)
LTM: JSONL 기반 영구 저장 (KeyMoment, RoutePattern, RouteSummary)
LTM 서브타입: Procedure / Semantic / Episodic / Disposition

---

## 4. Evaluation Framework 시스템

| Framework ID       | 차원 수 | 이론                     |
| ------------------ | ------- | ------------------------ |
| `place_pulse_2.0`  | 4D      | MIT Place Pulse (기본값) |
| `sagai_2025`       | 4D      | SAGAI                    |
| `streetagent_5d`   | 5D      | StreetAgent 자체         |
| `ewing_handy_5d`   | 5D      | Ewing & Handy            |
| `kaplan_4d`        | 4D      | Kaplan & Kaplan          |
| `phenomenology_3d` | 3D      | 현상학                   |

Framework-Agnostic 설계: `framework_id` 하나로 전체 파이프라인 동작.

---

## 5. Persona 시스템

| Preset      | 이름               | 특징                    |
| ----------- | ------------------ | ----------------------- |
| `safety`    | Safety Guardian    | functional_quality 2.2× |
| `scenic`    | Aesthetic Explorer | sensory_complexity 2.0× |
| `balanced`  | Balanced Navigator | 전 차원 1.0×            |
| `comfort`   | Comfort Seeker     | functional_quality 2.0× |
| `explorer`  | Urban Explorer     | spatial_sequence 2.0×   |
| `technical` | Technical Analyst  | 전 차원 1.0×            |

Dual Evaluation: 각 waypoint마다 objective + persona-aware 점수 동시 생성.

---

## 6. 검증 파이프라인 (CLIP + K-NN)

방법: CLIP 임베딩 → FAISS K-NN으로 Place Pulse 유사 이미지 검색 → KNN 예측값과 VLM 점수 Spearman 상관 계산

결과: ρ **0.57~0.85** (차원별), p < 0.001

```bash
python validate.py --input data/results/analysis_results.json --use-cache
```

---

## 7. 디렉토리 구조

```
city_walk_agent/
├── src/
│   ├── agent/
│   │   ├── walking_agent.py
│   │   ├── cognitive_controller.py
│   │   └── capabilities/
│   │       ├── persona_reasoner.py    # System 2 (신규)
│   │       ├── thinking.py            # Deprecated shim
│   │       ├── memory_manager.py
│   │       ├── short_term_memory.py
│   │       └── long_term_memory.py
│   ├── analysis/
│   │   └── continuous_analyzer.py     # System 1
│   ├── config/
│   │   ├── settings.py
│   │   ├── frameworks.py
│   │   └── framework_configs/         # Framework JSON들
│   ├── evaluation/
│   │   └── persona_prompt_builder.py
│   ├── validation/                    # CLIP + K-NN 검증
│   └── utils/
│       ├── visualization.py           # RouteVisualizer
│       └── data_models.py
├── experiments/
├── tests/
├── data/
│   ├── images/
│   ├── results/
│   └── place_pulse/
└── validate.py
```

---

## 8. 기술 스택

| 카테고리    | 기술                      |
| ----------- | ------------------------- |
| VLM         | Qwen VLM (qwen-vl-max)    |
| 이미지 수집 | ZenSVI (GSV + Mapillary)  |
| 검증 임베딩 | CLIP (ViT-B/32, ViT-L/14) |
| K-NN        | FAISS                     |
| 설정        | Pydantic Settings         |
| 로깅        | structlog                 |
| 테스트      | pytest                    |

---

## 9. 설계 원칙

1. **Framework-Agnostic**: framework_id만으로 전체 체계 교체
2. **Lazy Loading**: 고비용 컴포넌트는 사용 시 초기화
3. **Interleaved Pipeline**: STM 즉시 갱신으로 올바른 컨텍스트 전달
4. **Persona as Lens**: 점수 수정이 아닌 해석의 차이
5. **System 1 Scores are Final**: PersonaReasoner는 해석/판단만 수행

---

## 10. 현재 개발 상태

| 컴포넌트                      | 상태             |
| ----------------------------- | ---------------- |
| ContinuousAnalyzer (System 1) | ✅ 완성          |
| CognitiveController           | ✅ 완성          |
| MemoryManager + STM/LTM       | ✅ 완성          |
| CLIP + K-NN 검증              | ✅ 완성          |
| PersonaReasoner skeleton      | ✅ 완성          |
| PersonaReasoner LLM 구현      | ⏳ 미구현 (TODO) |
