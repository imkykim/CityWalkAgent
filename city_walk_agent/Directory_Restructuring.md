# Claude Code Task: Full Directory Restructuring

## 목표 구조 (Before → After)

```
Before                              After
──────────────────────────────────────────────────────
src/
├── agent/
│   ├── walking_agent.py        →   orchestrator.py  (class: CityWalkAgent)
│   ├── cognitive_controller.py →   system1/cognitive_controller.py
│   ├── capabilities/           →   memory/
│   │   ├── short_term_memory.py
│   │   ├── long_term_memory.py
│   │   └── memory_manager.py
│   ├── system2/                    (유지)
│   └── base.py / config/          (유지)
│
├── analysis/
│   ├── continuous_analyzer.py  →   agent/system1/continuous_analyzer.py
│   ├── sequential_analyzer.py  →   research/sequential_analyzer.py
│   ├── aggregator.py           →   research/aggregator.py
│   ├── comparator.py           →   research/comparator.py
│   └── metrics.py              →   research/metrics.py
│
├── config/                     →   core/
│   ├── settings.py             →   core/settings.py
│   ├── constants.py            →   core/constants.py
│   └── frameworks.py           →   core/frameworks/manager.py
│   └── framework_configs/      →   core/frameworks/configs/
│
└── evaluation/                 →   core/evaluation/
    ├── evaluator.py
    ├── vlm_client.py
    ├── prompt_builder.py
    ├── persona_prompt_builder.py
    ├── response_parser.py
    └── batch_processor.py
```

---

## Step 1: 파일 이동

모든 파일을 내용 수정 없이 이동. 새 디렉토리는 생성 필요.

```bash
# 1-A. capabilities → memory
mv src/agent/capabilities/short_term_memory.py  src/agent/memory/short_term_memory.py
mv src/agent/capabilities/long_term_memory.py   src/agent/memory/long_term_memory.py
mv src/agent/capabilities/memory_manager.py     src/agent/memory/memory_manager.py

# 1-B. analysis 분리
mv src/analysis/continuous_analyzer.py   src/agent/system1/continuous_analyzer.py
mv src/analysis/sequential_analyzer.py  src/research/sequential_analyzer.py
mv src/analysis/aggregator.py           src/research/aggregator.py
mv src/analysis/comparator.py           src/research/comparator.py
mv src/analysis/metrics.py              src/research/metrics.py

# 1-C. cognitive_controller → system1
mv src/agent/cognitive_controller.py    src/agent/system1/cognitive_controller.py

# 1-D. evaluation → core/evaluation
mv src/evaluation/evaluator.py              src/core/evaluation/evaluator.py
mv src/evaluation/vlm_client.py             src/core/evaluation/vlm_client.py
mv src/evaluation/prompt_builder.py         src/core/evaluation/prompt_builder.py
mv src/evaluation/persona_prompt_builder.py src/core/evaluation/persona_prompt_builder.py
mv src/evaluation/response_parser.py        src/core/evaluation/response_parser.py
mv src/evaluation/batch_processor.py        src/core/evaluation/batch_processor.py

# 1-E. config → core
mv src/config/settings.py    src/core/settings.py
mv src/config/constants.py   src/core/constants.py
mv src/config/frameworks.py  src/core/frameworks/manager.py
mv src/config/framework_configs/  src/core/frameworks/configs/

# 1-F. walking_agent → orchestrator
mv src/agent/walking_agent.py  src/agent/orchestrator.py

# 1-G. 빈 디렉토리 및 구 파일 삭제
rm -rf src/agent/capabilities/
rm -rf src/analysis/
rm -rf src/evaluation/
rm -rf src/config/
```

---

## Step 2: `__init__.py` 파일 전부 새로 작성

### `src/agent/system1/__init__.py`

```python
"""System 1: Fast, continuous VLM perception."""

from src.agent.system1.continuous_analyzer import ContinuousAnalyzer, WaypointAnalysis
from src.agent.system1.cognitive_controller import CognitiveController

__all__ = ["ContinuousAnalyzer", "WaypointAnalysis", "CognitiveController"]
```

### `src/agent/memory/__init__.py`

```python
"""Memory systems: STM, LTM, MemoryManager."""

from src.agent.memory.short_term_memory import ShortTermMemory, MemoryItem
from src.agent.memory.long_term_memory import (
    LongTermMemory, KeyMoment, RoutePattern, RouteSummary,
)
from src.agent.memory.memory_manager import MemoryManager

__all__ = [
    "ShortTermMemory", "MemoryItem",
    "LongTermMemory", "KeyMoment", "RoutePattern", "RouteSummary",
    "MemoryManager",
]
```

### `src/core/__init__.py`

```python
"""Core infrastructure shared by System 1 and System 2."""

from src.core.constants import (
    DEFAULT_FRAMEWORK_ID,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_SAMPLING_INTERVAL,
    DEFAULT_VLM_TIMEOUT,
    DEFAULT_PHASH_THRESHOLD,
    DEFAULT_ENABLE_MULTI_IMAGE,
    DEFAULT_CONTEXT_WINDOW,
    MAX_IMAGE_SIZE_MB,
    MAX_SCORE, MIN_SCORE,
    MIN_SAMPLING_INTERVAL,
    OPTIMAL_BATCH_SIZE,
    SCORE_DECIMAL_PLACES,
)
from src.core.frameworks.manager import (
    FrameworkManager,
    get_framework_manager,
    load_framework,
    list_frameworks,
)
from src.core.settings import settings

__all__ = [
    "DEFAULT_FRAMEWORK_ID", "DEFAULT_MAX_CONCURRENT", "DEFAULT_RETRY_ATTEMPTS",
    "DEFAULT_SAMPLING_INTERVAL", "DEFAULT_VLM_TIMEOUT", "DEFAULT_PHASH_THRESHOLD",
    "DEFAULT_ENABLE_MULTI_IMAGE", "DEFAULT_CONTEXT_WINDOW",
    "MAX_IMAGE_SIZE_MB", "MAX_SCORE", "MIN_SCORE",
    "MIN_SAMPLING_INTERVAL", "OPTIMAL_BATCH_SIZE", "SCORE_DECIMAL_PLACES",
    "FrameworkManager", "get_framework_manager", "load_framework", "list_frameworks",
    "settings",
]
```

### `src/core/frameworks/__init__.py`

```python
"""Framework loading and management."""

from src.core.frameworks.manager import (
    FrameworkManager,
    get_framework_manager,
    load_framework,
    list_frameworks,
)

__all__ = ["FrameworkManager", "get_framework_manager", "load_framework", "list_frameworks"]
```

### `src/core/evaluation/__init__.py`

```python
"""VLM evaluation components."""

from src.core.evaluation.evaluator import Evaluator, DualEvaluationResult
from src.core.evaluation.vlm_client import VLMClient, VLMConfig
from src.core.evaluation.prompt_builder import PromptBuilder
from src.core.evaluation.persona_prompt_builder import PersonaPromptBuilder

__all__ = [
    "Evaluator", "DualEvaluationResult",
    "VLMClient", "VLMConfig",
    "PromptBuilder", "PersonaPromptBuilder",
]
```

### `src/research/__init__.py`

```python
"""Research tools for method comparison and paper validation.

Not part of the main pipeline — used in experiments/ scripts.
"""

from src.research.sequential_analyzer import SequentialAnalyzer
from src.research.aggregator import AggregateAnalyzer
from src.research.comparator import MethodComparator
from src.research.metrics import (
    calculate_volatility,
    detect_hidden_barriers,
    analyze_transitions,
    Barrier,
    TransitionAnalysis,
)

__all__ = [
    "SequentialAnalyzer", "AggregateAnalyzer", "MethodComparator",
    "calculate_volatility", "detect_hidden_barriers", "analyze_transitions",
    "Barrier", "TransitionAnalysis",
]
```

---

## Step 3: `src/core/settings.py` 경로 수정

`frameworks_dir` property:

```python
# Before
return self.project_root / "src" / "config" / "framework_configs"

# After
return self.project_root / "src" / "core" / "frameworks" / "configs"
```

`src/core/frameworks/manager.py` 내 fallback 경로:

```python
# Before
return Path(__file__).parent / "framework_configs"

# After
return Path(__file__).parent / "configs"
```

---

## Step 4: `orchestrator.py` — 클래스명 변경

```python
# Before
class WalkingAgent(BaseAgent):

# After
class CityWalkAgent(BaseAgent):
```

docstring, 내부 self-reference, `from_preset()` return type hint 등
파일 내 모든 `WalkingAgent` → `CityWalkAgent` 교체.

---

## Step 5: Import 전파

아래 패턴을 프로젝트 전체에서 일괄 교체.

### 교체 패턴 표

| Before                                             | After                                                    |
| -------------------------------------------------- | -------------------------------------------------------- |
| `from src.config import ...`                       | `from src.core import ...`                               |
| `from src.config.settings import settings`         | `from src.core.settings import settings`                 |
| `from src.config.frameworks import load_framework` | `from src.core.frameworks.manager import load_framework` |
| `from src.config import DEFAULT_MAX_CONCURRENT`    | `from src.core.constants import DEFAULT_MAX_CONCURRENT`  |
| `from src.evaluation.evaluator import Evaluator`   | `from src.core.evaluation.evaluator import Evaluator`    |
| `from src.evaluation.vlm_client import VLMConfig`  | `from src.core.evaluation.vlm_client import VLMConfig`   |
| `from src.evaluation import ...`                   | `from src.core.evaluation import ...`                    |
| `from src.analysis import ContinuousAnalyzer`      | `from src.agent.system1 import ContinuousAnalyzer`       |
| `from src.analysis.continuous_analyzer import ...` | `from src.agent.system1.continuous_analyzer import ...`  |
| `from src.analysis import SequentialAnalyzer`      | `from src.research import SequentialAnalyzer`            |
| `from src.analysis.metrics import ...`             | `from src.research.metrics import ...`                   |
| `from src.analysis.sequential_analyzer import ...` | `from src.research.sequential_analyzer import ...`       |
| `from src.agent.cognitive_controller import ...`   | `from src.agent.system1 import ...`                      |
| `from src.agent.capabilities import ...`           | `from src.agent.memory import ...`                       |
| `from src.agent.walking_agent import WalkingAgent` | `from src.agent.orchestrator import CityWalkAgent`       |
| `WalkingAgent` (클래스 사용처)                     | `CityWalkAgent`                                          |

### 파일별 수정 목록

**`src/agent/orchestrator.py`:**

- `from src.agent.capabilities import (...)` → `from src.agent.memory import (...)`
- `from src.agent.cognitive_controller import CognitiveController` → `from src.agent.system1 import CognitiveController`
- `from src.config import DEFAULT_FRAMEWORK_ID, settings` → `from src.core import DEFAULT_FRAMEWORK_ID, settings`
- lazy import 내부: `from src.analysis import ContinuousAnalyzer` → `from src.agent.system1 import ContinuousAnalyzer`
- lazy import 내부: `from src.agent.system2 import Interpreter` 등 (변경 없음)

**`src/agent/__init__.py`:**

- `from src.agent.walking_agent import WalkingAgent` → `from src.agent.orchestrator import CityWalkAgent`
- `from src.agent.capabilities import (...)` → `from src.agent.memory import (...)`
- `__all__` 에서 `WalkingAgent` → `CityWalkAgent`

**`src/agent/memory/memory_manager.py`:**

- `from src.agent.capabilities.long_term_memory import ...` → `from src.agent.memory.long_term_memory import ...`
- `from src.agent.capabilities.short_term_memory import ...` → `from src.agent.memory.short_term_memory import ...`

**`src/agent/system1/continuous_analyzer.py`:**

- `from src.evaluation.evaluator import Evaluator` → `from src.core.evaluation.evaluator import Evaluator`
- `from src.evaluation.vlm_client import VLMConfig` → `from src.core.evaluation.vlm_client import VLMConfig`
- `from src.config import settings, load_framework` → `from src.core import settings, load_framework`

**`src/agent/system2/persona_reasoner.py`:**

- `from src.config import DEFAULT_FRAMEWORK_ID, settings` → `from src.core import DEFAULT_FRAMEWORK_ID, settings`

**`src/agent/system2/` (interpreter, decider, planner, reporter):**

- `from src.config import ...` → `from src.core import ...` (해당 파일마다)

**`src/core/evaluation/evaluator.py`:**

- `from src.config import DEFAULT_MAX_CONCURRENT` → `from src.core.constants import DEFAULT_MAX_CONCURRENT`

**`src/core/evaluation/` (나머지 파일들):**

- `from src.config import ...` → `from src.core import ...`

**`src/research/comparator.py`:**

- `from .sequential_analyzer import ...` (상대 경로 — 변경 없음)
- `from .aggregator import ...` (상대 경로 — 변경 없음)

**`src/validation/` 파일들:**

- `from src.config.settings import settings` → `from src.core.settings import settings`
- `from src.config import settings` → `from src.core import settings`

**`src/data_collection/` 파일들:**

- `from src.config import settings` → `from src.core import settings`

**`main.py`:**

- `from src.config import DEFAULT_FRAMEWORK_ID, settings` → `from src.core import DEFAULT_FRAMEWORK_ID, settings`
- `from src.agent.walking_agent import WalkingAgent` → `from src.agent.orchestrator import CityWalkAgent`
- `WalkingAgent(` → `CityWalkAgent(`

**`validate.py`:**

- `from src.config.settings import settings` → `from src.core.settings import settings`

**`tests/` 전체:**

- 위 패턴 동일 적용
- `WalkingAgent` → `CityWalkAgent`
- `from src.agent.capabilities import ...` → `from src.agent.memory import ...`
- `from src.analysis import ...` → 새 경로로

**`experiments/` 스크립트 전체:**

- `from src.analysis import SequentialAnalyzer, MethodComparator` → `from src.research import ...`
- `from src.config import ...` → `from src.core import ...`
- `from src.evaluation import ...` → `from src.core.evaluation import ...`

**`tests/test_framework.py`:**

- `from analysis.metrics import ...` → `from src.research.metrics import ...`
- `from analysis import SequentialAnalyzer, MethodComparator` → `from src.research import ...`
- `from evaluation.prompt_builder import PromptBuilder` → `from src.core.evaluation.prompt_builder import PromptBuilder`

---

## Step 6: 검증

```bash
# 최종 구조 확인
find src -type d | sort

# 핵심 import 검증
python -c "
from src.agent.orchestrator import CityWalkAgent
from src.agent.system1 import ContinuousAnalyzer, CognitiveController, WaypointAnalysis
from src.agent.system2 import PersonaReasoner, Interpreter, Decider, Planner, Reporter
from src.agent.memory import MemoryManager, ShortTermMemory, LongTermMemory
from src.core import settings, load_framework, DEFAULT_FRAMEWORK_ID
from src.core.evaluation import Evaluator, VLMConfig
from src.research import SequentialAnalyzer, MethodComparator
print('All imports OK')
"

# 전체 테스트
pytest tests/ -v --tb=short

# main.py 실행
python main.py --help
```

---

## Definition of Done

- [ ] `src/agent/memory/` 존재, `src/agent/capabilities/` 삭제됨
- [ ] `src/agent/system1/` 존재 (continuous_analyzer + cognitive_controller)
- [ ] `src/agent/orchestrator.py` 존재, 클래스명 `CityWalkAgent`
- [ ] `src/research/` 존재 (sequential_analyzer, aggregator, comparator, metrics)
- [ ] `src/core/` 존재 (evaluation/, frameworks/, settings.py, constants.py)
- [ ] `src/config/`, `src/evaluation/`, `src/analysis/`, `src/agent/walking_agent.py` 삭제됨
- [ ] `settings.py`의 `frameworks_dir` → `src/core/frameworks/configs/`
- [ ] 프로젝트 전체에 `WalkingAgent` 참조 0개
- [ ] `pytest tests/ -v` 통과
- [ ] `python main.py --help` 정상 실행
