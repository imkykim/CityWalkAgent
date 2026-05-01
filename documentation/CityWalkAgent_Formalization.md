# CityWalkAgent — Method Formalization (수식 최종본)

> 각 Definition 아래에 **KaTeX/LaTeX 소스**를 코드블록으로 함께 제공합니다.
> 노션에서는 `/math` 블록에, LaTeX 논문에서는 `equation` 환경에 직접 붙여넣으면 됩니다.

---

## 1. Problem Statement

### Definition 1 — Sequential Pedestrian Environment Evaluation

보행 환경 평가를 **순차적 다차원 인지 문제**로 정의한다.
각 시간 $t$에서 에이전트는 현재 이미지, 페르소나, 누적 기억, 평가 프레임워크를 기반으로 환경 품질 점수 벡터를 생성한다:

$$
s_t = \Psi\bigl(I_t,\;P,\;\mathcal{M}_t,\;F\bigr)
$$

경로 $R$의 최종 출력은 개별 점수의 **순서 시퀀스**이다:

$$
\mathbf{S}_R = \bigl(s_1,\;s_2,\;\ldots,\;s_T\bigr)
$$

여기서:

| 기호                                              | 의미                                                                   |
| ------------------------------------------------- | ---------------------------------------------------------------------- |
| $I_t \in \mathbb{R}^{H \times W \times 3}$        | 시간 $t$의 Street View 이미지                                          |
| $P \in \mathcal{P}$                               | 페르소나 (homebuyer, parent, photographer, tourist, runner)            |
| $\mathcal{M}_t = (\mathrm{STM}_t,\;\mathrm{LTM})$ | 시간 $t$까지의 누적 공간 기억 (단기 + 장기)                            |
| $F = \{f_1, f_2, \ldots, f_D\}$                   | 평가 프레임워크의 $D$개 차원 (e.g. Safety, Lively, Beautiful, Wealthy) |
| $s_t \in [1, 10]^D$                               | 시간 $t$에서의 차원별 점수 벡터                                        |
| $\mathbf{S}_R$                                    | 경로 전체의 점수 시퀀스                                                |
| $\Psi$                                            | 에이전트의 인지 정책 (Dual-Process architecture로 구현)                |

출력이 단일 스칼라가 아닌 **점수 시퀀스** $\mathbf{S}_R$라는 점이 기존 point-aggregation 방식과의 핵심 차이이다.
동일한 평균 점수를 가진 두 경로도 $\mathbf{S}_R$의 변동성(volatility)과 전환 구간(transition)에 따라 전혀 다른 보행 경험을 나타낸다.

```latex
s_t = \Psi\bigl(I_t,\;P,\;\mathcal{M}_t,\;F\bigr)
```

```latex
\mathbf{S}_R = \bigl(s_1,\;s_2,\;\ldots,\;s_T\bigr)
```

```latex
\mathcal{M}_t = (\mathrm{STM}_t,\;\mathrm{LTM})
```

---

## 2. Dual-Process Cognitive Architecture

### Definition 2 — Cognitive Gating Function

Kahneman의 이중 처리 이론에 기반하여, 각 waypoint $t$에서 에이전트는 먼저 **gating function** $G$로 분석 여부를 결정한다.

$$
G(t) =
\begin{cases}
1 & \text{if } t = 0 \quad \text{(first step)} \\
1 & \text{if } |L_t| \geq 3 \quad \text{(intersection)} \\
1 & \text{if } \Delta\theta_t > \tau_\theta \quad \text{(heading change)} \\
1 & \text{if } d_{\mathrm{geo}}(p_t, p_{\hat{t}}) > \tau_d \quad \text{(distance)} \\
1 & \text{if } d_{\mathrm{pHash}}(I_t, I_{\hat{t}}) > \tau_p \quad \text{(visual change)} \\
0 & \text{otherwise (skip)}
\end{cases}
$$

여기서:

| 기호                 | 의미                                                                                | 기본값       |
| -------------------- | ----------------------------------------------------------------------------------- | ------------ |
| $\lvert L_t \rvert$  | 현재 pano의 navigable link 수                                                       | —            |
| $\hat{t}$            | 마지막으로 분석이 수행된 시점                                                       | —            |
| $\Delta\theta_t$     | heading 변화량: $\lvert((\theta_t - \theta_{\hat{t}}) + 180) \bmod 360 - 180\rvert$ | —            |
| $d_{\mathrm{geo}}$   | 측지 거리 (geodesic distance, meters)                                               | —            |
| $d_{\mathrm{pHash}}$ | perceptual hash의 Hamming distance                                                  | —            |
| $\tau_\theta$        | heading 변화 임계값                                                                 | $30°$        |
| $\tau_d$             | 거리 임계값                                                                         | $30\text{m}$ |
| $\tau_p$             | pHash 임계값                                                                        | $30$         |

$G(t) = 0$이면 VLM 호출 없이 단순 navigation만 수행한다 (skip).
신호들은 **계단적으로 평가**된다: heading → distance → pHash 순으로, 먼저 충족되는 신호가 분석을 트리거한다.

```latex
G(t) =
\begin{cases}
1 & \text{if } t = 0 \\
1 & \text{if } |L_t| \geq 3 \\
1 & \text{if } \Delta\theta_t > \tau_\theta \\
1 & \text{if } d_{\mathrm{geo}}(p_t, p_{\hat{t}}) > \tau_d \\
1 & \text{if } d_{\mathrm{pHash}}(I_t, I_{\hat{t}}) > \tau_p \\
0 & \text{otherwise}
\end{cases}
```

```latex
\Delta\theta_t = \bigl|((\theta_t - \theta_{\hat{t}}) + 180) \bmod 360 - 180\bigr|
```

---

## 3. System 1: Fast Perception

### Definition 3 — ContinuousAnalyzer

$G(t) = 1$인 모든 waypoint에 대해, VLM이 각 차원을 **독립적으로** 평가한다:

$$
s_t^{f_d} = \mathrm{VLM}\bigl(I_t,\;\pi(f_d, P),\;C_t\bigr), \quad d = 1, \ldots, D
$$

여기서:

| 기호          | 의미                                                              |
| ------------- | ----------------------------------------------------------------- |
| $\pi(f_d, P)$ | 차원 $f_d$와 페르소나 $P$의 interaction으로 생성된 프롬프트       |
| $C_t$         | STM에서 추출한 temporal context (최근 $k$ waypoint의 점수와 추론) |

페르소나는 점수를 사후 변환하는 것이 아니라, VLM 호출 시점에 **해석 렌즈(interpretive lens)**로 프롬프트에 내장된다.

```latex
s_t^{f_d} = \mathrm{VLM}\bigl(I_t,\;\pi(f_d, P),\;C_t\bigr), \quad d = 1, \ldots, D
```

---

## 4. Short-Term Memory (STM)

### Definition 4 — Sliding Window Memory

STM은 sliding window로 최근 $k$개 분석 결과를 유지하며, 각 waypoint 분석 직후 **즉시 갱신**된다:

$$
\mathrm{STM}_t = \Bigl\{\bigl(s_i,\;r_i,\;I_i\bigr) \;\Big|\; i \in \bigl[\max(1,\;t - k + 1),\;\;t\bigr]\Bigr\}
$$

여기서 $r_i = \{r_i^{f_d}\}_{d=1}^{D}$는 각 차원의 VLM reasoning text이다.

STM에서 추출되는 context $C_t$는 다음 waypoint $t+1$의 System 1 평가에 입력된다:

$$
C_t = \mathrm{extract}(\mathrm{STM}_t) = \Bigl(\{s_i\}_{i=t-k+1}^{t},\;\{\bar{s}^{f_d}\}_{d=1}^{D},\;\sigma_{\mathrm{recent}}\Bigr)
$$

여기서 $\bar{s}^{f_d}$는 window 내 차원별 평균, $\sigma_{\mathrm{recent}}$는 최근 점수 표준편차(volatility indicator)이다.

```latex
\mathrm{STM}_t = \Bigl\{\bigl(s_i,\;r_i,\;I_i\bigr) \;\Big|\; i \in \bigl[\max(1,\;t{-}k{+}1),\;\;t\bigr]\Bigr\}
```

```latex
C_t = \bigl(\{s_i\}_{i=t-k+1}^{t},\;\{\bar{s}^{f_d}\}_{d=1}^{D},\;\sigma_{\mathrm{recent}}\bigr)
```

---

## 5. System 2: Deep Reasoning

### Definition 5 — Selective Trigger

System 2는 모든 waypoint에서 활성화되지 않고, 조건부로 트리거된다:

$$
\mathrm{Trigger}(t) = \mathbb{1}[\text{visual\_change}] \;\lor\; \mathbb{1}[\text{score\_volatility}] \;\lor\; \mathbb{1}[\text{intersection}] \;\lor\; \mathbb{1}[\text{distance\_milestone}]
$$

목표 트리거 비율: 분석된 waypoint의 **15–25%**.

```latex
\mathrm{Trigger}(t) = \mathbb{1}[\text{visual\_change}] \;\lor\; \mathbb{1}[\text{score\_volatility}] \;\lor\; \mathbb{1}[\text{intersection}] \;\lor\; \mathbb{1}[\text{distance\_milestone}]
```

### Definition 6 — Reasoning Chain (PersonaReasoner)

트리거 시 **4단계 순차 추론 파이프라인**을 수행한다. 각 단계의 출력이 다음 단계의 입력에 누적된다:

$$
\mathrm{S2}(t) = \mathrm{Report} \;\circ\; \mathrm{Plan} \;\circ\; \mathrm{Decide} \;\circ\; \mathrm{Interpret}\bigl(s_t,\;\mathrm{STM}_t,\;\mathrm{LTM},\;P\bigr)
$$

구체적으로:

$$
\begin{aligned}
\mathcal{I}_t &= \mathrm{Interpret}(s_t,\;r_t,\;\mathrm{STM}_t,\;\mathrm{LTM},\;P) \\
\mathcal{D}_t &= \mathrm{Decide}(\mathcal{I}_t,\;s_t,\;\mathrm{LTM},\;P) \\
\mathcal{P}_t &= \mathrm{Plan}(\mathcal{D}_t,\;\mathrm{STM}_t,\;R_{\mathrm{meta}}) \\
\mathcal{E}_t &= \mathrm{Report}(\mathcal{I}_t,\;\mathcal{D}_t,\;s_t,\;P)
\end{aligned}
$$

| 단계          | 입력                                         | 출력                                       | 역할                           |
| ------------- | -------------------------------------------- | ------------------------------------------ | ------------------------------ |
| **Interpret** | $s_t$, $r_t$, STM, LTM, $P$                  | $\mathcal{I}_t$: 점수 변화 원인, 핵심 우려 | WHY: 왜 이 점수인가?           |
| **Decide**    | $\mathcal{I}_t$, $s_t$, LTM, $P$             | $\mathcal{D}_t$: significance, avoid 여부  | WHAT: 어떤 판단을 내릴 것인가? |
| **Plan**      | $\mathcal{D}_t$, STM, route metadata         | $\mathcal{P}_t$: 예측, 대안 제안           | HOW: 향후 경로는?              |
| **Report**    | $\mathcal{I}_t$, $\mathcal{D}_t$, $s_t$, $P$ | $\mathcal{E}_t$: narrative, episode        | OUT: 사용자 전달 메시지        |

System 2는 **System 1 점수를 수정하지 않는다** — 점수는 S1에서 확정되고, S2는 해석·판단·서사만 생성한다.

```latex
\mathrm{S2}(t) = \mathrm{Report} \;\circ\; \mathrm{Plan} \;\circ\; \mathrm{Decide} \;\circ\; \mathrm{Interpret}\bigl(s_t,\;\mathrm{STM}_t,\;\mathrm{LTM},\;P\bigr)
```

```latex
\begin{aligned}
\mathcal{I}_t &= \mathrm{Interpret}(s_t,\;r_t,\;\mathrm{STM}_t,\;\mathrm{LTM},\;P) \\
\mathcal{D}_t &= \mathrm{Decide}(\mathcal{I}_t,\;s_t,\;\mathrm{LTM},\;P) \\
\mathcal{P}_t &= \mathrm{Plan}(\mathcal{D}_t,\;\mathrm{STM}_t,\;R_{\mathrm{meta}}) \\
\mathcal{E}_t &= \mathrm{Report}(\mathcal{I}_t,\;\mathcal{D}_t,\;s_t,\;P)
\end{aligned}
```

---

## 6. Persona as Interpretive Lens

### Definition 7 — Persona-Invariant Observation, Persona-Variant Interpretation

모든 페르소나는 동일한 이미지와 동일한 프레임워크를 사용하되, 프롬프트 $\pi_P$가 **해석 렌즈**로 작용한다:

$$
\forall\;P_1, P_2 \in \mathcal{P}:\quad I_t^{P_1} = I_t^{P_2},\quad F^{P_1} = F^{P_2},\quad \text{but}\quad s_t^{P_1} \neq s_t^{P_2}
$$

점수 차이는 결정론적 가중치 변환(ScoreTransformer)이 아니라, VLM의 **contextual understanding** 에서 자연적으로 발생한다:

$$
s_t^{P} = \mathrm{VLM}\bigl(I_t,\;\pi(f_d, P),\;C_t\bigr) \neq \mathrm{VLM}\bigl(I_t,\;\pi(f_d, P'),\;C_t\bigr) = s_t^{P'}
$$

이는 "동일 환경이 사용자에 따라 다르게 인지된다"는 환경심리학의 핵심 전제를 VLM-native prompting으로 구현한 것이다.

```latex
\forall\;P_1, P_2 \in \mathcal{P}:\quad I_t^{P_1} = I_t^{P_2},\quad F^{P_1} = F^{P_2},\quad \text{but}\quad s_t^{P_1} \neq s_t^{P_2}
```

---

## 7. Navigation Decision at Intersections

### Definition 8 — Branch Decision with Soft Priority

교차로에서 $|L_t| \geq 2$개 후보 방향에 대해, 각 방향의 **lookahead chain**을 병렬 탐색하고 LLM이 최종 결정한다:

$$
d^{*} = \mathrm{Decider}_{\mathrm{LLM}}\Bigl(\bigl\{(\bar{s}_d,\;\mathcal{I}_d,\;\Delta\phi_d)\bigr\}_{d \in L_t},\;\mathrm{LTM},\;P\Bigr)
$$

여기서:

| 기호                   | 의미                                                                      |
| ---------------------- | ------------------------------------------------------------------------- |
| $\bar{s}_d$            | 방향 $d$의 lookahead chain 평균 점수                                      |
| $\mathcal{I}_d$        | 방향 $d$에 대한 Interpreter의 해석                                        |
| $\Delta\phi_d$         | 방향 $d$와 waypoint bearing $\theta_{\mathrm{wp}}$ 간의 angular deviation |
| $\theta_{\mathrm{wp}}$ | Google Directions API에서 제공하는 다음 waypoint 방향                     |

Bearing deviation $\Delta\phi_d$는 **soft priority**로만 제공된다:

$$
\Delta\phi_d = \bigl|((\theta_d - \theta_{\mathrm{wp}}) + 180) \bmod 360 - 180\bigr|
$$

LLM은 환경 품질과 경로 충실도를 **자율적으로 균형** 잡으며 결정한다.
Hard cone filtering 대신 soft priority를 사용함으로써 Decider LLM의 contextual judgment를 보존한다.

```latex
d^{*} = \mathrm{Decider}_{\mathrm{LLM}}\Bigl(\bigl\{(\bar{s}_d,\;\mathcal{I}_d,\;\Delta\phi_d)\bigr\}_{d \in L_t},\;\mathrm{LTM},\;P\Bigr)
```

```latex
\Delta\phi_d = \bigl|((\theta_d - \theta_{\mathrm{wp}}) + 180) \bmod 360 - 180\bigr|
```

---

## 8. Complete Agent Loop

### Definition 9 — Waypoint Processing Pipeline

전체 파이프라인을 하나의 알고리즘으로 요약한다:

$$
\text{For each } t = 1, \ldots, T:
$$

$$
\begin{aligned}
&\textbf{Step 1 (Gate):} & G(t) &= \mathrm{Gate}(t,\;\hat{t},\;L_t,\;\tau_\theta,\;\tau_d,\;\tau_p) \\
&\textbf{Step 2 (Perceive):} & s_t &=
\begin{cases}
\mathrm{VLM}(I_t, \pi(F, P), C_t) & \text{if } G(t) = 1 \\
\varnothing & \text{if } G(t) = 0
\end{cases} \\
&\textbf{Step 3 (Remember):} & \mathrm{STM}_t &\leftarrow \mathrm{STM}_{t-1} \cup \{(s_t, r_t, I_t)\} \\
&\textbf{Step 4 (Reason):} & \mathcal{E}_t &=
\begin{cases}
\mathrm{S2}(s_t, \mathrm{STM}_t, \mathrm{LTM}, P) & \text{if } \mathrm{Trigger}(t) \\
\varnothing & \text{otherwise}
\end{cases} \\
&\textbf{Step 5 (Navigate):} & d^{*} &=
\begin{cases}
\mathrm{Decider}(\ldots) & \text{if } |L_t| \geq 2 \\
L_t[0] & \text{otherwise}
\end{cases}
\end{aligned}
$$

```latex
\begin{aligned}
&\textbf{Step 1 (Gate):} & G(t) &= \mathrm{Gate}(t,\;\hat{t},\;L_t,\;\tau_\theta,\;\tau_d,\;\tau_p) \\
&\textbf{Step 2 (Perceive):} & s_t &=
\begin{cases}
\mathrm{VLM}(I_t, \pi(F, P), C_t) & \text{if } G(t) = 1 \\
\varnothing & \text{if } G(t) = 0
\end{cases} \\
&\textbf{Step 3 (Remember):} & \mathrm{STM}_t &\leftarrow \mathrm{STM}_{t-1} \cup \{(s_t, r_t, I_t)\} \\
&\textbf{Step 4 (Reason):} & \mathcal{E}_t &=
\begin{cases}
\mathrm{S2}(s_t, \mathrm{STM}_t, \mathrm{LTM}, P) & \text{if Trigger}(t) \\
\varnothing & \text{otherwise}
\end{cases} \\
&\textbf{Step 5 (Navigate):} & d^{*} &=
\begin{cases}
\mathrm{Decider}(\ldots) & \text{if } |L_t| \geq 2 \\
L_t[0] & \text{otherwise}
\end{cases}
\end{aligned}
```

---

## 8. Validation

### Definition 9 — CLIP-KNN Alignment with Human Perception

System 1 점수의 인간 지각과의 정합성을 Place Pulse 2.0 데이터셋으로 검증한다:

$$
\rho_{f_d} = \mathrm{Spearman}\bigl(\hat{y}_{f_d}^{\,\mathrm{CLIP\text{-}KNN}},\;\;y_{f_d}^{\,\mathrm{PP2.0}}\bigr)
$$

여기서:

| 기호                                         | 의미                                         |
| -------------------------------------------- | -------------------------------------------- |
| $\hat{y}_{f_d}^{\,\mathrm{CLIP\text{-}KNN}}$ | CLIP embedding + K-NN으로 예측한 차원별 순위 |
| $y_{f_d}^{\,\mathrm{PP2.0}}$                 | MIT Place Pulse 2.0의 1.17M 인간 판단 데이터 |
| $\rho_{f_d}$                                 | 달성된 Spearman 순위 상관 계수 (0.57–0.85)   |

```latex
\rho_{f_d} = \mathrm{Spearman}\bigl(\hat{y}_{f_d}^{\,\mathrm{CLIP\text{-}KNN}},\;\;y_{f_d}^{\,\mathrm{PP2.0}}\bigr)
```

---

## Summary of Notation

| 기호                                                         | 의미                                 |
| ------------------------------------------------------------ | ------------------------------------ |
| $R$                                                          | 경로 (waypoint 시퀀스)               |
| $T$                                                          | 경로의 총 waypoint 수                |
| $I_t$                                                        | 시간 $t$의 Street View 이미지        |
| $p_t$                                                        | 시간 $t$의 GPS 좌표                  |
| $P$                                                          | 페르소나                             |
| $\mathcal{P}$                                                | 페르소나 집합                        |
| $F$                                                          | 평가 프레임워크                      |
| $D$                                                          | 프레임워크의 차원 수                 |
| $f_d$                                                        | $d$-번째 평가 차원                   |
| $s_t$                                                        | 시간 $t$의 점수 벡터 $\in [1,10]^D$  |
| $s_t^{f_d}$                                                  | 시간 $t$, 차원 $f_d$의 점수          |
| $\mathbf{S}_R$                                               | 경로 전체의 점수 시퀀스              |
| $G(t)$                                                       | gating function (0 or 1)             |
| $\hat{t}$                                                    | 마지막 분석 시점                     |
| $\tau_\theta, \tau_d, \tau_p$                                | gating thresholds                    |
| $L_t$                                                        | 시간 $t$의 navigable links           |
| $\pi(f_d, P)$                                                | 차원-페르소나 프롬프트               |
| $C_t$                                                        | STM에서 추출된 temporal context      |
| $\mathrm{STM}_t$                                             | 시간 $t$의 short-term memory         |
| $\mathrm{LTM}$                                               | long-term memory                     |
| $\mathrm{S2}(t)$                                             | System 2 reasoning chain 출력        |
| $\mathcal{I}_t, \mathcal{D}_t, \mathcal{P}_t, \mathcal{E}_t$ | Interpret, Decide, Plan, Report 출력 |
| $d^*$                                                        | 교차로에서 선택된 방향               |
| $\theta_{\mathrm{wp}}$                                       | waypoint bearing (Directions API)    |
| $\rho_{f_d}$                                                 | Spearman 상관 계수                   |
