import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
file_path = "outputs/dual_system_demo/analysis_results.json"
with open(file_path, "r") as f:
    data = json.load(f)

# 2. 데이터 가공 (비교 분석용)
records_long = []  # 박스플롯/라인차트용 (Long Format)
records_delta = []  # 히트맵용 (변화량 계산)

metrics = [
    "spatial_sequence",
    "visual_coherence",
    "sensory_complexity",
    "spatial_legibility",
    "functional_quality",
]

for item in data:
    wp_id = item["waypoint_id"]
    s1 = item["system1_scores"]

    # System 2가 트리거되지 않았거나 점수가 없으면 System 1 점수를 그대로 사용 (변화 없음으로 간주)
    s2 = (
        item["system2_scores"]
        if item.get("system2_triggered") and item["system2_scores"]
        else s1
    )

    # 2-1. Long Format 데이터 생성 (System 1 vs System 2 태깅)
    # 평균 점수 계산
    s1_avg = np.mean(list(s1.values()))
    s2_avg = np.mean(list(s2.values()))

    records_long.append(
        {
            "Waypoint ID": wp_id,
            "System": "System 1",
            "Metric": "Average",
            "Score": s1_avg,
        }
    )
    records_long.append(
        {
            "Waypoint ID": wp_id,
            "System": "System 2",
            "Metric": "Average",
            "Score": s2_avg,
        }
    )

    for m in metrics:
        records_long.append(
            {
                "Waypoint ID": wp_id,
                "System": "System 1",
                "Metric": m,
                "Score": s1.get(m, 0),
            }
        )
        records_long.append(
            {
                "Waypoint ID": wp_id,
                "System": "System 2",
                "Metric": m,
                "Score": s2.get(m, 0),
            }
        )

    # 2-2. Delta(변화량) 데이터 생성 (System 2 - System 1)
    delta_record = {"Waypoint ID": wp_id}
    for m in metrics:
        change = s2.get(m, 0) - s1.get(m, 0)
        delta_record[m] = change
    records_delta.append(delta_record)

# 데이터프레임 변환
df_long = pd.DataFrame(records_long)
df_delta = pd.DataFrame(records_delta).set_index("Waypoint ID")

# 3. 시각화 설정
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (20, 12)
fig = plt.figure()
gs = fig.add_gridspec(2, 2)  # 2x2 그리드 레이아웃

# --- 차트 1: 평균 점수 트렌드 비교 (Line Chart) ---
ax1 = fig.add_subplot(gs[0, 0])
sns.lineplot(
    data=df_long[df_long["Metric"] == "Average"],
    x="Waypoint ID",
    y="Score",
    hue="System",
    style="System",
    markers=True,
    dashes=False,
    palette=["#bdc3c7", "#e74c3c"],
    linewidth=2.5,
    ax=ax1,
)
ax1.set_title(
    "Trend Comparison: System 1 vs System 2 (Average Score)",
    fontsize=14,
    fontweight="bold",
)
ax1.set_ylim(2, 8)
ax1.set_ylabel("Average Score")
ax1.fill_between(
    df_long[df_long["Metric"] == "Average"]["Waypoint ID"].unique(),
    df_long[(df_long["Metric"] == "Average") & (df_long["System"] == "System 1")][
        "Score"
    ],
    df_long[(df_long["Metric"] == "Average") & (df_long["System"] == "System 2")][
        "Score"
    ],
    color="red",
    alpha=0.1,
    label="Difference Zone",
)

# --- 차트 2: 지표별 분포 비교 (Paired Box Plot) ---
ax2 = fig.add_subplot(gs[0, 1])
# Average를 제외한 개별 지표만 필터링
metrics_df = df_long[df_long["Metric"] != "Average"]
sns.boxplot(
    data=metrics_df,
    x="Metric",
    y="Score",
    hue="System",
    palette=["#bdc3c7", "#e74c3c"],
    ax=ax2,
)
ax2.set_title("Score Distribution by System & Metric", fontsize=14, fontweight="bold")
ax2.set_ylim(0, 10)
ax2.tick_params(axis="x", rotation=15)

# --- 차트 3: 점수 변화량 히트맵 (Heatmap of Adjustments) ---
# 가로로 긴 하단 영역 사용
ax3 = fig.add_subplot(gs[1, :])
# 0을 기준으로 색상이 나뉘는 Diverging Colormap 사용 (Red=감소, Blue=증가)
sns.heatmap(
    df_delta.transpose(),
    annot=True,
    cmap="vlag_r",
    center=0,
    fmt=".1f",
    linewidths=0.5,
    ax=ax3,
)
ax3.set_title(
    "Impact of System 2: Score Adjustments (Delta = Sys2 - Sys1)",
    fontsize=14,
    fontweight="bold",
)
ax3.set_xlabel("Waypoint ID")

plt.tight_layout()
plt.show()
