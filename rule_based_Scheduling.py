
# ============================================================
#  Rule-Based Scheduling
#  AI 智慧工單排程與生產優先順序分析 
#
#  作者 : 國立雲林科技大學電機工程系 林家仁
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

np.random.seed(42)

# ============================================================
# 1. 配色
# ============================================================

COLOR_BLUE   = "#4E79A7"
COLOR_ORANGE = "#F28E2B"
COLOR_RED    = "#E15759"
COLOR_TEAL   = "#76B7B2"
COLOR_GREEN  = "#59A14F"
COLOR_YELLOW = "#EDC948"
COLOR_PURPLE = "#B07AA1"
COLOR_PINK   = "#FF9DA7"
COLOR_BROWN  = "#9C755F"
COLOR_GRAY   = "#BAB0AC"

BAR_COLORS = [
    COLOR_BLUE,
    COLOR_ORANGE,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_PURPLE,
    COLOR_TEAL,
    COLOR_YELLOW,
    COLOR_PINK,
    COLOR_BROWN,
    COLOR_GRAY,
    "#6B8EAD",
    "#D18B47",
    "#7FA36B",
    "#C96A72",
    "#8A76A8"
]

PASTEL_BLUE   = "#DCE8F5"
PASTEL_GREEN  = "#DCEBD8"
PASTEL_ORANGE = "#FBE4CD"
PASTEL_RED    = "#F7D8D8"
PASTEL_PURPLE = "#E9DDF0"
PASTEL_YELLOW = "#F8EFC8"
PASTEL_PINK   = "#F9DDE7"
PASTEL_TEAL   = "#D9EEEC"
PASTEL_GRAY   = "#E8E8E8"
PASTEL_BROWN  = "#EADFD6"

TEXT_DARK = "#3F4650"
ROW_WHITE = "#FFFFFF"
ROW_ALT   = "#F8FAFC"

plt.rcParams["figure.figsize"] = (13, 6.1)
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.style"] = "normal"

def style_table(df_show, header_color=PASTEL_BLUE, formats=None):
    styled = (
        df_show.style
        .set_properties(**{
            "color": TEXT_DARK,
            "text-align": "center",
            "font-style": "normal"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", header_color),
                    ("color", TEXT_DARK),
                    ("font-weight", "bold"),
                    ("font-style", "normal"),
                    ("text-align", "center"),
                    ("padding", "8px"),
                    ("border", "2px solid white")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("padding", "8px"),
                    ("border", "2px solid white"),
                    ("font-style", "normal")
                ]
            },
            {
                "selector": "tbody tr:nth-child(odd) td",
                "props": [("background-color", ROW_WHITE)]
            },
            {
                "selector": "tbody tr:nth-child(even) td",
                "props": [("background-color", ROW_ALT)]
            }
        ])
    )

    if formats:
        styled = styled.format(formats)

    return styled


# ============================================================
# 2. 建立 100 筆模擬工單
# ============================================================

N = 100

df = pd.DataFrame({
    "工單編號": [f"WO-{i+1:04d}" for i in range(N)],

    "客戶等級": np.random.choice(
        ["VIP", "重要客戶", "一般客戶"],
        size=N,
        p=[0.20, 0.30, 0.50]
    ),

    "訂單金額_KNTD": np.clip(
        np.random.normal(850, 350, N),
        150, 1800
    ).round(0),

    "距離交期天數": np.clip(
        np.round(np.random.normal(5.5, 3.0, N)),
        0, 14
    ).astype(int),

    "是否急單": np.random.choice(
        ["是", "否"],
        size=N,
        p=[0.14, 0.86]
    ),

    "生產工時": np.clip(
        np.random.normal(12, 5, N),
        3, 28
    ).round(1),

    "換線工時": np.clip(
        np.random.normal(2.8, 1.1, N),
        0.5, 6.0
    ).round(1),

    "機台負載率": np.clip(
        np.random.normal(78, 10, N),
        50, 98
    ).round(1),

    "材料是否到齊": np.random.choice(
        ["是", "否"],
        size=N,
        p=[0.88, 0.12]
    ),

    "品質風險": np.random.choice(
        ["低", "中", "高"],
        size=N,
        p=[0.68, 0.22, 0.10]
    )
})


# ============================================================
# 3. Rule-Based 排程分數
# ============================================================

def score_urgent(row):
    return 30 if row["是否急單"] == "是" else 0

def score_due(row):
    d = row["距離交期天數"]
    if d <= 1:
        return 30
    elif d <= 2:
        return 25
    elif d <= 4:
        return 15
    elif d <= 7:
        return 5
    return 0

def score_customer(row):
    if row["客戶等級"] == "VIP":
        return 20
    elif row["客戶等級"] == "重要客戶":
        return 10
    return 0

def score_material(row):
    return 15 if row["材料是否到齊"] == "是" else -35

def score_setup(row):
    if row["換線工時"] <= 2:
        return 10
    elif row["換線工時"] >= 4.5:
        return -5
    return 0

def score_load(row):
    return -10 if row["機台負載率"] >= 90 else 0

def score_quality(row):
    if row["品質風險"] == "高":
        return -15
    elif row["品質風險"] == "中":
        return -5
    return 0

df["急單分數"] = df.apply(score_urgent, axis=1)
df["交期分數"] = df.apply(score_due, axis=1)
df["客戶分數"] = df.apply(score_customer, axis=1)
df["材料分數"] = df.apply(score_material, axis=1)
df["換線分數"] = df.apply(score_setup, axis=1)
df["負載分數"] = df.apply(score_load, axis=1)
df["品質分數"] = df.apply(score_quality, axis=1)

score_cols = [
    "急單分數",
    "交期分數",
    "客戶分數",
    "材料分數",
    "換線分數",
    "負載分數",
    "品質分數"
]

df["排程分數"] = df[score_cols].sum(axis=1)


# ============================================================
# 4. 規則說明
# ============================================================

def explain_rule(row):
    reasons = []

    if row["急單分數"] != 0:
        reasons.append(f"急單 {row['急單分數']:+d}")

    if row["交期分數"] != 0:
        reasons.append(f"交期 {row['交期分數']:+d}")

    if row["客戶分數"] != 0:
        reasons.append(f"{row['客戶等級']} {row['客戶分數']:+d}")

    reasons.append(
        "材料到齊 +15"
        if row["材料分數"] > 0
        else "材料未到 -35"
    )

    if row["換線分數"] != 0:
        reasons.append(f"換線 {row['換線分數']:+d}")

    if row["負載分數"] != 0:
        reasons.append(f"高負載 {row['負載分數']:+d}")

    if row["品質分數"] != 0:
        reasons.append(f"品質風險 {row['品質分數']:+d}")

    return "；".join(reasons)

df["規則說明"] = df.apply(explain_rule, axis=1)


# ============================================================
# 5. 排程排序
# ============================================================

df["可立即排產"] = np.where(
    df["材料是否到齊"] == "是",
    "是",
    "否"
)

df_sorted = (
    df.sort_values(
        by=[
            "可立即排產",
            "排程分數",
            "距離交期天數",
            "換線工時"
        ],
        ascending=[
            False,
            False,
            True,
            True
        ]
    )
    .reset_index(drop=True)
)

df_sorted["排程順位"] = np.arange(
    1,
    len(df_sorted) + 1
)

def schedule_status(row):
    if row["材料是否到齊"] == "否":
        return "暫緩－待料"
    if row["品質風險"] == "高":
        return "品質確認"
    if row["排程順位"] <= 10:
        return "今日優先"
    if row["距離交期天數"] <= 2:
        return "交期風險"
    return "一般排程"

df_sorted["排程狀態"] = df_sorted.apply(
    schedule_status,
    axis=1
)


# ============================================================
# 6. 表格輸出
# ============================================================

print("=" * 76)
print("Day 1｜Rule-Based Scheduling－AI 智慧工單排程與生產優先順序分析")
print("=" * 76)
print(f"待排程工單數：{len(df_sorted)}")
print("分析目的：把生管人員的排程經驗轉換成可計算、可解釋的排程規則。")
print()

raw_cols = [
    "工單編號","客戶等級","訂單金額_KNTD","距離交期天數",
    "是否急單","生產工時","換線工時","機台負載率",
    "材料是否到齊","品質風險"
]

display(
    style_table(
        df.head(15)[raw_cols],
        PASTEL_BLUE,
        formats={
            "訂單金額_KNTD": "{:,.0f}",
            "生產工時": "{:.1f}",
            "換線工時": "{:.1f}",
            "機台負載率": "{:.1f}%"
        }
    )
)

rules_df = pd.DataFrame({
    "規則類別": [
        "急單","交期","交期","交期","交期","客戶","客戶",
        "材料","材料","換線","換線","機台負載","品質","品質"
    ],
    "判斷條件": [
        "急單 = 是","交期 ≤ 1 天","交期 ≤ 2 天","交期 ≤ 4 天","交期 ≤ 7 天",
        "VIP 客戶","重要客戶","材料到齊","材料未到",
        "換線工時 ≤ 2 小時","換線工時 ≥ 4.5 小時",
        "機台負載率 ≥ 90%","品質風險 = 高","品質風險 = 中"
    ],
    "加扣分": [
        "+30","+30","+25","+15","+5","+20","+10","+15","-35","+10","-5","-10","-15","-5"
    ],
    "管理意義": [
        "提高急單處理優先度","交期壓力最高","交期即將到期","需提前安排",
        "一般交期提醒","核心客戶優先","重要客戶適度加權",
        "具備立即生產條件","目前無法投入生產","降低換線損失",
        "避免長時間換線","避免設備持續高負載","先確認品質條件",
        "適度降低排程優先度"
    ]
})

print("\n【表 1｜Rule-Based Scheduling 排程規則】")
display(style_table(rules_df, PASTEL_GREEN))

full_schedule_cols = [
    "排程順位","工單編號","客戶等級","距離交期天數","是否急單",
    "材料是否到齊","品質風險","換線工時","機台負載率",
    "排程分數","排程狀態"
]

print("\n【表 2｜完整排程結果－前 30 筆】")
display(
    style_table(
        df_sorted.head(30)[full_schedule_cols],
        PASTEL_TEAL,
        formats={
            "換線工時": "{:.1f}",
            "機台負載率": "{:.1f}%",
            "排程分數": "{:.0f}"
        }
    )
)

print("\n【表 3｜排程順位 Top 10】")
display(
    style_table(
        df_sorted.head(10)[full_schedule_cols],
        PASTEL_ORANGE,
        formats={
            "換線工時": "{:.1f}",
            "機台負載率": "{:.1f}%",
            "排程分數": "{:.0f}"
        }
    )
)

breakdown_cols = [
    "排程順位","工單編號","急單分數","交期分數","客戶分數",
    "材料分數","換線分數","負載分數","品質分數","排程分數"
]

print("\n【表 4｜Top 10 排程分數來源拆解】")
display(
    style_table(
        df_sorted.head(10)[breakdown_cols],
        PASTEL_YELLOW,
        formats={
            "急單分數": "{:+.0f}",
            "交期分數": "{:+.0f}",
            "客戶分數": "{:+.0f}",
            "材料分數": "{:+.0f}",
            "換線分數": "{:+.0f}",
            "負載分數": "{:+.0f}",
            "品質分數": "{:+.0f}",
            "排程分數": "{:.0f}"
        }
    )
)

print("\n【表 5｜Top 10 為什麼排在前面】")
display(
    style_table(
        df_sorted.head(10)[[
            "排程順位","工單編號","排程分數","規則說明"
        ]],
        PASTEL_PINK,
        formats={"排程分數": "{:.0f}"}
    )
)

urgent_df = df_sorted[df_sorted["是否急單"] == "是"].copy()
print("\n【表 6｜急單分析】")
display(
    style_table(
        urgent_df[[
            "排程順位","工單編號","客戶等級","距離交期天數",
            "材料是否到齊","品質風險","排程分數","排程狀態"
        ]],
        PASTEL_RED,
        formats={"排程分數": "{:.0f}"}
    )
)

due_risk_df = df_sorted[df_sorted["距離交期天數"] <= 2].copy()
print("\n【表 7｜交期 ≤ 2 天風險工單】")
display(
    style_table(
        due_risk_df[[
            "排程順位","工單編號","客戶等級","距離交期天數",
            "是否急單","材料是否到齊","品質風險","排程分數","排程狀態"
        ]],
        PASTEL_ORANGE,
        formats={"排程分數": "{:.0f}"}
    )
)

material_hold_df = df_sorted[df_sorted["材料是否到齊"] == "否"].copy()
material_hold_df["材料到齊後預估分數"] = material_hold_df["排程分數"] + 50

print("\n【表 8｜待料工單與材料到齊後潛在分數】")
display(
    style_table(
        material_hold_df[[
            "工單編號","客戶等級","距離交期天數","是否急單",
            "排程分數","材料到齊後預估分數","排程狀態"
        ]],
        PASTEL_BROWN,
        formats={
            "排程分數": "{:.0f}",
            "材料到齊後預估分數": "{:.0f}"
        }
    )
)

quality_df = df_sorted[df_sorted["品質風險"] == "高"].copy()
print("\n【表 9｜高品質風險工單】")
display(
    style_table(
        quality_df[[
            "排程順位","工單編號","客戶等級","距離交期天數",
            "是否急單","排程分數","排程狀態"
        ]],
        PASTEL_PURPLE,
        formats={"排程分數": "{:.0f}"}
    )
)

high_load_df = df_sorted[df_sorted["機台負載率"] >= 90].copy()
print("\n【表 10｜機台高負載工單】")
display(
    style_table(
        high_load_df[[
            "排程順位","工單編號","機台負載率",
            "距離交期天數","是否急單","排程分數","排程狀態"
        ]],
        PASTEL_GRAY,
        formats={
            "機台負載率": "{:.1f}%",
            "排程分數": "{:.0f}"
        }
    )
)

vip_df = df_sorted[df_sorted["客戶等級"] == "VIP"].copy()
print("\n【表 11｜VIP 客戶工單】")
display(
    style_table(
        vip_df[[
            "排程順位","工單編號","距離交期天數","是否急單",
            "材料是否到齊","品質風險","排程分數","排程狀態"
        ]],
        PASTEL_BLUE,
        formats={"排程分數": "{:.0f}"}
    )
)

summary_df = pd.DataFrame({
    "分析項目": [
        "總待排程工單","排程順位 Top 10","急單","交期 ≤ 2 天",
        "材料未到","高品質風險","機台負載 ≥ 90%","VIP 工單"
    ],
    "工單數": [
        len(df_sorted),
        10,
        (df_sorted["是否急單"] == "是").sum(),
        (df_sorted["距離交期天數"] <= 2).sum(),
        (df_sorted["材料是否到齊"] == "否").sum(),
        (df_sorted["品質風險"] == "高").sum(),
        (df_sorted["機台負載率"] >= 90).sum(),
        (df_sorted["客戶等級"] == "VIP").sum()
    ],
    "管理意義": [
        "本次需要安排生產順位的工單",
        "目前優先度最高的工單",
        "需優先關注客戶需求",
        "具有近期交期壓力",
        "目前無法立即投入生產",
        "生產前需先確認品質風險",
        "需注意設備負荷",
        "核心客戶工單"
    ]
})

print("\n【表 12｜排程管理決策摘要】")
display(style_table(summary_df, PASTEL_GREEN))


# ============================================================
# 7. 圖 1－Top 15 排程優先分數
#    每一根長條不同顏色
# ============================================================

plot_df = df_sorted.head(15).copy()
wo_labels = [x.replace("WO-", "WO\n") for x in plot_df["工單編號"]]

fig, ax = plt.subplots(figsize=(13, 6.1))

bars = ax.bar(
    np.arange(len(plot_df)),
    plot_df["排程分數"],
    width=0.52,
    color=BAR_COLORS[:len(plot_df)]
)

ax.set_xticks(np.arange(len(plot_df)))
ax.set_xticklabels(
    wo_labels,
    rotation=0,
    ha="center",
    fontstyle="normal"
)

ax.set_title("Top 15 Work Orders by Scheduling Priority Score")
ax.set_xlabel("Work Order")
ax.set_ylabel("Priority Score")
ax.grid(axis="y", alpha=0.22)

for bar, value in zip(bars, plot_df["排程分數"]):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        value + 1,
        f"{int(value)}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontstyle="normal"
    )

plt.tight_layout()
plt.show()


# ============================================================
# 8. 圖 2－排程分數分布
# ============================================================

fig, ax = plt.subplots(figsize=(13, 6.1))

ax.hist(
    df_sorted["排程分數"],
    bins=12,
    alpha=0.82,
    color=COLOR_TEAL
)

ax.axvline(
    df_sorted["排程分數"].mean(),
    linestyle="--",
    linewidth=2,
    color=COLOR_RED,
    label=f"Mean = {df_sorted['排程分數'].mean():.1f}"
)

ax.set_title("Distribution of Scheduling Priority Scores")
ax.set_xlabel("Priority Score")
ax.set_ylabel("Number of Work Orders")
ax.grid(axis="y", alpha=0.22)
ax.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 9. 圖 3－排程狀態數量
#    各類別不同顏色
# ============================================================

status_order = [
    "今日優先",
    "交期風險",
    "一般排程",
    "品質確認",
    "暫緩－待料"
]

status_count = (
    df_sorted["排程狀態"]
    .value_counts()
    .reindex(status_order, fill_value=0)
)

status_labels_en = [
    "Today\nPriority",
    "Due-Date\nRisk",
    "Normal\nSchedule",
    "Quality\nCheck",
    "Material\nHold"
]

status_colors = [
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_BLUE,
    COLOR_PURPLE,
    COLOR_RED
]

fig, ax = plt.subplots(figsize=(13, 6.1))

bars = ax.bar(
    np.arange(len(status_labels_en)),
    status_count.values,
    width=0.52,
    color=status_colors
)

ax.set_xticks(np.arange(len(status_labels_en)))
ax.set_xticklabels(
    status_labels_en,
    rotation=0,
    ha="center",
    fontstyle="normal"
)

ax.set_title("Work Order Scheduling Status")
ax.set_ylabel("Number of Work Orders")
ax.grid(axis="y", alpha=0.22)

for bar, value in zip(bars, status_count.values):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.5,
        f"{value}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# ============================================================
# 10. 圖 4－交期 vs 排程分數
#     依排程狀態上色
# ============================================================

status_color_map = {
    "今日優先": COLOR_GREEN,
    "交期風險": COLOR_ORANGE,
    "一般排程": COLOR_BLUE,
    "品質確認": COLOR_PURPLE,
    "暫緩－待料": COLOR_RED
}

fig, ax = plt.subplots(figsize=(13, 6.1))

for status in status_order:
    sub = df_sorted[df_sorted["排程狀態"] == status]

    if len(sub) == 0:
        continue

    ax.scatter(
        sub["距離交期天數"],
        sub["排程分數"],
        s=60,
        alpha=0.75,
        label={
            "今日優先": "Today Priority",
            "交期風險": "Due-Date Risk",
            "一般排程": "Normal",
            "品質確認": "Quality Check",
            "暫緩－待料": "Material Hold"
        }[status],
        color=status_color_map[status]
    )

ax.set_title("Due Date vs Scheduling Priority Score")
ax.set_xlabel("Days to Due Date")
ax.set_ylabel("Priority Score")
ax.grid(alpha=0.22)
ax.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 11. 圖 5－各規則平均貢獻
#     每一根規則長條不同顏色
# ============================================================

component_mean = df_sorted[score_cols].mean()

component_labels = [
    "Urgent",
    "Due\nDate",
    "Customer",
    "Material",
    "Setup\nTime",
    "Machine\nLoad",
    "Quality\nRisk"
]

component_colors = [
    COLOR_RED,
    COLOR_ORANGE,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_TEAL,
    COLOR_PURPLE,
    COLOR_PINK
]

fig, ax = plt.subplots(figsize=(13, 6.1))

bars = ax.bar(
    np.arange(len(component_labels)),
    component_mean.values,
    width=0.52,
    color=component_colors
)

ax.set_xticks(np.arange(len(component_labels)))
ax.set_xticklabels(
    component_labels,
    rotation=0,
    ha="center",
    fontstyle="normal"
)

ax.axhline(0, linewidth=1, color="#666666")

ax.set_title("Average Contribution of Scheduling Rules")
ax.set_ylabel("Average Score Contribution")
ax.grid(axis="y", alpha=0.22)

for bar, value in zip(bars, component_mean.values):
    offset = 0.3 if value >= 0 else -0.8

    ax.text(
        bar.get_x() + bar.get_width()/2,
        value + offset,
        f"{value:.1f}",
        ha="center",
        fontstyle="normal"
    )

plt.tight_layout()
plt.show()


# ============================================================
# 12. 最後摘要
# ============================================================

print()
print("=" * 76)
print("Rule-Based Scheduling 分析摘要")
print("=" * 76)

print(f"總待排程工單：{len(df_sorted)}")
print(f"急單：{(df_sorted['是否急單']=='是').sum()}")
print(f"交期 ≤ 2 天：{(df_sorted['距離交期天數']<=2).sum()}")
print(f"材料未到：{(df_sorted['材料是否到齊']=='否').sum()}")
print(f"高品質風險：{(df_sorted['品質風險']=='高').sum()}")
print(f"高機台負載：{(df_sorted['機台負載率']>=90).sum()}")

print()
print("核心概念：")
print("Rule-Based Scheduling 將主管的排程經驗轉成明確規則，")
print("透過加分、扣分與限制條件，自動產生工單優先順序。")

print()
print("AI Agent 延伸：")
print("可根據排程結果，自動產生今日優先生產清單、")
print("待料提醒、交期風險提醒、品質確認與管理建議。")

output_file = "RuleBasedScheduling_100工單_彩色圖表修正版.csv"

df_sorted.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print()
print(f"完整排程結果已輸出：{output_file}")
