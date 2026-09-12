
# ===========================================================
#  主題 : Isolation Forest - AI 生產異常工單偵測與風險分析
#  目標 : 從大量工單中自動找出少數異常工單，
#         並產生異常分數、異常原因與優先檢查名單
#  情境 : 金屬機電／設備製造業 ERP / MES 工單資料資料模擬
#
#  資料設計：
#  共 1,000 筆工單
#  正常工單：950 筆
#  模擬異常：50 筆
#
#
#  作者 : 國立雲林科技大學電機工程系 林家仁
# ===========================================================

!pip -q install scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score

# ===========================================================
# 1. 配色
# ===========================================================

COLOR_BLUE   = "#4E79A7"
COLOR_ORANGE = "#F28E2B"
COLOR_RED    = "#E15759"
COLOR_TEAL   = "#76B7B2"
COLOR_GREEN  = "#59A14F"
COLOR_YELLOW = "#EDC948"
COLOR_PURPLE = "#B07AA1"
COLOR_PINK   = "#FF9DA7"

PASTEL_BLUE   = "#DCE8F5"
PASTEL_ORANGE = "#FBE4CD"
PASTEL_RED    = "#F7D8D8"
PASTEL_GREEN  = "#DCEBD8"

TEXT_DARK = "#4A4A4A"
GRID_COLOR = "#E6E6E6"

FIG_W = 13
FIG_H = 6.1

plt.rcParams["figure.figsize"] = (FIG_W, FIG_H)
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.unicode_minus"] = False

# ===========================================================
# 2. 表格樣式
# ===========================================================

def style_table(df_show, header_color=PASTEL_BLUE, formats=None):
    styled = (
        df_show.style
        .set_properties(**{
            "background-color": "#FBFCFE",
            "color": TEXT_DARK,
            "text-align": "center"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", header_color),
                    ("color", TEXT_DARK),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("padding", "9px"),
                    ("border", "2px solid white")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("padding", "8px"),
                    ("border", "2px solid white")
                ]
            }
        ])
    )

    if formats:
        styled = styled.format(formats)

    return styled

# ===========================================================
# 3. 建立 1,000 筆模擬製造工單
# ===========================================================

np.random.seed(42)

N_NORMAL = 950
N_ANOMALY = 50

def clipped_normal(mean, std, low, high, n):
    values = np.random.normal(mean, std, n)
    return np.clip(values, low, high)

# -----------------------------------------------------------
# 3-1 正常工單
# -----------------------------------------------------------

normal = pd.DataFrame({
    "生產數量":
        np.random.randint(20, 201, N_NORMAL),

    "材料成本_KNTD":
        clipped_normal(1200, 280, 450, 2200, N_NORMAL),

    "加工工時":
        clipped_normal(95, 22, 35, 165, N_NORMAL),

    "組裝工時":
        clipped_normal(72, 18, 20, 130, N_NORMAL),

    "測試工時":
        clipped_normal(38, 10, 10, 75, N_NORMAL),

    "換線工時":
        clipped_normal(7, 1.8, 2, 13, N_NORMAL),

    "機台負載率":
        clipped_normal(78, 8, 50, 98, N_NORMAL),

    "不良率":
        clipped_normal(2.4, 0.9, 0.3, 6.0, N_NORMAL)
})

normal["實際製造成本_KNTD"] = (
    180
    + normal["生產數量"] * 5.5
    + normal["材料成本_KNTD"] * 0.88
    + normal["加工工時"] * 7.2
    + normal["組裝工時"] * 5.0
    + normal["測試工時"] * 6.2
    + normal["換線工時"] * 8.5
    + normal["不良率"] * 28
    + np.random.normal(0, 65, N_NORMAL)
)

normal["真實狀態"] = "正常"
normal["異常類型_模擬"] = "正常"

# -----------------------------------------------------------
# 3-2 異常工單
#     A 加工工時異常
#     B 材料成本異常
#     C 不良率異常
#     D 綜合成本異常
# -----------------------------------------------------------

anomaly_parts = []

# A. 加工工時異常 15 筆
n = 15
a = normal.sample(n=n, random_state=100).copy().reset_index(drop=True)
a["加工工時"] = (
    a["加工工時"]
    * np.random.uniform(1.8, 2.4, n)
)
a["實際製造成本_KNTD"] += (
    (a["加工工時"] - 95) * 8
)
a["真實狀態"] = "異常"
a["異常類型_模擬"] = "加工工時異常"
anomaly_parts.append(a)

# B. 材料成本異常 12 筆
n = 12
a = normal.sample(n=n, random_state=101).copy().reset_index(drop=True)
a["材料成本_KNTD"] = (
    a["材料成本_KNTD"]
    * np.random.uniform(1.8, 2.3, n)
)
a["實際製造成本_KNTD"] += (
    (a["材料成本_KNTD"] - 1200) * 0.9
)
a["真實狀態"] = "異常"
a["異常類型_模擬"] = "材料成本異常"
anomaly_parts.append(a)

# C. 不良率異常 12 筆
n = 12
a = normal.sample(n=n, random_state=102).copy().reset_index(drop=True)
a["不良率"] = np.random.uniform(9.0, 16.0, n)
a["實際製造成本_KNTD"] += (
    a["不良率"] * 45
)
a["真實狀態"] = "異常"
a["異常類型_模擬"] = "不良率異常"
anomaly_parts.append(a)

# D. 綜合成本異常 11 筆
n = 11
a = normal.sample(n=n, random_state=103).copy().reset_index(drop=True)
a["實際製造成本_KNTD"] = (
    a["實際製造成本_KNTD"]
    * np.random.uniform(1.35, 1.60, n)
)
a["換線工時"] = (
    a["換線工時"]
    * np.random.uniform(1.4, 1.9, n)
)
a["真實狀態"] = "異常"
a["異常類型_模擬"] = "綜合成本異常"
anomaly_parts.append(a)

anomalies = pd.concat(
    anomaly_parts,
    ignore_index=True
)

df = pd.concat(
    [normal, anomalies],
    ignore_index=True
)

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

df.insert(
    0,
    "工單編號",
    [f"WO-{i+1:04d}" for i in range(len(df))]
)

# 格式整理
df["生產數量"] = (
    df["生產數量"]
    .round(0)
    .astype(int)
)

for c in [
    "材料成本_KNTD",
    "實際製造成本_KNTD"
]:
    df[c] = df[c].round(0)

for c in [
    "加工工時",
    "組裝工時",
    "測試工時",
    "換線工時",
    "機台負載率",
    "不良率"
]:
    df[c] = df[c].round(1)

print("=" * 68)
print("Day 1｜Isolation Forest 生產異常工單偵測與風險分析")
print("=" * 68)

print(f"資料筆數：{len(df):,}")
print("目標：從大量工單中，自動找出少數值得優先檢查的異常工單。\n")

display(
    style_table(
        df.head(10)[[
            "工單編號",
            "生產數量",
            "材料成本_KNTD",
            "加工工時",
            "組裝工時",
            "測試工時",
            "換線工時",
            "機台負載率",
            "不良率",
            "實際製造成本_KNTD"
        ]],
        header_color=PASTEL_BLUE,
        formats={
            "材料成本_KNTD": "{:,.0f}",
            "實際製造成本_KNTD": "{:,.0f}",
            "機台負載率": "{:.1f}%",
            "不良率": "{:.1f}%"
        }
    )
)

# ===========================================================
# 4. 統計摘要
# ===========================================================

print("\n【資料統計摘要】")

stats_df = df[[
    "生產數量",
    "材料成本_KNTD",
    "加工工時",
    "組裝工時",
    "測試工時",
    "換線工時",
    "機台負載率",
    "不良率",
    "實際製造成本_KNTD"
]].describe()

display(
    style_table(
        stats_df,
        header_color=PASTEL_GREEN,
        formats={
            "生產數量": "{:.1f}",
            "材料成本_KNTD": "{:,.0f}",
            "加工工時": "{:.1f}",
            "組裝工時": "{:.1f}",
            "測試工時": "{:.1f}",
            "換線工時": "{:.1f}",
            "機台負載率": "{:.1f}",
            "不良率": "{:.1f}",
            "實際製造成本_KNTD": "{:,.0f}"
        }
    )
)

# ===========================================================
# 5. 建立 Isolation Forest 模型
# ===========================================================

features = [
    "生產數量",
    "材料成本_KNTD",
    "加工工時",
    "組裝工時",
    "測試工時",
    "換線工時",
    "機台負載率",
    "不良率",
    "實際製造成本_KNTD"
]

X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(
    n_estimators=300,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

pred = iso.fit_predict(
    X_scaled
)

df["AI判定"] = np.where(
    pred == -1,
    "異常",
    "正常"
)

# Isolation Forest 的 score_samples 越小越異常
# 為方便教學，轉成「越高 = 越異常」
df["異常分數"] = (
    -iso.score_samples(X_scaled)
).round(4)

# ===========================================================
# 6. 模型評估
#    因為本範例是模擬資料，所以知道哪些資料是人工植入的異常
# ===========================================================

y_true = (
    df["真實狀態"] == "異常"
).astype(int)

y_pred = (
    df["AI判定"] == "異常"
).astype(int)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

summary_df = pd.DataFrame({
    "評估項目": [
        "總工單數",
        "AI 判定正常",
        "AI 判定異常",
        "異常比例",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "分析結果": [
        f"{len(df):,}",
        f"{(df['AI判定']=='正常').sum():,}",
        f"{(df['AI判定']=='異常').sum():,}",
        f"{(df['AI判定']=='異常').mean()*100:.1f}%",
        f"{precision*100:.1f}%",
        f"{recall*100:.1f}%",
        f"{f1:.3f}"
    ],
    "中文說明": [
        "本次分析的工單總數",
        "模型判定為正常的工單",
        "模型判定為需優先檢查的工單",
        "異常工單占全部資料的比例",
        "AI 判定異常中，有多少是真異常",
        "真實異常中，有多少被 AI 找出",
        "Precision 與 Recall 的綜合指標"
    ]
})

print("\n【Isolation Forest 分析結果摘要】")

display(
    style_table(
        summary_df,
        header_color=PASTEL_GREEN
    )
)

# ===========================================================
# 7. 圖表 1：正常 vs 異常工單數量
# ===========================================================

counts = (
    df["AI判定"]
    .value_counts()
    .reindex(["正常", "異常"])
)

fig, ax = plt.subplots(
    figsize=(FIG_W, FIG_H)
)

bars = ax.bar(
    ["Normal", "Anomaly"],
    counts.values,
    width=0.55,
    color=[
        COLOR_BLUE,
        COLOR_RED
    ]
)

ax.set_title(
    "Manufacturing Work Order Anomaly Detection"
)

ax.set_ylabel(
    "Number of Work Orders"
)

ax.grid(
    axis="y",
    alpha=0.22
)

for bar, value in zip(
    bars,
    counts.values
):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        value + 8,
        f"{value}",
        ha="center",
        fontsize=11
    )

plt.tight_layout()
plt.show()

print(
    "圖 1：AI 將 1,000 張工單分成正常與異常兩類，"
    "讓主管先聚焦於少數需要優先檢查的工單。"
)

# ===========================================================
# 8. 圖表 2：異常分數分布
# ===========================================================

fig, ax = plt.subplots(
    figsize=(FIG_W, FIG_H)
)

ax.hist(
    df.loc[
        df["AI判定"] == "正常",
        "異常分數"
    ],
    bins=25,
    alpha=0.72,
    label="Normal",
    color=COLOR_BLUE
)

ax.hist(
    df.loc[
        df["AI判定"] == "異常",
        "異常分數"
    ],
    bins=15,
    alpha=0.78,
    label="Anomaly",
    color=COLOR_RED
)

threshold = (
    df.loc[
        df["AI判定"] == "異常",
        "異常分數"
    ]
    .min()
)

ax.axvline(
    threshold,
    linestyle="--",
    linewidth=2,
    color="#555555",
    label="Anomaly Threshold"
)

ax.set_title(
    "Distribution of Isolation Forest Anomaly Scores"
)

ax.set_xlabel(
    "Anomaly Score (Higher = More Anomalous)"
)

ax.set_ylabel(
    "Number of Work Orders"
)

ax.grid(
    axis="y",
    alpha=0.22
)

ax.legend()

plt.tight_layout()
plt.show()

print(
    "圖 2：異常分數越高，代表該工單和大多數正常工單越不相似。"
)

# ===========================================================
# 9. PCA 2D 異常分布
# ===========================================================

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

normal_mask = (
    df["AI判定"] == "正常"
)

anom_mask = (
    df["AI判定"] == "異常"
)

fig, ax = plt.subplots(
    figsize=(FIG_W, FIG_H)
)

ax.scatter(
    df.loc[normal_mask, "PCA1"],
    df.loc[normal_mask, "PCA2"],
    s=28,
    alpha=0.45,
    color=COLOR_BLUE,
    label="Normal"
)

ax.scatter(
    df.loc[anom_mask, "PCA1"],
    df.loc[anom_mask, "PCA2"],
    s=70,
    alpha=0.92,
    color=COLOR_RED,
    edgecolors="white",
    linewidths=0.7,
    label="Anomaly"
)

ax.set_title(
    "PCA Visualization of Manufacturing Anomalies"
)

ax.set_xlabel(
    "Principal Component 1"
)

ax.set_ylabel(
    "Principal Component 2"
)

ax.grid(
    alpha=0.22
)

ax.legend()

plt.tight_layout()
plt.show()

print(
    "圖 3：藍色為一般工單，紅色為 AI 判定的異常工單。"
)

# ===========================================================
# 10. 推估主要異常原因
# ===========================================================

normal_ref = (
    df[df["AI判定"] == "正常"]
    [features]
)

means = normal_ref.mean()

stds = (
    normal_ref.std(ddof=0)
    .replace(0, 1)
)

z = (
    (df[features] - means)
    / stds
).abs()

reason_map = {
    "生產數量":
        "生產數量異常",

    "材料成本_KNTD":
        "材料成本異常",

    "加工工時":
        "加工工時異常",

    "組裝工時":
        "組裝工時異常",

    "測試工時":
        "測試工時異常",

    "換線工時":
        "換線工時異常",

    "機台負載率":
        "機台負載異常",

    "不良率":
        "不良率異常",

    "實際製造成本_KNTD":
        "製造成本異常"
}

df["主要異常原因"] = (
    z.idxmax(axis=1)
    .map(reason_map)
)

# ===========================================================
# 11. Top 10 異常工單
# ===========================================================

top10 = (
    df[df["AI判定"] == "異常"]
    .sort_values(
        "異常分數",
        ascending=False
    )
    .head(10)
    [[
        "工單編號",
        "異常分數",
        "主要異常原因",
        "生產數量",
        "材料成本_KNTD",
        "加工工時",
        "換線工時",
        "不良率",
        "實際製造成本_KNTD"
    ]]
)

print("\n【異常程度最高 Top 10 工單】")

display(
    style_table(
        top10,
        header_color=PASTEL_RED,
        formats={
            "異常分數": "{:.4f}",
            "材料成本_KNTD": "{:,.0f}",
            "實際製造成本_KNTD": "{:,.0f}",
            "不良率": "{:.1f}%"
        }
    )
)

print(
    "管理重點：異常分數越高，代表越值得優先檢查。"
)

# ===========================================================
# 12. 正常 vs 異常工單平均特徵
# ===========================================================

compare = (
    df.groupby(
        "AI判定"
    )[features]
    .mean()
    .loc[
        ["正常", "異常"]
    ]
)

print("\n【正常與異常工單平均特徵比較】")

display(
    style_table(
        compare,
        header_color=PASTEL_ORANGE,
        formats={
            "生產數量": "{:.1f}",
            "材料成本_KNTD": "{:,.0f}",
            "加工工時": "{:.1f}",
            "組裝工時": "{:.1f}",
            "測試工時": "{:.1f}",
            "換線工時": "{:.1f}",
            "機台負載率": "{:.1f}%",
            "不良率": "{:.1f}%",
            "實際製造成本_KNTD": "{:,.0f}"
        }
    )
)

# ===========================================================
# 13. 圖表 4：Normal vs Anomaly Profile
# ===========================================================

profile = pd.DataFrame(
    X_scaled,
    columns=features
)

profile["AI判定"] = (
    df["AI判定"]
    .values
)

profile_mean = (
    profile.groupby(
        "AI判定"
    )[features]
    .mean()
    .loc[
        ["正常", "異常"]
    ]
)

plot_features = [
    "材料成本_KNTD",
    "加工工時",
    "組裝工時",
    "測試工時",
    "換線工時",
    "不良率",
    "實際製造成本_KNTD"
]

labels_en = [
    "Material\nCost",
    "Machining\nHours",
    "Assembly\nHours",
    "Test\nHours",
    "Setup\nHours",
    "Defect\nRate",
    "Actual\nCost"
]

x = np.arange(
    len(plot_features)
)

width = 0.34

fig, ax = plt.subplots(
    figsize=(FIG_W, FIG_H)
)

ax.bar(
    x - width/2,
    profile_mean.loc[
        "正常",
        plot_features
    ],
    width=width,
    label="Normal",
    color=COLOR_BLUE
)

ax.bar(
    x + width/2,
    profile_mean.loc[
        "異常",
        plot_features
    ],
    width=width,
    label="Anomaly",
    color=COLOR_RED
)

ax.axhline(
    0,
    color="#777777",
    linewidth=1
)

ax.set_xticks(x)

ax.set_xticklabels(
    labels_en
)

ax.set_ylabel(
    "Standardized Average"
)

ax.set_title(
    "Normal vs Anomaly Work Order Profile"
)

ax.grid(
    axis="y",
    alpha=0.22
)

ax.legend()

plt.tight_layout()
plt.show()

print(
    "圖 4：紅色越高，表示異常工單在該項特徵上明顯高於一般工單。"
)

# ===========================================================
# 14. 最後摘要
# ===========================================================

print()
print("=" * 68)
print("Day 1｜Isolation Forest 分析結果摘要")
print("=" * 68)

print(
    f"總工單數          ：{len(df):,}"
)

print(
    f"AI 判定正常       ：{(df['AI判定']=='正常').sum():,}"
)

print(
    f"AI 判定異常       ：{(df['AI判定']=='異常').sum():,}"
)

print(
    f"異常比例          ：{(df['AI判定']=='異常').mean()*100:.1f}%"
)

print(
    f"Precision         ：{precision*100:.1f}%"
)

print(
    f"Recall            ：{recall*100:.1f}%"
)

print(
    f"F1 Score          ：{f1:.3f}"
)

print()
print("核心概念：")
print(
    "Isolation Forest 的目的不是預測一個數字，"
)
print(
    "而是從大量資料中找出與多數工單明顯不同的少數異常工單。"
)

print()
print("管理應用：")
print(
    "AI 可先把 1,000 張工單縮小成約 50 張優先檢查名單，"
)
print(
    "再由主管或 AI Agent 進一步判斷異常原因與處理順序。"
)

# ===========================================================
# 15. 匯出結果 CSV
# ===========================================================

output_file = "IsolationForest_1000工單_分析結果.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print()
print(
    f"分析結果已輸出：{output_file}"
)
