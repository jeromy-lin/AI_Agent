
# ===========================================================
#  主題 : K-Means 客戶分群－AI 客戶分群與業務策略分析
#  目標 : 使學員了解如何利用客戶交易與互動資料，
#         自動找出不同類型的客戶群組，並轉化成業務策略
#  情境 : 以金屬機電／設備製造業的 B2B 客戶為例，
#         利用採購金額、訂單次數、平均訂單金額、報價次數、
#         成交率、平均折扣率、毛利率與最近交易天數進行分群
#
#  作者 : 國立雲林科技大學電機工程系 林家仁
# ===========================================================

# ===========================================================
# 0. 載入套件
# ===========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ===========================================================
# 1. 視覺配色設定
#    圖表使用英文避免亂碼；表格與說明可用中文
# ===========================================================

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

PASTEL_BLUE   = "#DCE8F5"
PASTEL_ORANGE = "#FBE4CD"
PASTEL_RED    = "#F7D8D8"
PASTEL_TEAL   = "#D9ECEA"
PASTEL_GREEN  = "#DCEBD8"
PASTEL_PURPLE = "#E9DDF0"
PASTEL_YELLOW = "#F7EFCB"
PASTEL_PINK   = "#F9E2EA"

TEXT_DARK = "#4A4A4A"
GRID_COLOR = "#E6E6E6"
ROW_BG_1 = "#FCFCFD"
ROW_BG_2 = "#F8FAFC"

FIG_W = 13
FIG_H = 6.1

plt.rcParams["figure.figsize"] = (FIG_W, FIG_H)
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.unicode_minus"] = False

# ===========================================================
# 2. 表格美化函數
# ===========================================================

def style_table(df_show, header_color=PASTEL_BLUE, formats=None):
    styled = (
        df_show.style
        .set_properties(**{
            "background-color": ROW_BG_1,
            "color": TEXT_DARK,
            "text-align": "center",
            "border-color": "white"
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
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [
                    ("background-color", ROW_BG_2)
                ]
            }
        ])
    )
    if formats:
        styled = styled.format(formats)
    return styled

# ===========================================================
# 3. 建立 200 筆模擬 B2B 客戶資料
#    先設計 4 種典型客群，再加入合理的隨機波動
# ===========================================================

np.random.seed(42)

n_per_cluster = 50

def clipped_normal(mean, std, low, high, n):
    values = np.random.normal(mean, std, n)
    return np.clip(values, low, high)

# Customer Segment A : VIP Accounts
vip = pd.DataFrame({
    "年度採購金額_KNTD": clipped_normal(9000, 1200, 6500, 12500, n_per_cluster),
    "訂單次數": clipped_normal(24, 4, 15, 35, n_per_cluster),
    "平均訂單金額_KNTD": clipped_normal(380, 55, 260, 520, n_per_cluster),
    "報價次數": clipped_normal(30, 5, 20, 42, n_per_cluster),
    "成交率": clipped_normal(78, 7, 60, 92, n_per_cluster),
    "平均折扣率": clipped_normal(5.5, 1.2, 2, 9, n_per_cluster),
    "毛利率": clipped_normal(29, 4, 20, 38, n_per_cluster),
    "最近交易天數": clipped_normal(18, 8, 2, 40, n_per_cluster),
})

# Customer Segment B : Growth Accounts
growth = pd.DataFrame({
    "年度採購金額_KNTD": clipped_normal(5200, 900, 3200, 7600, n_per_cluster),
    "訂單次數": clipped_normal(16, 3, 9, 24, n_per_cluster),
    "平均訂單金額_KNTD": clipped_normal(320, 45, 220, 430, n_per_cluster),
    "報價次數": clipped_normal(27, 5, 16, 40, n_per_cluster),
    "成交率": clipped_normal(59, 8, 40, 78, n_per_cluster),
    "平均折扣率": clipped_normal(7.5, 1.5, 4, 12, n_per_cluster),
    "毛利率": clipped_normal(23, 4, 14, 32, n_per_cluster),
    "最近交易天數": clipped_normal(35, 12, 8, 70, n_per_cluster),
})

# Customer Segment C : Price-Sensitive Accounts
price_sensitive = pd.DataFrame({
    "年度採購金額_KNTD": clipped_normal(3600, 850, 1600, 5800, n_per_cluster),
    "訂單次數": clipped_normal(11, 3, 5, 19, n_per_cluster),
    "平均訂單金額_KNTD": clipped_normal(300, 60, 160, 460, n_per_cluster),
    "報價次數": clipped_normal(34, 6, 20, 48, n_per_cluster),
    "成交率": clipped_normal(34, 7, 18, 50, n_per_cluster),
    "平均折扣率": clipped_normal(13.5, 2.2, 8, 19, n_per_cluster),
    "毛利率": clipped_normal(14, 3, 7, 22, n_per_cluster),
    "最近交易天數": clipped_normal(65, 18, 25, 110, n_per_cluster),
})

# Customer Segment D : Dormant / At-Risk Accounts
dormant = pd.DataFrame({
    "年度採購金額_KNTD": clipped_normal(1700, 650, 500, 3400, n_per_cluster),
    "訂單次數": clipped_normal(5, 2, 1, 10, n_per_cluster),
    "平均訂單金額_KNTD": clipped_normal(250, 60, 100, 420, n_per_cluster),
    "報價次數": clipped_normal(8, 3, 2, 16, n_per_cluster),
    "成交率": clipped_normal(27, 8, 8, 45, n_per_cluster),
    "平均折扣率": clipped_normal(8.5, 2.0, 4, 14, n_per_cluster),
    "毛利率": clipped_normal(18, 4, 8, 28, n_per_cluster),
    "最近交易天數": clipped_normal(190, 45, 100, 300, n_per_cluster),
})

df = pd.concat(
    [vip, growth, price_sensitive, dormant],
    ignore_index=True
)

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

df.insert(
    0,
    "客戶編號",
    [f"CUST-{i+1:03d}" for i in range(len(df))]
)

int_cols = [
    "年度採購金額_KNTD",
    "訂單次數",
    "平均訂單金額_KNTD",
    "報價次數",
    "最近交易天數"
]

for col in int_cols:
    df[col] = df[col].round(0).astype(int)

for col in [
    "成交率",
    "平均折扣率",
    "毛利率"
]:
    df[col] = df[col].round(1)

# ===========================================================
# 4. 顯示客戶資料
# ===========================================================

print("=" * 62)
print("Day 1｜K-Means 客戶分群與業務策略分析")
print("=" * 62)
print(f"客戶資料筆數：{len(df)}")
print("分析目標：找出不同類型的客戶群組，支援差異化業務策略。\n")

display(
    style_table(
        df.head(10),
        header_color=PASTEL_BLUE,
        formats={
            "年度採購金額_KNTD": "{:,.0f}",
            "平均訂單金額_KNTD": "{:,.0f}",
            "成交率": "{:.1f}%",
            "平均折扣率": "{:.1f}%",
            "毛利率": "{:.1f}%"
        }
    )
)

# ===========================================================
# 5. 資料統計摘要
# ===========================================================

print("\n【資料統計摘要】")

stats_df = df.drop(columns=["客戶編號"]).describe()

# 統計摘要顯示格式：
# - 金額、次數、天數：小數點 0～1 位即可
# - 成交率、折扣率、毛利率：以百分比格式呈現
stats_formats = {
    "年度採購金額_KNTD": "{:,.0f}",
    "訂單次數": "{:.1f}",
    "平均訂單金額_KNTD": "{:,.0f}",
    "報價次數": "{:.1f}",
    "成交率": "{:.1f}%",
    "平均折扣率": "{:.1f}%",
    "毛利率": "{:.1f}%",
    "最近交易天數": "{:.1f}"
}

display(
    style_table(
        stats_df,
        header_color=PASTEL_GREEN,
        formats=stats_formats
    )
)

print(
    "說明：建模前先了解客戶資料的大致範圍，"
    "確認是否存在明顯異常或不合理值。"
)
print(
    "成交率、平均折扣率與毛利率皆以百分比（%）呈現；"
    "例如成交率 78.0% 表示 100 次有效商機中約有 78 次成交。"
)

# ===========================================================
# 6. 選擇 K-Means 使用的特徵
# ===========================================================

features = [
    "年度採購金額_KNTD",
    "訂單次數",
    "平均訂單金額_KNTD",
    "報價次數",
    "成交率",
    "平均折扣率",
    "毛利率",
    "最近交易天數"
]

X = df[features].copy()

print("\n【K-Means 使用特徵】")
print(f"共使用 {len(features)} 個客戶特徵：")
for i, feature in enumerate(features, 1):
    print(f"{i}. {feature}")

# ===========================================================
# 7. StandardScaler 特徵標準化
# ===========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n【特徵標準化】")
print(
    "由於採購金額、成交率、交易天數等欄位單位不同，"
    "K-Means 前先使用 StandardScaler 將特徵轉成相同尺度。"
)

# ===========================================================
# 8. Elbow Method + Silhouette Score
# ===========================================================

k_values = range(2, 9)
inertias = []
silhouette_scores = []

for k in k_values:
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(
        silhouette_score(X_scaled, labels)
    )

# Elbow Method
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.plot(
    list(k_values),
    inertias,
    marker="o",
    markersize=8,
    linewidth=2.5,
    color=COLOR_BLUE
)
ax.axvline(
    4,
    linestyle="--",
    linewidth=2,
    color=COLOR_RED,
    label="Selected K = 4"
)
ax.set_title("Elbow Method for Customer Segmentation")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Within-Cluster Sum of Squares (Inertia)")
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()

print(
    "圖 1：Elbow Method 用來觀察 K 增加時群內距離下降的程度；"
    "當下降幅度開始趨緩時，可視為合理的群數候選。"
)

# Silhouette Score
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
bars = ax.bar(
    [str(k) for k in k_values],
    silhouette_scores,
    width=0.55,
    color=[
        COLOR_BLUE,
        COLOR_ORANGE,
        COLOR_GREEN,
        COLOR_TEAL,
        COLOR_PURPLE,
        COLOR_YELLOW,
        COLOR_RED
    ]
)
ax.set_title("Silhouette Score by Number of Clusters")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Silhouette Score")
ax.grid(axis="y", alpha=0.25)
for bar, score in zip(bars, silhouette_scores):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.01,
        f"{score:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )
plt.tight_layout()
plt.show()

print(
    "圖 2：Silhouette Score 越接近 1，表示群組內部越相似、"
    "不同群組之間的區隔越清楚。"
)

# ===========================================================
# 9. 正式建立 K-Means 模型
# ===========================================================

K = 4
kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=20
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\n【K-Means 分群完成】")
print(f"本次將 200 位客戶分為 {K} 群。")

# ===========================================================
# 10. 各群客戶數量
# ===========================================================

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="客戶數")
)

print("\n【各群客戶數量】")
display(
    style_table(
        cluster_counts,
        header_color=PASTEL_ORANGE
    )
)

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
bars = ax.bar(
    [f"Cluster {c}" for c in cluster_counts["Cluster"]],
    cluster_counts["客戶數"],
    width=0.55,
    color=[
        COLOR_BLUE,
        COLOR_ORANGE,
        COLOR_GREEN,
        COLOR_PURPLE
    ]
)
ax.set_title("Number of Customers in Each Cluster")
ax.set_xlabel("Customer Cluster")
ax.set_ylabel("Number of Customers")
ax.grid(axis="y", alpha=0.25)
for bar, value in zip(bars, cluster_counts["客戶數"]):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        value + 1,
        f"{value}",
        ha="center"
    )
plt.tight_layout()
plt.show()

# ===========================================================
# 11. 建立 Cluster Profile
# ===========================================================

cluster_profile = (
    df.groupby("Cluster")[features]
    .mean()
    .round(1)
)

print("\n【各群客戶平均特徵】")
display(
    style_table(
        cluster_profile,
        header_color=PASTEL_TEAL,
        formats={
            "年度採購金額_KNTD": "{:,.0f}",
            "訂單次數": "{:.1f}",
            "平均訂單金額_KNTD": "{:,.0f}",
            "報價次數": "{:.1f}",
            "成交率": "{:.1f}%",
            "平均折扣率": "{:.1f}%",
            "毛利率": "{:.1f}%",
            "最近交易天數": "{:.1f}"
        }
    )
)

# ===========================================================
# 12. 自動判定客群角色
# ===========================================================

vip_score = (
    cluster_profile["年度採購金額_KNTD"].rank()
    + cluster_profile["毛利率"].rank()
    + cluster_profile["成交率"].rank()
    + cluster_profile["最近交易天數"].rank(ascending=False)
)
vip_cluster = vip_score.idxmax()

remaining = [c for c in cluster_profile.index if c != vip_cluster]
dormant_score = (
    cluster_profile.loc[remaining, "最近交易天數"].rank()
    + cluster_profile.loc[remaining, "訂單次數"].rank(ascending=False)
)
dormant_cluster = dormant_score.idxmax()

remaining = [
    c for c in cluster_profile.index
    if c not in [vip_cluster, dormant_cluster]
]
price_score = (
    cluster_profile.loc[remaining, "平均折扣率"].rank()
    + cluster_profile.loc[remaining, "毛利率"].rank(ascending=False)
    + cluster_profile.loc[remaining, "成交率"].rank(ascending=False)
)
price_cluster = price_score.idxmax()

growth_cluster = [
    c for c in cluster_profile.index
    if c not in [vip_cluster, dormant_cluster, price_cluster]
][0]

cluster_name_map_zh = {
    vip_cluster: "VIP 核心客戶",
    growth_cluster: "成長潛力客戶",
    price_cluster: "價格敏感客戶",
    dormant_cluster: "沉睡／流失風險客戶"
}

cluster_name_map_en = {
    vip_cluster: "VIP Accounts",
    growth_cluster: "Growth Accounts",
    price_cluster: "Price-Sensitive Accounts",
    dormant_cluster: "Dormant / At-Risk Accounts"
}

strategy_map = {
    "VIP 核心客戶":
        "優先維繫、專人服務、交叉銷售與長期合作",
    "成長潛力客戶":
        "增加拜訪與技術提案，提升成交率與採購規模",
    "價格敏感客戶":
        "控管折扣與毛利，重新設計報價與產品組合",
    "沉睡／流失風險客戶":
        "啟動喚回機制，確認流失原因並安排再接觸"
}

df["客群名稱"] = df["Cluster"].map(cluster_name_map_zh)

# ===========================================================
# 13. 客群分析總表
# ===========================================================

cluster_summary = (
    df.groupby(["Cluster", "客群名稱"])[features]
    .mean()
    .round(1)
    .reset_index()
)

cluster_summary["業務策略"] = (
    cluster_summary["客群名稱"]
    .map(strategy_map)
)

print("\n【AI 客戶分群與業務策略】")
display(
    style_table(
        cluster_summary,
        header_color=PASTEL_PINK,
        formats={
            "年度採購金額_KNTD": "{:,.0f}",
            "平均訂單金額_KNTD": "{:,.0f}",
            "成交率": "{:.1f}%",
            "平均折扣率": "{:.1f}%",
            "毛利率": "{:.1f}%"
        }
    )
)

# ===========================================================
# 14. PCA 2D 客戶分群視覺化
#     圖中文字全部英文，避免亂碼
# ===========================================================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

cluster_colors = {
    vip_cluster: COLOR_BLUE,
    growth_cluster: COLOR_GREEN,
    price_cluster: COLOR_ORANGE,
    dormant_cluster: COLOR_PURPLE
}

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

for cluster_id in sorted(df["Cluster"].unique()):
    subset = df[df["Cluster"] == cluster_id]
    ax.scatter(
        subset["PCA1"],
        subset["PCA2"],
        s=62,
        alpha=0.78,
        color=cluster_colors[cluster_id],
        edgecolors="white",
        linewidths=0.5,
        label=cluster_name_map_en[cluster_id]
    )

ax.set_title("PCA Visualization of Customer Segments")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.grid(alpha=0.22)
ax.legend(title="Customer Segment", fontsize=9)
plt.tight_layout()
plt.show()

print(
    "圖 4：每一個點代表一位客戶；顏色不同代表不同客群。"
)
print(
    "若同色客戶集中、不同顏色彼此分開，表示分群結果較容易解釋。"
)

# ===========================================================
# 15. 客群特徵比較
# ===========================================================

scaled_df = pd.DataFrame(
    X_scaled,
    columns=features
)
scaled_df["Cluster"] = df["Cluster"].values

scaled_profile = (
    scaled_df.groupby("Cluster")[features]
    .mean()
)

profile_features = [
    "年度採購金額_KNTD",
    "訂單次數",
    "報價次數",
    "成交率",
    "平均折扣率",
    "最近交易天數"
]

english_labels = [
    "Annual\nPurchase",
    "Order\nFrequency",
    "Quotation\nCount",
    "Conversion\nRate",
    "Discount\nRate",
    "Recency"
]

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

x = np.arange(len(profile_features))
width = 0.18

plot_order = [
    vip_cluster,
    growth_cluster,
    price_cluster,
    dormant_cluster
]

for i, cluster_id in enumerate(plot_order):
    values = (
        scaled_profile.loc[cluster_id, profile_features]
        .values
    )
    ax.bar(
        x + (i - 1.5) * width,
        values,
        width=width,
        color=cluster_colors[cluster_id],
        label=cluster_name_map_en[cluster_id]
    )

ax.axhline(0, color="#777777", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(english_labels)
ax.set_ylabel("Standardized Cluster Average")
ax.set_title("Customer Segment Profile Comparison")
ax.grid(axis="y", alpha=0.22)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

print(
    "圖 5：0 代表全體客戶平均值；高於 0 表示該群在此特徵高於平均，"
    "低於 0 則表示低於平均。"
)

# ===========================================================
# 16. 顯示各客群代表客戶
# ===========================================================

print("\n【各客群代表客戶】")

show_cols = [
    "客戶編號",
    "客群名稱",
    "年度採購金額_KNTD",
    "訂單次數",
    "成交率",
    "平均折扣率",
    "毛利率",
    "最近交易天數"
]

segment_header_colors = {
    "VIP 核心客戶": PASTEL_BLUE,
    "成長潛力客戶": PASTEL_GREEN,
    "價格敏感客戶": PASTEL_ORANGE,
    "沉睡／流失風險客戶": PASTEL_PURPLE
}

for segment in [
    "VIP 核心客戶",
    "成長潛力客戶",
    "價格敏感客戶",
    "沉睡／流失風險客戶"
]:
    print(f"\n--- {segment} ---")
    temp = (
        df[df["客群名稱"] == segment][show_cols]
        .head(5)
    )
    display(
        style_table(
            temp,
            header_color=segment_header_colors[segment],
            formats={
                "年度採購金額_KNTD": "{:,.0f}",
                "成交率": "{:.1f}%",
                "平均折扣率": "{:.1f}%",
                "毛利率": "{:.1f}%"
            }
        )
    )

# ===========================================================
# 17. 新客戶分群示範
# ===========================================================

new_customer = pd.DataFrame({
    "年度採購金額_KNTD": [6200],
    "訂單次數": [18],
    "平均訂單金額_KNTD": [340],
    "報價次數": [29],
    "成交率": [64.0],
    "平均折扣率": [7.0],
    "毛利率": [24.0],
    "最近交易天數": [24]
})

new_customer_scaled = scaler.transform(new_customer)
new_cluster = kmeans.predict(new_customer_scaled)[0]
new_segment_zh = cluster_name_map_zh[new_cluster]
new_segment_en = cluster_name_map_en[new_cluster]

print("\n【新客戶 AI 分群】")
display(
    style_table(
        new_customer,
        header_color=PASTEL_YELLOW,
        formats={
            "年度採購金額_KNTD": "{:,.0f}",
            "平均訂單金額_KNTD": "{:,.0f}",
            "成交率": "{:.1f}%",
            "平均折扣率": "{:.1f}%",
            "毛利率": "{:.1f}%"
        }
    )
)

print(f"AI 判定客群：{new_segment_zh} ({new_segment_en})")
print(f"建議業務策略：{strategy_map[new_segment_zh]}")

# ===========================================================
# 18. 分群結果摘要
# ===========================================================

selected_silhouette = silhouette_score(
    X_scaled,
    df["Cluster"]
)

print()
print("=" * 62)
print("Day 1｜K-Means 客戶分群結果摘要")
print("=" * 62)
print(f"客戶資料筆數      ：{len(df)}")
print(f"使用特徵數        ：{len(features)}")
print(f"分群數 K          ：{K}")
print(f"Silhouette Score  ：{selected_silhouette:.3f}")

print()
for cluster_id in plot_order:
    name = cluster_name_map_zh[cluster_id]
    count = df["Cluster"].eq(cluster_id).sum()
    print(f"{name:<12}：{count} 位")

print()
print("核心概念：")
print(
    "K-Means 不是預測一個數字，而是依照客戶特徵的相似程度，"
)
print(
    "自動把原本混在一起的客戶分成不同群組。"
)

print()
print("業務應用價值：")
print(
    "透過客戶分群，企業可以進一步進行 VIP 維繫、"
)
print(
    "成長客戶開發、折扣策略調整與沉睡客戶喚回。"
)
print("=" * 62)
