# ============================================================
# YUNFA ERP - Method 02
# K-Means Customer Segmentation
#
# YUNFA ERP - K-Means Customer Segmentation
# Day 2 - Practice 2 
# Prompt Engineering Result
# 透過 ERP 中的「客戶主檔、報價紀錄、銷售訂單」
# 建立客戶層級的行為特徵，進行 K-Means 分群。
#
# 作者：國立雲林科技大學電機系 林家仁
# ============================================================

# 如 Colab 尚未安裝套件，可先執行：
# !pip install -q openpyxl scikit-learn ipywidgets

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from google.colab import files

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ============================================================
# 視覺設定：白底、暖色系
# ============================================================
TEXT_DARK = "#3F3F3F"
PALETTE = ["#F2C6A0", "#A8C69F", "#AFC8E6", "#D7C4E8"]

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "#FFFDFC"
plt.rcParams["axes.edgecolor"] = "#D9D9D9"
plt.rcParams["axes.labelcolor"] = TEXT_DARK
plt.rcParams["xtick.color"] = TEXT_DARK
plt.rcParams["ytick.color"] = TEXT_DARK
plt.rcParams["text.color"] = TEXT_DARK
plt.rcParams["grid.color"] = "#ECE7E1"
plt.rcParams["grid.alpha"] = 0.75
plt.rcParams["font.style"] = "normal"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11


# ============================================================
# STEP 1｜Upload ERP Excel
# ============================================================
uploaded = files.upload()
file_name = next(iter(uploaded))
excel_bytes = io.BytesIO(uploaded[file_name])

print(f"已載入檔案：{file_name}")


# ============================================================
# STEP 2｜Read ERP Sheets
# ============================================================
customer_df = pd.read_excel(excel_bytes, sheet_name="客戶主檔")

excel_bytes.seek(0)
quote_df = pd.read_excel(excel_bytes, sheet_name="報價紀錄")

excel_bytes.seek(0)
order_df = pd.read_excel(excel_bytes, sheet_name="銷售訂單")

print("\n【客戶主檔】")
display(
    customer_df.head().style
    .set_properties(**{
        "background-color": "#FFFDFC",
        "color": TEXT_DARK,
        "border-color": "#E8DDD2"
    })
    .set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#F7E3CF"),
            ("color", TEXT_DARK),
            ("font-weight", "bold"),
            ("border", "1px solid #E8DDD2")
        ]}
    ])
)

print("\n【報價紀錄】")
quote_display = quote_df.head().copy()
quote_format = {}
for c in ["預估毛利率", "折扣率"]:
    if c in quote_display.columns:
        quote_format[c] = "{:.2%}"

display(
    quote_display.style
    .format(quote_format)
    .set_properties(**{
        "background-color": "#FFFDFC",
        "color": TEXT_DARK,
        "border-color": "#E8DDD2"
    })
    .set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#DDEAD9"),
            ("color", TEXT_DARK),
            ("font-weight", "bold"),
            ("border", "1px solid #D8E2D2")
        ]}
    ])
)

print("\n【銷售訂單】")
display(
    order_df.head().style
    .set_properties(**{
        "background-color": "#FFFDFC",
        "color": TEXT_DARK,
        "border-color": "#E8DDD2"
    })
    .set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#E7E1F2"),
            ("color", TEXT_DARK),
            ("font-weight", "bold"),
            ("border", "1px solid #DDD4E8")
        ]}
    ])
)


# ============================================================
# STEP 3｜Data Cleaning
# ============================================================
quote_df["報價日期"] = pd.to_datetime(quote_df["報價日期"], errors="coerce")
order_df["下單日期"] = pd.to_datetime(order_df["下單日期"], errors="coerce")

quote_df["折扣率"] = pd.to_numeric(quote_df["折扣率"], errors="coerce")
quote_df["預估毛利率"] = pd.to_numeric(quote_df["預估毛利率"], errors="coerce")
order_df["訂購數量"] = pd.to_numeric(order_df["訂購數量"], errors="coerce")
order_df["成交單價_TWD"] = pd.to_numeric(order_df["成交單價_TWD"], errors="coerce")

order_df["訂單金額_TWD"] = (
    order_df["訂購數量"] * order_df["成交單價_TWD"]
)

reference_date = max(
    quote_df["報價日期"].max(),
    order_df["下單日期"].max()
)


# ============================================================
# STEP 4｜Customer-Level Feature Engineering
# ============================================================
# 報價行為
quote_agg = quote_df.groupby("客戶ID").agg(
    報價次數=("報價ID", "count"),
    成交報價次數=("結果", lambda x: (x == "成交").sum()),
    平均折扣率=("折扣率", "mean"),
    平均預估毛利率=("預估毛利率", "mean")
)

quote_agg["報價成交率"] = (
    quote_agg["成交報價次數"] /
    quote_agg["報價次數"].replace(0, np.nan)
)

# 訂單行為
order_agg = order_df.groupby("客戶ID").agg(
    總採購金額_TWD=("訂單金額_TWD", "sum"),
    訂單次數=("訂單ID", "count"),
    平均訂單金額_TWD=("訂單金額_TWD", "mean"),
    最近交易日=("下單日期", "max")
)

order_agg["最近交易天數"] = (
    reference_date - order_agg["最近交易日"]
).dt.days

# 依目前 ERP 約 3 年資料期間，換算成年化採購金額
order_agg["年化採購金額_TWD"] = (
    order_agg["總採購金額_TWD"] / 3
)

# 整合成客戶層級資料
customer_feature_df = (
    customer_df.rename(columns={"Customer_ID": "客戶ID"})
    .merge(
        quote_agg.reset_index(),
        on="客戶ID",
        how="left"
    )
    .merge(
        order_agg.reset_index(),
        on="客戶ID",
        how="left"
    )
)

# 缺值處理
numeric_fill_zero = [
    "報價次數",
    "成交報價次數",
    "平均折扣率",
    "平均預估毛利率",
    "報價成交率",
    "總採購金額_TWD",
    "訂單次數",
    "平均訂單金額_TWD",
    "最近交易天數",
    "年化採購金額_TWD",
]

for col in numeric_fill_zero:
    customer_feature_df[col] = pd.to_numeric(
        customer_feature_df[col],
        errors="coerce"
    ).fillna(0)


# ============================================================
# STEP 5｜Define K-Means Features
# ============================================================
FEATURES = [
    "年化採購金額_TWD",
    "訂單次數",
    "平均訂單金額_TWD",
    "報價次數",
    "報價成交率",
    "平均折扣率",
    "平均預估毛利率",
    "最近交易天數",
]

print("\n【K-Means 分群特徵】")
feature_desc = pd.DataFrame({
    "變數": FEATURES,
    "商業意義": [
        "客戶每年帶來的採購金額",
        "客戶下單頻率",
        "每張訂單的平均交易規模",
        "業務與客戶之間的報價互動頻率",
        "報價轉換成訂單的比例",
        "客戶平均取得的價格折扣",
        "報價階段的平均預估毛利",
        "距離最近一次交易的天數"
    ]
})
display(feature_desc)

print("\n注意：客戶等級、產業別、銷售區域不直接放入 K-Means，")
print("它們保留在分群完成後，用來解讀每一群的客戶輪廓。")


# ============================================================
# STEP 6｜Standardization
# ============================================================
X = customer_feature_df[FEATURES].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# STEP 7｜K-Means Clustering
# ============================================================
# 本課程設定 4 群，對應後續企業客戶經營策略。
N_CLUSTERS = 4

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=20
)

customer_feature_df["Cluster"] = kmeans.fit_predict(X_scaled)

silhouette = silhouette_score(
    X_scaled,
    customer_feature_df["Cluster"]
)

print("\n" + "=" * 72)
print("YUNFA ERP－K-Means 客戶分群分析")
print("=" * 72)
print(f"客戶數量       ：{len(customer_feature_df)}")
print(f"分群數量       ：{N_CLUSTERS}")
print(f"Silhouette Score：{silhouette:.3f}")


# ============================================================
# STEP 8｜Cluster Profiling
# ============================================================
profile = customer_feature_df.groupby("Cluster")[FEATURES].mean()

# 用標準化後的群組中心進行「商業輪廓命名」
center_df = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=FEATURES,
    index=range(N_CLUSTERS)
)

remaining = set(range(N_CLUSTERS))
cluster_name_map = {}

# 1. 沉睡／流失風險：最近交易天數最高，且採購活躍度偏低
risk_score = (
    center_df["最近交易天數"]
    - 0.50 * center_df["年化採購金額_TWD"]
    - 0.50 * center_df["訂單次數"]
)
risk_cluster = risk_score.idxmax()
cluster_name_map[risk_cluster] = "沉睡／流失風險"
remaining.remove(risk_cluster)

# 2. VIP 核心：採購金額、訂單次數、報價互動高，且最近有交易
vip_score = (
    center_df["年化採購金額_TWD"]
    + center_df["訂單次數"]
    + center_df["報價次數"]
    - center_df["最近交易天數"]
)
vip_cluster = vip_score.loc[list(remaining)].idxmax()
cluster_name_map[vip_cluster] = "VIP核心"
remaining.remove(vip_cluster)

# 3. 價格敏感：折扣較高、毛利偏低
price_score = (
    center_df["平均折扣率"]
    - center_df["平均預估毛利率"]
)
price_cluster = price_score.loc[list(remaining)].idxmax()
cluster_name_map[price_cluster] = "價格敏感"
remaining.remove(price_cluster)

# 4. 剩餘群組定義為成長潛力
growth_cluster = list(remaining)[0]
cluster_name_map[growth_cluster] = "成長潛力"

customer_feature_df["客群名稱"] = (
    customer_feature_df["Cluster"].map(cluster_name_map)
)

cluster_order = ["VIP核心", "成長潛力", "價格敏感", "沉睡／流失風險"]

print("\n【各客群資料筆數】")
cluster_count = (
    customer_feature_df["客群名稱"]
    .value_counts()
    .reindex(cluster_order)
    .fillna(0)
    .astype(int)
    .reset_index()
)
cluster_count.columns = ["客群名稱", "客戶數"]

display(
    cluster_count.style
    .set_properties(**{
        "background-color": "#FFFDFC",
        "color": TEXT_DARK,
        "border-color": "#E8DDD2"
    })
    .set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#F5D778"),
            ("color", TEXT_DARK),
            ("font-weight", "bold"),
            ("border", "1px solid #E7CF7A")
        ]}
    ])
)


# ============================================================
# STEP 9｜Cluster Profile Table
# ============================================================
profile_table = (
    customer_feature_df
    .groupby("客群名稱")
    .agg(
        客戶數=("客戶ID", "count"),
        年化採購金額_TWD=("年化採購金額_TWD", "mean"),
        訂單次數=("訂單次數", "mean"),
        平均訂單金額_TWD=("平均訂單金額_TWD", "mean"),
        報價次數=("報價次數", "mean"),
        報價成交率=("報價成交率", "mean"),
        平均折扣率=("平均折扣率", "mean"),
        平均預估毛利率=("平均預估毛利率", "mean"),
        最近交易天數=("最近交易天數", "mean")
    )
    .reindex(cluster_order)
    .reset_index()
)

print("\n【客戶分群輪廓】")
display(
    profile_table.style
    .format({
        "年化採購金額_TWD": "{:,.0f}",
        "訂單次數": "{:.1f}",
        "平均訂單金額_TWD": "{:,.0f}",
        "報價次數": "{:.1f}",
        "報價成交率": "{:.2%}",
        "平均折扣率": "{:.2%}",
        "平均預估毛利率": "{:.2%}",
        "最近交易天數": "{:.1f}",
    })
    .set_properties(**{
        "background-color": "#FFFDFC",
        "color": TEXT_DARK,
        "border-color": "#E8DDD2"
    })
    .set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#DDEAD9"),
            ("color", TEXT_DARK),
            ("font-weight", "bold"),
            ("border", "1px solid #D8E2D2")
        ]}
    ])
)


# ============================================================
# STEP 10｜Customer Result Table
# ============================================================
strategy_map = {
    "VIP核心": "維持關係、優先服務、交叉銷售與長期合作",
    "成長潛力": "提高成交率、增加產品組合與業務接觸",
    "價格敏感": "管理折扣、檢查毛利、採差異化報價策略",
    "沉睡／流失風險": "優先喚回、追蹤未成交原因與近期需求"
}

customer_feature_df["建議策略"] = (
    customer_feature_df["客群名稱"].map(strategy_map)
)

result_cols = [
    "客戶ID",
    "客戶名稱",
    "產業別",
    "銷售區域",
    "客戶等級",
    "年化採購金額_TWD",
    "訂單次數",
    "平均訂單金額_TWD",
    "報價次數",
    "報價成交率",
    "平均折扣率",
    "平均預估毛利率",
    "最近交易天數",
    "客群名稱",
    "建議策略"
]

customer_result = customer_feature_df[result_cols].copy()

print("\n【客戶分群結果】")

segment_table_colors = {
    "VIP核心": {
        "header": "#F7D8B5",
        "body": "#FFF8F1",
        "border": "#EBC9A4"
    },
    "成長潛力": {
        "header": "#D9EAD3",
        "body": "#F7FBF5",
        "border": "#C6DDBF"
    },
    "價格敏感": {
        "header": "#D9EAF7",
        "body": "#F5FAFD",
        "border": "#C8DCEA"
    },
    "沉睡／流失風險": {
        "header": "#E7DDF2",
        "body": "#FAF7FD",
        "border": "#D5C8E5"
    }
}

for segment in cluster_order:
    segment_df = (
        customer_result[customer_result["客群名稱"] == segment]
        .sort_values("年化採購金額_TWD", ascending=False)
        .reset_index(drop=True)
    )

    colors = segment_table_colors[segment]

    print(f"\n【{segment}】 共 {len(segment_df)} 位客戶")

    display(
        segment_df.style
        .format({
            "年化採購金額_TWD": "{:,.0f}",
            "平均訂單金額_TWD": "{:,.0f}",
            "報價成交率": "{:.2%}",
            "平均折扣率": "{:.2%}",
            "平均預估毛利率": "{:.2%}",
        })
        .set_properties(**{
            "background-color": colors["body"],
            "color": TEXT_DARK,
            "border-color": colors["border"]
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", colors["header"]),
                ("color", TEXT_DARK),
                ("font-weight", "bold"),
                ("font-style", "normal"),
                ("border", f"1px solid {colors['border']}")
            ]},
            {"selector": "td", "props": [
                ("font-style", "normal")
            ]}
        ])
    )


# ============================================================
# STEP 11｜PCA 2D Visualization
# ============================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "ClusterName": customer_feature_df["客群名稱"].values
})

name_en = {
    "VIP核心": "Core VIP",
    "成長潛力": "Growth Potential",
    "價格敏感": "Price Sensitive",
    "沉睡／流失風險": "Dormant / Churn Risk"
}

color_map = {
    "VIP核心": PALETTE[0],
    "成長潛力": PALETTE[1],
    "價格敏感": PALETTE[2],
    "沉睡／流失風險": PALETTE[3]
}

plt.figure(figsize=(11, 7))

for group in cluster_order:
    part = plot_df[plot_df["ClusterName"] == group]
    plt.scatter(
        part["PC1"],
        part["PC2"],
        s=85,
        alpha=0.82,
        label=name_en[group],
        color=color_map[group],
        edgecolors="white",
        linewidths=0.8
    )

plt.xlabel("Principal Component 1", fontstyle="normal")
plt.ylabel("Principal Component 2", fontstyle="normal")
plt.title(
    "ERP Customer Segmentation - K-Means",
    fontsize=15,
    fontweight="bold",
    fontstyle="normal"
)
plt.xticks(fontstyle="normal")
plt.yticks(fontstyle="normal")
plt.legend(frameon=False, prop={"style": "normal"})
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# STEP 12｜Cluster Size Visualization
# ============================================================
size_plot = (
    customer_feature_df["客群名稱"]
    .value_counts()
    .reindex(cluster_order)
)

plt.figure(figsize=(10, 6))
bars = plt.bar(
    [name_en[x] for x in size_plot.index],
    size_plot.values,
    color=[color_map[x] for x in size_plot.index],
    edgecolor="white",
    linewidth=1
)

for bar, value in zip(bars, size_plot.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.25,
        f"{int(value)}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.xlabel("Customer Segment", fontstyle="normal")
plt.ylabel("Number of Customers", fontstyle="normal")
plt.title(
    "Customer Segment Distribution",
    fontsize=14,
    fontweight="bold",
    fontstyle="normal"
)
plt.xticks(rotation=0, fontstyle="normal")
plt.yticks(fontstyle="normal")
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 13｜Normalized Cluster Profile
# ============================================================
profile_en = {
    "年化採購金額_TWD": "Annual Purchase",
    "訂單次數": "Order Frequency",
    "平均訂單金額_TWD": "Avg. Order",
    "報價次數": "Quote Count",
    "報價成交率": "Conversion",
    "平均折扣率": "Discount",
    "平均預估毛利率": "Gross Margin",
    "最近交易天數": "Recency"
}

z_df = pd.DataFrame(
    X_scaled,
    columns=FEATURES
)
z_df["客群名稱"] = customer_feature_df["客群名稱"].values

z_profile = (
    z_df.groupby("客群名稱")[FEATURES]
    .mean()
    .reindex(cluster_order)
)

fig, ax = plt.subplots(figsize=(14, 6.5))

x = np.arange(len(FEATURES))
width = 0.18

for i, group in enumerate(cluster_order):
    ax.bar(
        x + (i - 1.5) * width,
        z_profile.loc[group].values,
        width=width,
        label=name_en[group],
        color=color_map[group],
        edgecolor="white"
    )

ax.axhline(0, color="#888888", linewidth=1, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(
    [profile_en[f] for f in FEATURES],
    rotation=0,
    ha="center",
    fontstyle="normal",
    fontsize=9
)
ax.set_ylabel("Standardized Cluster Mean", fontstyle="normal")
ax.set_title(
    "Customer Segment Profile",
    fontsize=14,
    fontweight="bold",
    fontstyle="normal"
)
for label in ax.get_xticklabels():
    label.set_fontstyle("normal")
for label in ax.get_yticklabels():
    label.set_fontstyle("normal")
ax.legend(frameon=False, ncol=2, prop={"style": "normal"})
ax.grid(axis="y")
plt.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.90,
    bottom=0.16
)
plt.show()


# ============================================================
# STEP 14｜AI Agent Decision Meaning
# ============================================================
print("\n【AI Agent 決策意義】")
print("VIP核心：維持關係、優先服務、交叉銷售與長期合作。")
print("成長潛力：提高成交率、增加產品組合與業務接觸。")
print("價格敏感：管理折扣、檢查毛利，採差異化報價策略。")
print("沉睡／流失風險：優先喚回，追蹤未成交原因與近期需求。")

print("\nK-Means 的角色不是直接替企業做決策，")
print("而是把具有相似交易行為的客戶自動聚集成群，")
print("再提供給後續智慧看板與 AI Agent 作為差異化策略的依據。")
