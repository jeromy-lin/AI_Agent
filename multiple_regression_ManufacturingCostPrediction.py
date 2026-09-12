
# ============================================================
#  主題 : Multiple Regression 多重回歸分析練習
#  目標 : 使學員了解多個製造因素如何共同影響製造成本，
#         並學習模型訓練、預測與基本評估方法
#  情境 : 以智慧機電製造為例，利用訂單數量、材料成本、
#         加工工時、組裝工時、測試工時與換線工時，
#         預測每張工單的實際製造成本
#
#  模擬智慧機電製造資料：
#  x1 : 訂單數量 Order Quantity
#  x2 : 材料成本 Material Cost
#  x3 : 加工工時 Machining Hours
#  x4 : 組裝工時 Assembly Hours
#  x5 : 測試工時 Test Hours
#  x6 : 換線工時 Setup Hours
#  ε  : 生產過程中的隨機誤差與未觀測因素
#
#  ŷ  : 預測的實際製造成本 Predicted Manufacturing Cost
#  β0 : 截距項 Intercept
#  β1～β6 : 各項製造因素所對應的回歸係數
#
#  多重回歸概念：
#  ŷ = β0 + β1x1 + β2x2 + β3x3
#        + β4x4 + β5x5 + β6x6 + ε
#
#  作者 : 國立雲林科技大學電機工程系 林家仁
# ============================================================

!pip -q install scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. 配色
# ------------------------------------------------------------
COLOR_BLUE   = "#4E79A7"
COLOR_ORANGE = "#F28E2B"
COLOR_RED    = "#E15759"
COLOR_TEAL   = "#76B7B2"
COLOR_GREEN  = "#59A14F"
COLOR_YELLOW = "#EDC948"
COLOR_PURPLE = "#B07AA1"

FIG_W = 13
FIG_H = 6.1

# ------------------------------------------------------------
# 2. 建立教學用企業資料
# 情境：智慧機電設備製造商，用已知的工單條件預估製造成本
# ------------------------------------------------------------
np.random.seed(42)

N = 180

order_qty = np.random.randint(1, 21, N)

material_cost = (
    np.random.normal(1200, 350, N)
    .clip(450, 2400)
)

machining_hours = (
    np.random.normal(95, 28, N)
    .clip(25, 180)
)

assembly_hours = (
    np.random.normal(70, 22, N)
    .clip(20, 150)
)

test_hours = (
    np.random.normal(38, 14, N)
    .clip(8, 90)
)

setup_hours = (
    np.random.normal(7, 2.5, N)
    .clip(1, 16)
)

# 少量現場波動
# 原本 noise 太大，第一天會讓學員覺得模型效果不明顯。
# 這裡改成 70 K NTD，保留真實感，但模型關係更清楚。
noise = np.random.normal(0, 70, N)

actual_cost = (
    220
    + order_qty * 18
    + material_cost * 0.92
    + machining_hours * 7.0
    + assembly_hours * 5.2
    + test_hours * 6.5
    + setup_hours * 9.5
    + noise
)

df = pd.DataFrame({
    "Order_Qty": order_qty,
    "Material_Cost_KNTD": np.round(material_cost, 0),
    "Machining_Hours": np.round(machining_hours, 1),
    "Assembly_Hours": np.round(assembly_hours, 1),
    "Test_Hours": np.round(test_hours, 1),
    "Setup_Hours": np.round(setup_hours, 1),
    "Actual_Cost_KNTD": np.round(actual_cost, 0)
})

print("=" * 60)
print("Day 1｜Multiple Regression")
print("=" * 60)
print(f"資料筆數：{len(df):,}")
print("X 變數：6 個")
print("Y：Actual Manufacturing Cost")
print()

display(df.head(10))

# ------------------------------------------------------------
# 3. 基本統計
# ------------------------------------------------------------
print("\n【資料統計摘要】")
display(df.describe().round(2))

# ------------------------------------------------------------
# 4. Correlation Analysis
# ------------------------------------------------------------
corr = df.corr(numeric_only=True)

corr_target = (
    corr["Actual_Cost_KNTD"]
    .drop("Actual_Cost_KNTD")
    .sort_values(ascending=False)
    .to_frame("相關係數")
)

print("\n【與實際製造成本的相關性分析】")
display(corr_target.round(3))

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

bars = ax.bar(
    corr_target.index,
    corr_target["相關係數"],
    width=0.52,
    color=[
        COLOR_BLUE,
        COLOR_ORANGE,
        COLOR_TEAL,
        COLOR_GREEN,
        COLOR_PURPLE,
        COLOR_YELLOW
    ]
)

ax.set_title(
    "Correlation with Actual Manufacturing Cost",
    pad=14
)
ax.set_ylabel("Correlation Coefficient")
ax.set_ylim(-1, 1)
ax.grid(axis="y", alpha=0.22)

for bar, value in zip(
    bars,
    corr_target["相關係數"]
):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.03,
        f"{value:.2f}",
        ha="center"
    )

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. X / Y
# ------------------------------------------------------------
features = [
    "Order_Qty",
    "Material_Cost_KNTD",
    "Machining_Hours",
    "Assembly_Hours",
    "Test_Hours",
    "Setup_Hours"
]

target = "Actual_Cost_KNTD"

X = df[features]
y = df[target]

# ------------------------------------------------------------
# 6. Train / Test Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\n【訓練資料與測試資料切分】")
print(f"訓練資料：{len(X_train):,} 筆")
print(f"測試資料：{len(X_test):,} 筆")

# ------------------------------------------------------------
# 7. Multiple Regression
# ------------------------------------------------------------
model = LinearRegression()

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)

# ------------------------------------------------------------
# 8. Model Evaluation
# ------------------------------------------------------------
mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

mape = np.mean(
    np.abs(
        (y_test.to_numpy() - y_pred)
        / y_test.to_numpy()
    )
) * 100

avg_actual = y_test.mean()

mae_pct = (
    mae / avg_actual * 100
)

rmse_pct = (
    rmse / avg_actual * 100
)

metrics_df = pd.DataFrame({
    "評估指標": [
        "R² 解釋能力",
        "MAPE",
        "MAE",
        "MAE / 平均實際成本",
        "RMSE",
        "RMSE / 平均實際成本"
    ],
    "分析結果": [
        f"{r2*100:.2f}%",
        f"{mape:.2f}%",
        f"{mae:,.0f} K NTD",
        f"{mae_pct:.2f}%",
        f"{rmse:,.0f} K NTD",
        f"{rmse_pct:.2f}%"
    ],
    "中文說明": [
        "模型可以解釋多少實際成本的變化",
        "平均百分比預測誤差",
        "平均每筆工單預測誤差金額",
        "MAE 相對於平均實際成本的比例",
        "對較大誤差更敏感的預測誤差指標",
        "RMSE 相對於平均實際成本的比例"
    ]
})

print("\n【模型評估結果】")
display(metrics_df)

# ------------------------------------------------------------
# 9. Actual vs Predicted
#    點越接近紅色虛線，代表預測越接近實際值。
#    本圖另外標示 ±3% 的 Accurate Prediction Zone。
# ------------------------------------------------------------
compare_df = pd.DataFrame({
    "實際成本_KNTD":
        y_test.to_numpy(),
    "預測成本_KNTD":
        np.round(y_pred, 0),
    "預測誤差_KNTD":
        np.round(
            y_test.to_numpy() - y_pred,
            0
        )
})

compare_df["誤差百分比"] = (
    np.abs(
        compare_df["實際成本_KNTD"]
        - compare_df["預測成本_KNTD"]
    )
    / compare_df["實際成本_KNTD"]
    * 100
)

print("\n【實際成本與預測成本比較】")

display(
    compare_df.head(15)
    .style.format({
        "實際成本_KNTD":
            "{:,.0f}",
        "預測成本_KNTD":
            "{:,.0f}",
        "預測誤差_KNTD":
            "{:,.0f}",
        "誤差百分比":
            "{:.2f}%"
    })
)

actual_values = y_test.to_numpy()

error_pct = (
    np.abs(actual_values - y_pred)
    / actual_values
    * 100
)

# 課堂用三段式：
# 綠色：誤差 <= 3%，非常接近趨勢線
# 橘色：3% < 誤差 <= 5%，仍屬合理預測
# 紅色：誤差 > 5%，需要進一步檢查
accurate_mask = error_pct <= 3
acceptable_mask = (
    (error_pct > 3)
    & (error_pct <= 5)
)
review_mask = error_pct > 5

fig, ax = plt.subplots(
    figsize=(FIG_W, FIG_H)
)

# ±3% 準確預測區間
low = min(
    actual_values.min(),
    y_pred.min()
)

high = max(
    actual_values.max(),
    y_pred.max()
)

line_x = np.linspace(
    low,
    high,
    200
)

ax.fill_between(
    line_x,
    line_x * 0.97,
    line_x * 1.03,
    color=COLOR_GREEN,
    alpha=0.10,
    label="Accurate Prediction Zone (±3%)"
)

# 不同預測誤差以不同顏色顯示
ax.scatter(
    actual_values[accurate_mask],
    y_pred[accurate_mask],
    s=62,
    alpha=0.85,
    color=COLOR_GREEN,
    edgecolors="white",
    linewidths=0.6,
    label="Accurate (Error ≤ 3%)"
)

ax.scatter(
    actual_values[acceptable_mask],
    y_pred[acceptable_mask],
    s=62,
    alpha=0.85,
    color=COLOR_ORANGE,
    edgecolors="white",
    linewidths=0.6,
    label="Acceptable (3% < Error ≤ 5%)"
)

ax.scatter(
    actual_values[review_mask],
    y_pred[review_mask],
    s=62,
    alpha=0.85,
    color=COLOR_RED,
    edgecolors="white",
    linewidths=0.6,
    label="Review (Error > 5%)"
)

# Perfect Prediction = Actual = Predicted
ax.plot(
    [low, high],
    [low, high],
    "--",
    color="#555555",
    linewidth=2.0,
    label="Perfect Prediction"
)

ax.set_xlabel(
    "Actual Cost (K NTD)"
)

ax.set_ylabel(
    "Predicted Cost (K NTD)"
)

ax.set_title(
    "Actual vs Predicted Manufacturing Cost",
    pad=14
)

ax.grid(
    alpha=0.22
)

ax.set_xlim(
    low - 100,
    high + 100
)

ax.set_ylim(
    low - 100,
    high + 100
)

# 圖上的教學說明
accurate_rate = (
    accurate_mask.sum()
    / len(error_pct)
    * 100
)

ax.text(
    0.03,
    0.95,
    "Closer to the diagonal line = More accurate prediction",
    transform=ax.transAxes,
    fontsize=11,
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor=COLOR_GREEN,
        alpha=0.90
    )
)

ax.text(
    0.03,
    0.86,
    f"Predictions within ±3%: {accurate_mask.sum()}/{len(error_pct)} ({accurate_rate:.1f}%)",
    transform=ax.transAxes,
    fontsize=10,
    va="top"
)

ax.legend(
    loc="lower right",
    fontsize=9
)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10. Prediction Error Distribution
#     使用不同顏色呈現各誤差區間，讓分布更容易閱讀。
# ------------------------------------------------------------
errors = (
    y_test.to_numpy()
    - y_pred
)

fig, ax = plt.subplots(
    figsize=(FIG_W, FIG_H)
)

counts, bins, patches = ax.hist(
    errors,
    bins=12,
    edgecolor="white",
    linewidth=1.0,
    alpha=0.88
)

hist_colors = [
    "#B07AA1",
    "#9C8BC4",
    "#76B7B2",
    "#59A14F",
    "#8CD17D",
    "#EDC948",
    "#F2CF5B",
    "#F28E2B",
    "#FF9DA7",
    "#E15759",
    "#D37295",
    "#AF7AA1"
]

for patch, color in zip(
    patches,
    hist_colors
):
    patch.set_facecolor(color)

ax.axvline(
    0,
    linestyle="--",
    color="#555555",
    linewidth=2,
    label="Zero Error"
)

ax.set_xlabel(
    "Prediction Error (K NTD)"
)

ax.set_ylabel(
    "Number of Jobs"
)

ax.set_title(
    "Prediction Error Distribution",
    pad=14
)

ax.grid(
    axis="y",
    alpha=0.22
)

ax.legend()

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 11. Standardized Coefficients
# ------------------------------------------------------------
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train_std = scaler_x.fit_transform(
    X_train
)

y_train_std = scaler_y.fit_transform(
    y_train.to_numpy()
    .reshape(-1, 1)
).ravel()

std_model = LinearRegression()

std_model.fit(
    X_train_std,
    y_train_std
)

feature_name_zh = {
    "Order_Qty": "訂單數量",
    "Material_Cost_KNTD": "材料成本",
    "Machining_Hours": "加工工時",
    "Assembly_Hours": "組裝工時",
    "Test_Hours": "測試工時",
    "Setup_Hours": "換線工時"
}

coef_df = pd.DataFrame({
    "Feature":
        features,
    "特徵名稱":
        [feature_name_zh[f] for f in features],
    "標準化回歸係數":
        std_model.coef_
})

coef_df["影響方向"] = np.where(
    coef_df["標準化回歸係數"] >= 0,
    "正向影響",
    "負向影響"
)

coef_df = coef_df.sort_values(
    "標準化回歸係數",
    ascending=False
)

print("\n【各項因素對製造成本的影響】")

display(
    coef_df[["特徵名稱", "標準化回歸係數", "影響方向"]]
    .style.format({
        "標準化回歸係數":
            "{:.3f}"
    })
)

plot_df = coef_df.sort_values(
    "標準化回歸係數"
)

fig, ax = plt.subplots(
    figsize=(FIG_W, FIG_H)
)

feature_colors = {
    "Material_Cost_KNTD": COLOR_BLUE,
    "Machining_Hours": COLOR_ORANGE,
    "Assembly_Hours": COLOR_TEAL,
    "Order_Qty": COLOR_GREEN,
    "Test_Hours": COLOR_PURPLE,
    "Setup_Hours": COLOR_YELLOW
}

bars = ax.barh(
    plot_df["Feature"],
    plot_df[
        "標準化回歸係數"
    ],
    height=0.52,
    color=[
        feature_colors.get(
            feature,
            COLOR_BLUE
        )
        for feature in plot_df["Feature"]
    ]
)

ax.axvline(
    0,
    color="#333333",
    linewidth=1
)

ax.set_xlabel(
    "Standardized Regression Coefficient"
)

ax.set_title(
    "Feature Influence on Manufacturing Cost",
    pad=14
)

ax.grid(
    axis="x",
    alpha=0.22
)

ax.text(
    0.98,
    0.05,
    "Larger standardized coefficient = Stronger influence",
    transform=ax.transAxes,
    fontsize=10,
    ha="right",
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor=COLOR_BLUE,
        alpha=0.90
    )
)

for bar, value in zip(
    bars,
    plot_df[
        "標準化回歸係數"
    ]
):
    ax.text(
        value + 0.012,
        bar.get_y()
        + bar.get_height()/2,
        f"{value:.2f}",
        va="center",
        ha="left"
    )

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 12. Multiple Regression Equation
# ------------------------------------------------------------
print("\n【多重回歸方程式】")

equation = (
    f"Predicted Cost = "
    f"{model.intercept_:,.2f}"
)

for feature, coef in zip(
    features,
    model.coef_
):
    sign = "+" if coef >= 0 else "-"

    equation += (
        f" {sign} "
        f"{abs(coef):,.2f}"
        f" × {feature}"
    )

print(equation)

# ------------------------------------------------------------
# 13. New Job Prediction
# ------------------------------------------------------------
new_job = pd.DataFrame({
    "Order_Qty": [8],
    "Material_Cost_KNTD": [1450],
    "Machining_Hours": [105],
    "Assembly_Hours": [80],
    "Test_Hours": [42],
    "Setup_Hours": [7]
})

predicted_cost = (
    model.predict(new_job)[0]
)

print("\n【新工單成本預測】")

new_job_display = new_job.rename(columns={
    "Order_Qty": "訂單數量",
    "Material_Cost_KNTD": "材料成本_KNTD",
    "Machining_Hours": "加工工時",
    "Assembly_Hours": "組裝工時",
    "Test_Hours": "測試工時",
    "Setup_Hours": "換線工時"
})

display(new_job_display)

print(
    "AI 預測製造成本 = "
    f"{predicted_cost:,.0f} K NTD"
)

# ------------------------------------------------------------
# 14. 教學摘要
# ------------------------------------------------------------
print()
print("=" * 60)
print("Day 1｜多重回歸分析結果摘要")
print("=" * 60)

print(f"資料筆數          ：{len(df):,}")
print(f"訓練資料          ：{len(X_train):,}")
print(f"測試資料          ：{len(X_test):,}")
print(f"輸入變數數量      ：{len(features)}")

print()
print(f"R² 解釋能力       ：{r2*100:.2f}%")
print(f"MAPE              ：{mape:.2f}%")
print(f"MAE               ：{mae:,.0f} K NTD")
print(f"MAE / 平均實際成本：{mae_pct:.2f}%")
print(f"RMSE              ：{rmse:,.0f} K NTD")
print(f"RMSE / 平均實際成本：{rmse_pct:.2f}%")

print()
print("核心概念：")
print("多重回歸不是只看單一因素，而是同時考慮多個製造條件，")
print("共同預測最後的實際製造成本。")
print("=" * 60)

print(
    f"Data              : {len(df):,}"
)
print(
    f"Training Data     : {len(X_train):,}"
)
print(
    f"Testing Data      : {len(X_test):,}"
)
print(
    f"X Variables       : {len(features)}"
)

print()

print(
    f"R² Explained      : "
    f"{r2*100:.2f}%"
)

print(
    f"MAPE              : "
    f"{mape:.2f}%"
)

print(
    f"MAE               : "
    f"{mae:,.0f} K NTD"
)

print(
    f"MAE / Avg. Actual : "
    f"{mae_pct:.2f}%"
)

print(
    f"RMSE              : "
    f"{rmse:,.0f} K NTD"
)

print(
    f"RMSE / Avg. Actual: "
    f"{rmse_pct:.2f}%"
)

print()
print("核心概念：")
print("多重回歸不是只看一個因素，")
print("而是同時讓多個 X 共同預測一個 Y。")
print("=" * 60)
