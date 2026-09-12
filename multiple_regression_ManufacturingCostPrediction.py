# ===========================================================
#  多重回歸分析 - 智慧機電製造模擬資料 - 製造成本預測
#  x1 : 訂單數量 Order Quantity
#  x2 : 材料成本 Material Cost
#  x3 : 加工工時 Machining Hours
#  x4 : 組裝工時 Assembly Hours
#  x5 : 測試工時 Test Hours
#  x6 : 換線工時 Setup Hours
#  ε  : 生產過程中的隨機誤差與未觀測因素
#  ŷ  : 預測的實際製造成本 Predicted Manufacturing Cost
#  β0 : 截距項 Intercept
#  β1～β6 : 各項製造因素所對應的回歸係數
#  多重回歸概念：#  ŷ = β0 + β1x1 + β2x2 + β3x3  + β4x4 + β5x5 + β6x6 + ε
#  分析目的   ：利用多項已知的生產條件共同預測製造成本，
#  並透過 R²、MAE、RMSE、MAPE 等指標評估模型預測效果。
#  作者 : 國立雲林科技大學電機工程系 林家仁
# ===========================================================
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
    .to_frame("Correlation")
)

print("\n【Correlation with Actual Cost】")
display(corr_target.round(3))

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

bars = ax.bar(
    corr_target.index,
    corr_target["Correlation"],
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
    corr_target["Correlation"]
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

print("\n【Train / Test Split】")
print(f"Training Data：{len(X_train):,} 筆")
print(f"Testing Data ：{len(X_test):,} 筆")

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
    "Metric": [
        "R² Explained Variance",
        "MAPE",
        "MAE",
        "MAE / Avg. Actual",
        "RMSE",
        "RMSE / Avg. Actual"
    ],
    "Result": [
        f"{r2*100:.2f}%",
        f"{mape:.2f}%",
        f"{mae:,.0f} K NTD",
        f"{mae_pct:.2f}%",
        f"{rmse:,.0f} K NTD",
        f"{rmse_pct:.2f}%"
    ],
    "Interpretation": [
        "模型可以解釋多少成本變化",
        "平均百分比預測誤差",
        "平均每筆工單預測錯多少成本",
        "MAE 相對平均實際成本",
        "對大誤差較敏感的誤差指標",
        "RMSE 相對平均實際成本"
    ]
})

print("\n【Model Evaluation】")
display(metrics_df)

# ------------------------------------------------------------
# 9. Actual vs Predicted
# ------------------------------------------------------------
compare_df = pd.DataFrame({
    "Actual_Cost_KNTD":
        y_test.to_numpy(),
    "Predicted_Cost_KNTD":
        np.round(y_pred, 0),
    "Error_KNTD":
        np.round(
            y_test.to_numpy() - y_pred,
            0
        )
})

print("\n【Actual vs Predicted】")

display(
    compare_df.head(15)
    .style.format({
        "Actual_Cost_KNTD":
            "{:,.0f}",
        "Predicted_Cost_KNTD":
            "{:,.0f}",
        "Error_KNTD":
            "{:,.0f}"
    })
)

fig, ax = plt.subplots(
    figsize=(FIG_W, FIG_H)
)

ax.scatter(
    y_test,
    y_pred,
    s=58,
    alpha=0.78,
    color=COLOR_BLUE,
    edgecolors="white",
    linewidths=0.5
)

low = min(
    y_test.min(),
    y_pred.min()
)

high = max(
    y_test.max(),
    y_pred.max()
)

ax.plot(
    [low, high],
    [low, high],
    "--",
    color=COLOR_RED,
    linewidth=2.2,
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

ax.legend()

# 把 X/Y 軸固定在相同範圍，
# 學員會更容易看出點是否貼近 45 度線
ax.set_xlim(low - 100, high + 100)
ax.set_ylim(low - 100, high + 100)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 10. Prediction Error Distribution
# 第一天下午比 Residual Scatter 更容易理解
# ------------------------------------------------------------
errors = (
    y_test.to_numpy()
    - y_pred
)

fig, ax = plt.subplots(
    figsize=(FIG_W, FIG_H)
)

ax.hist(
    errors,
    bins=12,
    color=COLOR_TEAL,
    edgecolor="white",
    alpha=0.85
)

ax.axvline(
    0,
    linestyle="--",
    color=COLOR_RED,
    linewidth=2
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

coef_df = pd.DataFrame({
    "Feature":
        features,
    "Standardized_Coefficient":
        std_model.coef_
})

coef_df["Direction"] = np.where(
    coef_df[
        "Standardized_Coefficient"
    ] >= 0,
    "Positive",
    "Negative"
)

coef_df = coef_df.sort_values(
    "Standardized_Coefficient",
    ascending=False
)

print("\n【Feature Influence】")

display(
    coef_df.style.format({
        "Standardized_Coefficient":
            "{:.3f}"
    })
)

plot_df = coef_df.sort_values(
    "Standardized_Coefficient"
)

fig, ax = plt.subplots(
    figsize=(FIG_W, FIG_H)
)

bars = ax.barh(
    plot_df["Feature"],
    plot_df[
        "Standardized_Coefficient"
    ],
    height=0.52,
    color=COLOR_BLUE
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

for bar, value in zip(
    bars,
    plot_df[
        "Standardized_Coefficient"
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
print("\n【Multiple Regression Equation】")

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

print("\n【New Job Prediction】")

display(new_job)

print(
    "AI Predicted Manufacturing Cost = "
    f"{predicted_cost:,.0f} K NTD"
)

# ------------------------------------------------------------
# 14. 教學摘要
# ------------------------------------------------------------
print()
print("=" * 60)
print("Day 1｜Multiple Regression Summary")
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
