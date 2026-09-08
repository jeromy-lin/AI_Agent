# ===========================================================
# 主題：Regression 迴歸分析－智慧製造
# 目標：利用機器學習預測生產工單完成時間
#
# 模擬智慧製造資料：
# x1：生產數量 Production Quantity
# x2：機台負載 Machine Load
# x3：換線複雜度 Setup Complexity
# x4：材料加工難度 Material Difficulty
# x5：操作員經驗 Operator Experience
#
# y ：生產完成時間 Production Time
#
# 作者：國立雲林科技大學電機系 林家仁
# ===========================================================


# ===========================================================
# 0. 載入套件
# ===========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ===========================================================
# 1. 柔和配色設定
# ===========================================================

# 表格用柔和色
COLOR_BLUE   = "#A9C5E8"
COLOR_ORANGE = "#F6C28B"
COLOR_RED    = "#EFA6A6"
COLOR_TEAL   = "#A8D5D1"
COLOR_GREEN  = "#B8D8B0"
COLOR_YELLOW = "#F3DFA2"
COLOR_PURPLE = "#CDB7D9"
COLOR_PINK   = "#F4C6D7"
COLOR_BROWN  = "#D4B6A5"
COLOR_GRAY   = "#D6D6D6"

# 圖表散點使用較深藍色
CHART_BLUE = "#5B8FC9"

TEXT_DARK = "#4A4A4A"
GRID_COLOR = "#E6E6E6"


# 圖表基本設定
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 11


# ===========================================================
# 2. 產生智慧製造模擬資料
# ===========================================================

np.random.seed(0)

data_size = 150


# x1：生產數量
production_quantity = np.random.normal(
    120,
    35,
    data_size
)

production_quantity = np.maximum(
    20,
    production_quantity
)


# x2：機台負載率
machine_load = np.random.normal(
    0.75,
    0.12,
    data_size
)

machine_load = np.clip(
    machine_load,
    0.40,
    1.00
)


# x3：換線 / 換模複雜度
# 1 = 簡單
# 2 = 普通
# 3 = 複雜

setup_complexity = np.random.choice(
    [1, 2, 3],
    size=data_size,
    p=[0.4, 0.4, 0.2]
)


# x4：材料加工難度
# 1 = 容易
# 2 = 一般
# 3 = 困難

material_difficulty = np.random.choice(
    [1, 2, 3],
    size=data_size,
    p=[0.35, 0.45, 0.20]
)


# x5：操作員經驗
# 1 = 新手
# 2 = 初階
# 3 = 一般
# 4 = 熟練
# 5 = 資深

operator_experience = np.random.choice(
    [1, 2, 3, 4, 5],
    size=data_size,
    p=[0.10, 0.20, 0.35, 0.25, 0.10]
)


# ===========================================================
# 3. 建立模擬生產時間
# ===========================================================

production_time = (
    30
    + 1.8 * production_quantity
    + 80 * machine_load
    + 25 * setup_complexity
    + 18 * material_difficulty
    - 10 * operator_experience
    + np.random.normal(
        0,
        15,
        data_size
    )
)

production_time = np.maximum(
    0,
    production_time
)


# ===========================================================
# 4. 建立 DataFrame
# ===========================================================

df = pd.DataFrame({

    "生產數量":
        production_quantity.round(0).astype(int),

    "機台負載率":
        machine_load.round(2),

    "換線複雜度":
        setup_complexity,

    "材料加工難度":
        material_difficulty,

    "操作員經驗":
        operator_experience,

    "實際生產時間":
        production_time.round(1)

})


# ===========================================================
# 5. 顯示模擬資料
# ===========================================================

print("【智慧製造模擬資料】")
print(f"共產生 {data_size} 筆生產工單\n")


display(

    df.head(10).style

    .set_properties(**{
        "background-color": "#FBFCFE",
        "color": TEXT_DARK,
        "border-color": "#E8EDF3",
        "text-align": "center"
    })

    .set_table_styles([

        {
            "selector": "th",
            "props": [
                ("background-color", COLOR_BLUE),
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

    .format({
        "機台負載率": "{:.2f}",
        "實際生產時間": "{:.1f}"
    })

)


# ===========================================================
# 6. 準備 Machine Learning 資料
# ===========================================================

X = df[[
    "生產數量",
    "機台負載率",
    "換線複雜度",
    "材料加工難度",
    "操作員經驗"
]]

y = df[
    "實際生產時間"
]


# ===========================================================
# 7. 切分訓練與測試資料
# ===========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)


print(
    f"訓練資料：{len(X_train)} 筆 ｜ "
    f"測試資料：{len(X_test)} 筆"
)


# ===========================================================
# 8. 建立 Linear Regression 模型
# ===========================================================

model = LinearRegression()


# ===========================================================
# 9. 訓練模型
# ===========================================================

model.fit(
    X_train,
    y_train
)


# ===========================================================
# 10. AI 預測
# ===========================================================

y_pred = model.predict(
    X_test
)

y_pred = np.maximum(
    0,
    y_pred
)


# ===========================================================
# 11. Regression Coefficients
# ===========================================================

coef_df = pd.DataFrame({

    "影響因素": [
        "生產數量",
        "機台負載率",
        "換線複雜度",
        "材料加工難度",
        "操作員經驗"
    ],

    "迴歸係數 β":
        model.coef_.round(2)

})


print("\n【各因素對生產時間的影響】")


display(

    coef_df.style

    .set_properties(**{
        "background-color": "#FFFDF9",
        "color": TEXT_DARK,
        "border-color": "#F4E8D8",
        "text-align": "center"
    })

    .set_table_styles([

        {
            "selector": "th",
            "props": [
                ("background-color", COLOR_ORANGE),
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

    .format({
        "迴歸係數 β": "{:.2f}"
    })

)


print(
    f"Intercept：{model.intercept_:.2f}"
)

print(
    "正係數：增加生產時間 ｜ "
    "負係數：降低生產時間"
)


# ===========================================================
# 12. 模型評估
# ===========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mse
)

r2 = r2_score(
    y_test,
    y_pred
)


# ===========================================================
# 13. 顯示部分 AI 預測結果
# ===========================================================

result = X_test.copy()

result["實際時間"] = (
    y_test.values.round(1)
)

result["AI預測時間"] = (
    y_pred.round(1)
)

result["誤差"] = (
    result["實際時間"]
    -
    result["AI預測時間"]
).round(1)


print("\n【部分工單預測結果】")


display(

    result.head(10).style

    .set_properties(**{
        "background-color": "#FFFDFD",
        "color": TEXT_DARK,
        "border-color": "#F3E4E4",
        "text-align": "center"
    })

    .set_table_styles([

        {
            "selector": "th",
            "props": [
                ("background-color", COLOR_RED),
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


# ===========================================================
# 14. 圖表 1
# Actual vs Predicted Production Time
# ===========================================================

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    y_test,
    y_pred,
    color=CHART_BLUE,
    alpha=0.92,
    s=78,
    edgecolors="white",
    linewidths=0.7
)


plt.plot(

    [
        y_test.min(),
        y_test.max()
    ],

    [
        y_test.min(),
        y_test.max()
    ],

    linestyle="--",

    color=COLOR_RED,

    linewidth=2,

    label="Ideal Prediction"

)


plt.xlabel(
    "Actual Production Time (minutes)"
)

plt.ylabel(
    "Predicted Production Time (minutes)"
)

plt.title(
    "Actual vs Predicted Production Time"
)

plt.legend()

plt.grid(
    color=GRID_COLOR,
    alpha=0.6
)

plt.tight_layout()

plt.show()


print(
    "圖 1：資料點越接近虛線，代表 AI 預測越準確。"
)


# ===========================================================
# 15. 圖表 2
# Regression Coefficients
# ===========================================================

feature_names = [

    "Production\nQuantity",

    "Machine\nLoad",

    "Setup\nComplexity",

    "Material\nDifficulty",

    "Operator\nExperience"

]


feature_colors = [

    COLOR_BLUE,

    COLOR_ORANGE,

    COLOR_RED,

    COLOR_TEAL,

    COLOR_GREEN

]


plt.figure(
    figsize=(10, 6)
)


bars = plt.bar(

    feature_names,

    model.coef_,

    color=feature_colors,

    edgecolor="white",

    linewidth=1

)


plt.axhline(
    0,
    color="#999999",
    linewidth=1
)


plt.ylabel(
    "Regression Coefficient"
)

plt.title(
    "Impact of Manufacturing Factors on Production Time"
)


plt.grid(
    axis="y",
    color=GRID_COLOR,
    alpha=0.6
)


for bar, value in zip(
    bars,
    model.coef_
):

    if value >= 0:

        plt.text(

            bar.get_x()
            + bar.get_width() / 2,

            value + 1,

            f"{value:.1f}",

            ha="center",

            va="bottom",

            fontsize=10,

            color=TEXT_DARK

        )

    else:

        plt.text(

            bar.get_x()
            + bar.get_width() / 2,

            value - 1,

            f"{value:.1f}",

            ha="center",

            va="top",

            fontsize=10,

            color=TEXT_DARK

        )


plt.tight_layout()

plt.show()


print(
    "圖 2：正值代表增加生產時間；負值代表降低生產時間。"
)


# ===========================================================
# 16. 圖表 3
# Prediction Error Distribution
# ===========================================================

errors = (
    y_test.values
    -
    y_pred
)


plt.figure(
    figsize=(10, 6)
)


plt.hist(

    errors,

    bins=15,

    color=COLOR_TEAL,

    edgecolor="white",

    linewidth=1

)


plt.axvline(

    0,

    linestyle="--",

    color=COLOR_RED,

    linewidth=2

)


plt.xlabel(
    "Prediction Error (minutes)"
)

plt.ylabel(
    "Number of Orders"
)

plt.title(
    "Distribution of Prediction Errors"
)


plt.grid(
    axis="y",
    color=GRID_COLOR,
    alpha=0.6
)

plt.tight_layout()

plt.show()


print(
    "圖 3：誤差越集中在 0 附近，代表模型預測越穩定。"
)


# ===========================================================
# 17. 圖表 4
# Production Quantity vs Production Time
# ===========================================================

plt.figure(
    figsize=(10, 6)
)


plt.scatter(

    df["生產數量"],

    df["實際生產時間"],

    color=COLOR_ORANGE,

    alpha=0.82,

    s=62,

    edgecolors="white",

    linewidths=0.6

)


trend = np.polyfit(

    df["生產數量"],

    df["實際生產時間"],

    1

)


trend_line = np.poly1d(
    trend
)


x_line = np.linspace(

    df["生產數量"].min(),

    df["生產數量"].max(),

    100

)


plt.plot(

    x_line,

    trend_line(x_line),

    color=COLOR_RED,

    linewidth=2,

    linestyle="--",

    label="Trend Line"

)


plt.xlabel(
    "Production Quantity"
)

plt.ylabel(
    "Production Time (minutes)"
)

plt.title(
    "Production Quantity vs Production Time"
)

plt.legend()

plt.grid(
    color=GRID_COLOR,
    alpha=0.6
)

plt.tight_layout()

plt.show()


print(
    "圖 4：生產數量增加時，整體生產時間通常也會增加。"
)


# ===========================================================
# 18. 最終模型評估結果表
# ===========================================================

summary_df = pd.DataFrame({

    "評估指標": [
        "MAE",
        "RMSE",
        "R²"
    ],

    "模型結果": [
        f"{mae:.2f} 分鐘",
        f"{rmse:.2f} 分鐘",
        f"{r2:.3f}"
    ],

    "指標意義": [
        "AI 預測時間與實際時間的平均差距",
        "反映整體預測誤差，對較大的誤差較敏感",
        "模型可以解釋生產時間變化的程度"
    ],

    "本次結果解讀": [
        f"平均每張工單約誤差 {mae:.1f} 分鐘",
        f"整體預測誤差約 {rmse:.1f} 分鐘",
        f"模型可解釋約 {r2 * 100:.1f}% 的生產時間變化"
    ]

})


# ===========================================================
# 19. 最終表格柔和配色
# ===========================================================

def highlight_metric(row):

    metric = row["評估指標"]

    if metric == "MAE":

        return [
            "background-color: #EEF5FC; color: #4A4A4A;"
        ] * len(row)

    elif metric == "RMSE":

        return [
            "background-color: #FFF5E9; color: #4A4A4A;"
        ] * len(row)

    elif metric == "R²":

        return [
            "background-color: #F0F7ED; color: #4A4A4A;"
        ] * len(row)

    return [""] * len(row)


print("\n【Regression 智慧製造分析結果】")


display(

    summary_df.style

    .apply(
        highlight_metric,
        axis=1
    )

    .set_table_styles([

        {
            "selector": "th",
            "props": [

                ("background-color", COLOR_BLUE),

                ("color", TEXT_DARK),

                ("font-weight", "bold"),

                ("text-align", "center"),

                ("padding", "12px"),

                ("border", "3px solid white")

            ]
        },

        {
            "selector": "td",
            "props": [

                ("padding", "12px"),

                ("border", "3px solid white"),

                ("text-align", "center"),

                ("font-size", "14px")

            ]
        }

    ])

    .set_properties(

        subset=[
            "指標意義",
            "本次結果解讀"
        ],

        **{
            "text-align": "left"
        }

    )

)


# ===========================================================
# 20. 最後結論
# ===========================================================

print(
    f"模型 R² = {r2:.3f}，"
    f"可解釋約 {r2 * 100:.1f}% 的生產時間變化；"
    f"平均預測誤差約 {mae:.1f} 分鐘。"
)

print(
    "Regression 可用於工單生產時間預測，"
    "支援產能規劃、生產排程與交期預估。"
)
