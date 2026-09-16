# ============================================================
# YUNFA ERP - Method 01
# Multiple Linear Regression by Product Line
# YUNFA ERP - Multiple Linear Regression by Product Line
# Day 2 - Practice 1 
# Prompt Engineering Result
# 作者 : 國立雲林科技大學電機工程系 林家仁
# ============================================================

# 如 Colab 尚未安裝套件，可先執行：
# !pip install -q openpyxl scikit-learn ipywidgets

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets

from IPython.display import display, clear_output
from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# 暖色系色票
# ============================================================
WARM_BLUE   = "#6FA8DC"   # soft blue
WARM_ORANGE = "#F6B26B"   # warm orange
WARM_GREEN  = "#93C47D"   # soft green
WARM_RED    = "#E06666"   # muted red
WARM_PURPLE = "#B4A7D6"   # soft purple
WARM_GOLD   = "#FFD966"   # light gold
WARM_BEIGE  = "#FCE5CD"   # light beige
WARM_PEACH  = "#F4CCCC"   # light peach
WARM_MINT   = "#D9EAD3"   # light mint
WARM_GRAY   = "#EDEDED"   # soft gray
TEXT_DARK   = "#3F3F3F"

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "#FFFDFC"
plt.rcParams["axes.edgecolor"] = "#D9D9D9"
plt.rcParams["axes.labelcolor"] = TEXT_DARK
plt.rcParams["xtick.color"] = TEXT_DARK
plt.rcParams["ytick.color"] = TEXT_DARK
plt.rcParams["text.color"] = TEXT_DARK
plt.rcParams["grid.color"] = "#EDE7E0"
plt.rcParams["grid.alpha"] = 0.75


# ============================================================
# STEP 1. 上傳 YUNFA ERP Excel
# ============================================================
uploaded = files.upload()
file_name = next(iter(uploaded))
excel_bytes = io.BytesIO(uploaded[file_name])

print(f"已載入檔案：{file_name}")


# ============================================================
# STEP 2. 讀取 ERP 資料表
# ============================================================
product_df = pd.read_excel(excel_bytes, sheet_name="產品主檔")

excel_bytes.seek(0)
cost_df = pd.read_excel(excel_bytes, sheet_name="成本結算")

print("\n【產品主檔】")
product_display = product_df.head().copy()
product_format = {}
if "標準毛利率" in product_display.columns:
    product_format["標準毛利率"] = "{:.2%}"

display(
    product_display.style
    .format(product_format)
    .set_properties(**{
        "background-color": "#FFFDFC",
        "color": TEXT_DARK,
        "border-color": "#E7DDD3"
    })
    .set_table_styles([
        {"selector": "th", "props": [
            ("background-color", WARM_BEIGE),
            ("color", TEXT_DARK),
            ("font-weight", "bold"),
            ("border", "1px solid #E7DDD3")
        ]}
    ])
)

print("\n【成本結算】")
cost_display = cost_df.head().rename(columns={
    "CNC加工工時_hr": "CNC加工工時(hr)",
    "組裝工時_hr": "組裝工時(hr)",
    "測試工時_hr": "測試工時(hr)",
    "換線工時_hr": "換線工時(hr)",
})
cost_format = {}
for col in cost_display.columns:
    if pd.api.types.is_numeric_dtype(cost_display[col]):
        if "(hr)" in str(col):
            cost_format[col] = "{:,.2f}"
        elif "率" in str(col) or "%" in str(col):
            cost_format[col] = "{:.2%}"
        else:
            cost_format[col] = "{:,.0f}"

display(
    cost_display.style
    .format(cost_format)
    .set_properties(**{
        "background-color": "#FFFDFC",
        "color": TEXT_DARK,
        "border-color": "#E7DDD3"
    })
    .set_table_styles([
        {"selector": "th", "props": [
            ("background-color", WARM_MINT),
            ("color", TEXT_DARK),
            ("font-weight", "bold"),
            ("border", "1px solid #D8E2D2")
        ]}
    ])
)


# ============================================================
# STEP 3. ERP 資料整合
# ============================================================
product_keep = [
    "SKU",
    "產品線",
    "標準材料成本_TWD",
    "標準人工成本_TWD",
    "標準委外成本_TWD",
    "標準製造費_TWD",
]

cost_keep = [
    "工單ID",
    "SKU",
    "完工數量",
    "CNC加工工時_hr",
    "組裝工時_hr",
    "測試工時_hr",
    "換線工時_hr",
    "實際製造成本_TWD",
]

df = cost_df[cost_keep].merge(
    product_df[product_keep],
    on="SKU",
    how="inner"
)

df["標準材料總成本"] = df["完工數量"] * df["標準材料成本_TWD"]
df["標準人工總成本"] = df["完工數量"] * df["標準人工成本_TWD"]
df["標準委外總成本"] = df["完工數量"] * df["標準委外成本_TWD"]
df["標準製造費總額"] = df["完工數量"] * df["標準製造費_TWD"]


# ============================================================
# STEP 4. 定義 X 與 Y
# ============================================================
FEATURES = [
    "完工數量",
    "標準材料總成本",
    "標準人工總成本",
    "標準委外總成本",
    "標準製造費總額",
    "CNC加工工時_hr",
    "組裝工時_hr",
    "測試工時_hr",
    "換線工時_hr",
]

FEATURE_LABELS = {
    "完工數量": "Completed Quantity",
    "標準材料總成本": "Standard Material Cost",
    "標準人工總成本": "Standard Labor Cost",
    "標準委外總成本": "Standard Outsourcing Cost",
    "標準製造費總額": "Standard Manufacturing Overhead",
    "CNC加工工時_hr": "CNC Machining Hours",
    "組裝工時_hr": "Assembly Hours",
    "測試工時_hr": "Test Hours",
    "換線工時_hr": "Setup Hours",
}

PRODUCT_LINE_EN = {
    "汽車精密零組件": "Automotive Precision Parts",
    "精密金屬加工件": "Precision Metal Components",
    "機電／自動化模組": "Mechatronics & Automation Modules",
    "半導體設備零組件": "Semiconductor Equipment Parts",
}

TARGET = "實際製造成本_TWD"

model_df = df[
    ["工單ID", "SKU", "產品線"] + FEATURES + [TARGET]
].dropna().copy()

print("\nERP 建模資料已建立完成。")
print(f"總建模筆數：{len(model_df):,}")

print("\n【各產品線資料筆數】")
product_count_df = (
    model_df.groupby("產品線")
    .size()
    .reset_index(name="資料筆數")
)

display(
    product_count_df.style
    .set_properties(**{
        "background-color": "#FFFDFC",
        "color": TEXT_DARK,
        "border-color": "#E7DDD3"
    })
    .set_table_styles([
        {"selector": "th", "props": [
            ("background-color", WARM_GOLD),
            ("color", TEXT_DARK),
            ("font-weight", "bold"),
            ("border", "1px solid #E7D8A4")
        ]}
    ])
)


# ============================================================
# STEP 5. 產品線選單
# ============================================================
product_lines = sorted(model_df["產品線"].unique().tolist())

product_dropdown = widgets.Dropdown(
    options=product_lines,
    value=product_lines[0],
    description="產品線：",
    style={"description_width": "initial"},
    layout=widgets.Layout(width="500px")
)

run_button = widgets.Button(
    description="執行多元回歸",
    button_style="warning",
    icon="play",
    layout=widgets.Layout(width="180px")
)

output = widgets.Output()

display(widgets.HBox([product_dropdown, run_button]))
display(output)


# ============================================================
# STEP 6. 多元回歸函數
# ============================================================
def run_regression(product_line):

    selected_df = model_df[
        model_df["產品線"] == product_line
    ].copy()

    if len(selected_df) < 20:
        print("此產品線資料筆數不足，無法建立穩定模型。")
        return

    product_line_en = PRODUCT_LINE_EN.get(
        product_line,
        "Selected Product Line"
    )

    X = selected_df[FEATURES]
    y = selected_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES)
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regression", LinearRegression())
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --------------------------------------------------------
    # 模型評估
    # --------------------------------------------------------
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(
        np.abs(
            (y_test.values - y_pred) /
            np.maximum(np.abs(y_test.values), 1)
        )
    ) * 100

    # --------------------------------------------------------
    # 預測結果表
    # --------------------------------------------------------
    result = selected_df.loc[
        X_test.index,
        ["工單ID", "SKU", "產品線"]
    ].copy()

    result["實際製造成本"] = y_test.values
    result["預測製造成本"] = y_pred
    result["預測誤差"] = (
        result["實際製造成本"] - result["預測製造成本"]
    )
    result["絕對誤差"] = np.abs(result["預測誤差"])
    result["誤差率(%)"] = (
        result["絕對誤差"] /
        np.maximum(result["實際製造成本"], 1)
    ) * 100

    # --------------------------------------------------------
    # 標準化迴歸係數
    # --------------------------------------------------------
    coef = model.named_steps["regression"].coef_

    coef_df = pd.DataFrame({
        "變數": FEATURES,
        "圖表英文名稱": [FEATURE_LABELS[f] for f in FEATURES],
        "標準化迴歸係數": coef
    })

    coef_df["影響程度"] = coef_df["標準化迴歸係數"].abs()
    coef_df = coef_df.sort_values(
        "影響程度",
        ascending=False
    )

    # --------------------------------------------------------
    # 中文模型摘要
    # --------------------------------------------------------
    print("=" * 72)
    print("YUNFA ERP－多元線性迴歸分析")
    print("=" * 72)
    print(f"選定產品線          ：{product_line}")
    print(f"建模資料筆數        ：{len(selected_df):,}")
    print(f"訓練集 / 測試集     ：{len(X_train):,} / {len(X_test):,}")
    print(f"R²                  ：{r2*100:.2f}%")
    print(f"MAE                 ：NT$ {mae:,.0f}")
    print(f"RMSE                ：NT$ {rmse:,.0f}")
    print(f"MAPE                ：{mape:.2f}%")
    print(f"測試集平均實際成本   ：NT$ {y_test.mean():,.0f}")
    print("=" * 72)

    print("\n【模型解說】")
    print("R²：表示模型可以解釋多少比例的製造成本變化。")
    print("MAE：表示每張工單平均相差多少金額。")
    print("RMSE：對較大的預測誤差給予較高權重。")
    print("MAPE：表示平均預測誤差占實際成本的百分比。")

    # --------------------------------------------------------
    # 暖色系係數表
    # --------------------------------------------------------
    print("\n【各變數對成本的影響程度】")
    coef_show = coef_df[["變數", "標準化迴歸係數"]].copy()
    coef_show["影響方向"] = np.where(
        coef_show["標準化迴歸係數"] >= 0,
        "正向",
        "負向"
    )
    coef_show["變數"] = coef_show["變數"].replace({
        "CNC加工工時_hr": "CNC加工工時(hr)",
        "組裝工時_hr": "組裝工時(hr)",
        "測試工時_hr": "測試工時(hr)",
        "換線工時_hr": "換線工時(hr)",
    })

    # 使用淡米杏色階，維持深色文字，避免深色背景造成閱讀困難
    coef_show = coef_show[["變數", "標準化迴歸係數", "影響方向"]]

    coef_abs_max = max(coef_show["標準化迴歸係數"].abs().max(), 1)

    def coef_soft_color(v):
        ratio = min(abs(v) / coef_abs_max, 1.0)
        if ratio >= 0.75:
            bg = "#F6D6B8"   # soft peach
        elif ratio >= 0.50:
            bg = "#FBE6D2"   # light apricot
        elif ratio >= 0.25:
            bg = "#FFF1E5"   # pale cream peach
        else:
            bg = "#FFFDFC"   # near white
        return f"background-color: {bg}; color: {TEXT_DARK};"

    display(
        coef_show.style
        .format({"標準化迴歸係數": "{:,.2f}"})
        .map(
            coef_soft_color,
            subset=["標準化迴歸係數"]
        )
        .set_properties(**{
            "border-color": "#E8DDD2"
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#F7E3CF"),
                ("color", TEXT_DARK),
                ("font-weight", "bold"),
                ("border", "1px solid #E8DDD2")
            ]},
            {"selector": "td", "props": [
                ("padding", "7px 10px")
            ]}
        ])
    )

    # --------------------------------------------------------
    # 暖色系誤差表
    # --------------------------------------------------------
    print("\n【迴歸係數方向說明】")
    print("正係數：在其他變數固定下，該變數增加時，預測製造成本傾向增加。")
    print("負係數：在其他變數固定下，該變數增加時，預測製造成本傾向降低。")
    print("標準化迴歸係數主要用來比較各變數的影響方向與相對程度，")
    print("不應直接解讀為因果關係。")

    print("\n【預測誤差最大的 10 筆工單】")
    top_error = (
        result.sort_values(
            "絕對誤差",
            ascending=False
        ).head(10)
    )

    display(
        top_error.style
        .format({
            "實際製造成本": "{:,.0f}",
            "預測製造成本": "{:,.0f}",
            "預測誤差": "{:,.0f}",
            "絕對誤差": "{:,.0f}",
            "誤差率(%)": "{:.2f}",
        })
        .map(
            lambda v: (
                "background-color: #FDE9D9; color: #3F3F3F"
                if v >= top_error["誤差率(%)"].quantile(0.75)
                else "background-color: #FFF4E6; color: #3F3F3F"
                if v >= top_error["誤差率(%)"].quantile(0.40)
                else "background-color: #FFFDFC; color: #3F3F3F"
            ),
            subset=["誤差率(%)"]
        )
        .set_properties(**{
            "color": TEXT_DARK,
            "border-color": "#E8DDD2"
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", WARM_MINT),
                ("color", TEXT_DARK),
                ("font-weight", "bold"),
                ("border", "1px solid #D8E2D2")
            ]}
        ])
    )

    # ========================================================
    # CHART 1 - Actual vs Predicted
    # ========================================================
    plt.figure(figsize=(10, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.78,
        s=58,
        color=WARM_ORANGE,
        edgecolors="white",
        linewidths=0.7
    )

    min_v = min(y_test.min(), y_pred.min())
    max_v = max(y_test.max(), y_pred.max())

    plt.plot(
        [min_v, max_v],
        [min_v, max_v],
        linestyle="--",
        linewidth=2,
        color=WARM_BLUE,
        label="Perfect Prediction"
    )

    plt.xlabel("Actual Manufacturing Cost (TWD)")
    plt.ylabel("Predicted Manufacturing Cost (TWD)")
    plt.title(
        f"Actual vs Predicted Cost - {product_line_en}",
        fontweight="bold"
    )
    plt.grid(True)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    # ========================================================
    # CHART 2 - Cost Drivers
    # ========================================================
    plot_coef = coef_df.sort_values(
        "影響程度",
        ascending=True
    ).copy()

    bar_colors = []
    warm_palette = [
        "#A8C69F",  # sage green
        "#F2C6A0",  # peach
        "#AFC8E6",  # powder blue
        "#D7C4E8",  # lavender
        "#E8D58A",  # soft gold
        "#C9D8B6",  # light olive
        "#E6B8AF",  # dusty rose
        "#B7D7D0",  # mint teal
        "#F4D7B9"   # cream apricot
    ]

    for i in range(len(plot_coef)):
        bar_colors.append(warm_palette[i % len(warm_palette)])

    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        plot_coef["圖表英文名稱"],
        plot_coef["標準化迴歸係數"],
        color=bar_colors,
        edgecolor="#F7F2EC",
        linewidth=1.2,
        height=0.62
    )

    # Add subtle value labels to reduce the "solid block" feeling
    for bar, value in zip(bars, plot_coef["標準化迴歸係數"]):
        x = bar.get_width()
        offset = 0.012 * max(plot_coef["標準化迴歸係數"].abs().max(), 1)
        plt.text(
            x + (offset if x >= 0 else -offset),
            bar.get_y() + bar.get_height()/2,
            f"{value:,.0f}",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=9,
            color=TEXT_DARK
        )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1.5,
        color="#888888"
    )

    plt.xlabel("Standardized Coefficient")
    plt.ylabel("Feature")
    plt.title(
        f"Manufacturing Cost Drivers - {product_line_en}",
        fontweight="bold"
    )
    plt.grid(axis="x")
    plt.tight_layout()
    plt.show()

    # ========================================================
    # CHART 3 - Error Distribution
    # ========================================================
    plt.figure(figsize=(10, 6))
    plt.hist(
        result["誤差率(%)"],
        bins=15,
        color="#B7D7C4",
        edgecolor="white",
        linewidth=1
    )

    plt.xlabel("Absolute Error Rate (%)")
    plt.ylabel("Number of Work Orders")
    plt.title(
        f"Prediction Error Distribution - {product_line_en}",
        fontweight="bold"
    )
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


    return model, result, coef_df


# ============================================================
# STEP 7. 按鈕事件
# ============================================================
def on_run_clicked(b):
    with output:
        clear_output(wait=True)
        selected_line = product_dropdown.value
        run_regression(selected_line)

run_button.on_click(on_run_clicked)

print("\n請從上方選單選擇一個產品線，再按「執行多元回歸」。")
