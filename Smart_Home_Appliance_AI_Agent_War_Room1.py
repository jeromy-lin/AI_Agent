# ============================================================
# Day 1 - Lab 1
# Manufacturing AI Agent
# Product Agent with Visual Dashboard
#
# 功能：
# 1. 上傳 Excel
# 2. 讀取 SKU總表
# 3. 自動檢查欄位
# 4. 建立產品視覺化 Dashboard
# 5. 建立 Product Search Tools
# 6. 建立 Rule-based Product Agent
# 7. 使用 Gradio 建立 Web UI
#
# 最終成果：
# Manufacturing AI Dashboard v0.1
# ============================================================


# ============================================================
# Step 1. 安裝套件
# ============================================================

!pip install -q gradio plotly openpyxl


# ============================================================
# Step 2. 載入套件
# ============================================================

import pandas as pd
import plotly.express as px
import gradio as gr

from google.colab import files
from IPython.display import display

print("✅ 套件載入完成")


# ============================================================
# Step 3. 上傳 Excel
# ============================================================

print("請上傳課程提供的 Excel 檔案")

uploaded = files.upload()

file_name = list(uploaded.keys())[0]

print("")
print("✅ Excel 上傳完成")
print("檔案名稱：", file_name)


# ============================================================
# Step 4. 查看 Excel 工作表
# ============================================================

excel_file = pd.ExcelFile(file_name)

print("")
print("📘 Excel 工作表：")

for i, sheet in enumerate(
    excel_file.sheet_names,
    start=1
):
    print(f"{i}. {sheet}")


# ============================================================
# Step 5. 讀取 SKU總表
#
# 這份課程 Excel 的真正欄位名稱在第3列
# 所以使用 header=2
# ============================================================

product_df = pd.read_excel(
    file_name,
    sheet_name="SKU總表",
    header=2
)

# 清除欄位名稱前後空白
product_df.columns = (
    product_df.columns
    .astype(str)
    .str.strip()
)

print("")
print("✅ SKU總表讀取完成")

print(f"資料筆數：{len(product_df):,}")
print(f"欄位數量：{len(product_df.columns)}")


# ============================================================
# Step 6. 顯示欄位
# ============================================================

print("")
print("===== 目前欄位 =====")

for i, col in enumerate(
    product_df.columns,
    start=1
):
    print(f"{i:02d}. {col}")


# ============================================================
# Step 7. 必要欄位檢查
# ============================================================

required_columns = [
    "SKU",
    "產品群",
    "產品別",
    "產品定位",
    "業務模式",
    "品牌/委託客戶",
    "主規格值",
    "單位",
    "上市年月",
    "生命週期"
]

missing_columns = [
    col
    for col in required_columns
    if col not in product_df.columns
]

if missing_columns:

    print("")
    print("❌ Excel 欄位檢查失敗")
    print("缺少以下欄位：")

    for col in missing_columns:
        print("-", col)

    raise ValueError(
        "請確認 SKU總表格式，"
        "以及標題列是否仍位於 Excel 第3列。"
    )

else:

    print("")
    print("✅ 必要欄位檢查完成")


# ============================================================
# Step 8. 顯示前10筆資料
# ============================================================

print("")
print("===== 前10筆資料 =====")

display(
    product_df.head(10)
)


# ============================================================
# Step 9. 日期欄位整理
# ============================================================

date_columns = [
    "上市年月",
    "停產年月"
]

for col in date_columns:

    if col in product_df.columns:

        product_df[col] = pd.to_datetime(
            product_df[col],
            errors="coerce"
        )

print("✅ 日期欄位處理完成")


# ============================================================
# Step 10. 基本統計
# ============================================================

total_sku = len(product_df)

product_count = (
    product_df["產品別"]
    .nunique()
)

product_group_count = (
    product_df["產品群"]
    .nunique()
)

obm_count = (
    product_df[
        product_df["業務模式"] == "OBM"
    ]
    .shape[0]
)

oem_count = (
    product_df[
        product_df["業務模式"] == "OEM"
    ]
    .shape[0]
)

odm_count = (
    product_df[
        product_df["業務模式"] == "ODM"
    ]
    .shape[0]
)

print("")
print("==============================")
print("      公司產品基本資料")
print("==============================")

print(f"SKU 總數：{total_sku}")
print(f"產品別：{product_count}")
print(f"產品群：{product_group_count}")
print(f"OBM SKU：{obm_count}")
print(f"OEM SKU：{oem_count}")
print(f"ODM SKU：{odm_count}")


# ============================================================
# Step 11. 產品別 SKU 統計
# ============================================================

product_summary = (
    product_df
    .groupby("產品別")
    .size()
    .reset_index(name="SKU數")
    .sort_values(
        "SKU數",
        ascending=False
    )
)


# ============================================================
# Step 12. Dashboard 圖表1
# 產品別 SKU 分布
# ============================================================

fig_product = px.bar(
    product_summary,
    x="產品別",
    y="SKU數",
    text="SKU數",
    title="各產品別 SKU 數量"
)

fig_product.update_layout(
    xaxis_title="產品別",
    yaxis_title="SKU 數"
)


# ============================================================
# Step 13. Dashboard 圖表2
# OBM / OEM / ODM 分布
# ============================================================

business_summary = (
    product_df["業務模式"]
    .value_counts()
    .reset_index()
)

business_summary.columns = [
    "業務模式",
    "SKU數"
]

fig_business = px.pie(
    business_summary,
    names="業務模式",
    values="SKU數",
    title="OBM / OEM / ODM 產品結構",
    hole=0.45
)


# ============================================================
# Step 14. Dashboard 圖表3
# 產品定位
# ============================================================

tier_summary = (
    product_df["產品定位"]
    .value_counts()
    .reset_index()
)

tier_summary.columns = [
    "產品定位",
    "SKU數"
]

fig_tier = px.bar(
    tier_summary,
    x="產品定位",
    y="SKU數",
    text="SKU數",
    title="入門 / 中階 / 高階產品結構"
)


# ============================================================
# Step 15. Dashboard 圖表4
# 產品別 × 業務模式
# ============================================================

product_business = (
    product_df
    .groupby(
        [
            "產品別",
            "業務模式"
        ]
    )
    .size()
    .reset_index(
        name="SKU數"
    )
)

fig_product_business = px.bar(
    product_business,
    x="產品別",
    y="SKU數",
    color="業務模式",
    barmode="group",
    title="產品別 × OBM / OEM / ODM"
)


# ============================================================
# Step 16. Dashboard 圖表5
# 上市年度分析
# ============================================================

launch_df = (
    product_df
    .dropna(
        subset=["上市年月"]
    )
    .copy()
)

launch_df["上市年份"] = (
    launch_df["上市年月"]
    .dt.year
)

launch_summary = (
    launch_df
    .groupby(
        [
            "上市年份",
            "產品別"
        ]
    )
    .size()
    .reset_index(
        name="SKU數"
    )
)

fig_launch = px.bar(
    launch_summary,
    x="上市年份",
    y="SKU數",
    color="產品別",
    barmode="stack",
    title="各年度上市產品數量"
)


# ============================================================
# Step 17. Product Search Tool
# ============================================================

def search_product(product_name):

    result = product_df[
        product_df["產品別"]
        .astype(str)
        .str.contains(
            product_name,
            na=False
        )
    ].copy()

    if result.empty:

        return result

    columns = [
        "SKU",
        "產品別",
        "產品定位",
        "業務模式",
        "品牌/委託客戶",
        "主規格值",
        "單位",
        "上市年月",
        "生命週期"
    ]

    return result[
        columns
    ]


# ============================================================
# Step 18. SKU Search Tool
# ============================================================

def search_sku(sku):

    result = product_df[
        product_df["SKU"]
        .astype(str)
        .str.upper()
        == sku.upper()
    ].copy()

    return result


# ============================================================
# Step 19. 多條件 Product Filter
# ============================================================

def product_filter(
    product=None,
    tier=None,
    business_mode=None,
    lifecycle=None
):

    result = product_df.copy()

    if product:

        result = result[
            result["產品別"] == product
        ]

    if tier:

        result = result[
            result["產品定位"] == tier
        ]

    if business_mode:

        result = result[
            result["業務模式"] == business_mode
        ]

    if lifecycle:

        result = result[
            result["生命週期"] == lifecycle
        ]

    return result


# ============================================================
# Step 20. 自然語言條件辨識
# ============================================================

def detect_conditions(question):

    question = str(question)

    detected_product = None
    detected_tier = None
    detected_mode = None
    detected_lifecycle = None


    # --------------------------------------------------------
    # 產品名稱
    # --------------------------------------------------------

    product_list = (
        product_df["產品別"]
        .dropna()
        .unique()
        .tolist()
    )

    for product in product_list:

        if product in question:

            detected_product = product
            break


    # --------------------------------------------------------
    # 產品定位
    # --------------------------------------------------------

    tier_keywords = {
        "入門": "入門型",
        "中階": "中階型",
        "高階": "高階型"
    }

    for keyword, tier in tier_keywords.items():

        if keyword in question:

            detected_tier = tier
            break


    # --------------------------------------------------------
    # 商業模式
    # --------------------------------------------------------

    upper_question = question.upper()

    for mode in [
        "OBM",
        "OEM",
        "ODM"
    ]:

        if mode in upper_question:

            detected_mode = mode
            break


    # --------------------------------------------------------
    # 生命週期
    # --------------------------------------------------------

    lifecycle_keywords = {
        "新品": "新品",
        "成長": "成長機種",
        "成熟": "主力/成熟",
        "主力": "主力/成熟",
        "停產": "歷史停產"
    }

    for keyword, lifecycle in lifecycle_keywords.items():

        if keyword in question:

            detected_lifecycle = lifecycle
            break


    return (
        detected_product,
        detected_tier,
        detected_mode,
        detected_lifecycle
    )


# ============================================================
# Step 21. Product Agent
# ============================================================

def product_agent(question):

    (
        product,
        tier,
        mode,
        lifecycle
    ) = detect_conditions(
        question
    )


    # --------------------------------------------------------
    # 如果完全沒有辨識到任何條件
    # --------------------------------------------------------

    if (
        product is None
        and tier is None
        and mode is None
        and lifecycle is None
    ):

        return (
            """
🤖 Product Agent

我目前沒有辨識到查詢條件。

你可以試著問：

・請找高階咖啡機
・請找 OBM 中階氣炸鍋
・有哪些新品？
・有哪些停產除濕機？
            """,
            pd.DataFrame(),
            None
        )


    # --------------------------------------------------------
    # 查詢
    # --------------------------------------------------------

    result = product_filter(
        product=product,
        tier=tier,
        business_mode=mode,
        lifecycle=lifecycle
    )


    if result.empty:

        return (
            "⚠️ 找不到符合條件的產品。",
            pd.DataFrame(),
            None
        )


    # --------------------------------------------------------
    # Agent 摘要
    # --------------------------------------------------------

    summary = []

    summary.append(
        "🤖 Product Agent 分析結果"
    )

    summary.append("")

    if product:
        summary.append(
            f"產品別：{product}"
        )

    if tier:
        summary.append(
            f"產品定位：{tier}"
        )

    if mode:
        summary.append(
            f"業務模式：{mode}"
        )

    if lifecycle:
        summary.append(
            f"生命週期：{lifecycle}"
        )

    summary.append("")

    summary.append(
        f"符合條件 SKU：{len(result)} 筆"
    )


    # --------------------------------------------------------
    # 產品定位摘要
    # --------------------------------------------------------

    summary.append("")
    summary.append(
        "產品定位分布："
    )

    tier_counts = (
        result["產品定位"]
        .value_counts()
    )

    for name, count in tier_counts.items():

        summary.append(
            f"• {name}：{count} SKU"
        )


    # --------------------------------------------------------
    # 商業模式摘要
    # --------------------------------------------------------

    summary.append("")
    summary.append(
        "業務模式分布："
    )

    mode_counts = (
        result["業務模式"]
        .value_counts()
    )

    for name, count in mode_counts.items():

        summary.append(
            f"• {name}：{count} SKU"
        )


    summary_text = "\n".join(
        summary
    )


    # --------------------------------------------------------
    # 表格
    # --------------------------------------------------------

    table_columns = [
        "SKU",
        "產品別",
        "產品定位",
        "業務模式",
        "品牌/委託客戶",
        "主規格值",
        "單位",
        "上市年月",
        "生命週期"
    ]

    table = result[
        table_columns
    ].copy()


    # --------------------------------------------------------
    # 查詢結果圖表
    # --------------------------------------------------------

    chart_data = (
        result
        .groupby(
            "產品定位"
        )
        .size()
        .reset_index(
            name="SKU數"
        )
    )

    fig = px.bar(
        chart_data,
        x="產品定位",
        y="SKU數",
        text="SKU數",
        title="Agent 查詢結果｜產品定位分布"
    )

    return (
        summary_text,
        table,
        fig
    )


# ============================================================
# Step 22. 建立 Gradio Dashboard
# ============================================================

with gr.Blocks(
    title="Manufacturing AI Dashboard"
) as demo:


    # ========================================================
    # Header
    # ========================================================

    gr.Markdown(
        """
# 🏭 Manufacturing AI Dashboard

### Product Agent v0.1

第一版製造業產品戰情室

**Data → Python → Visualization → Agent**
        """
    )


    # ========================================================
    # Tab 1：產品總覽
    # ========================================================

    with gr.Tab(
        "📊 產品總覽"
    ):


        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        gr.Markdown(
            f"""
## 公司產品概況

| SKU 總數 | 產品別 | 產品群 | OBM | OEM | ODM |
|---:|---:|---:|---:|---:|---:|
| **{total_sku}** | **{product_count}** | **{product_group_count}** | **{obm_count}** | **{oem_count}** | **{odm_count}** |
            """
        )


        # ----------------------------------------------------
        # 第一排圖表
        # ----------------------------------------------------

        with gr.Row():

            gr.Plot(
                value=fig_product,
                label="產品別 SKU"
            )

            gr.Plot(
                value=fig_business,
                label="商業模式"
            )


        # ----------------------------------------------------
        # 第二排圖表
        # ----------------------------------------------------

        with gr.Row():

            gr.Plot(
                value=fig_tier,
                label="產品定位"
            )

            gr.Plot(
                value=fig_product_business,
                label="產品 × 業務模式"
            )


        # ----------------------------------------------------
        # 第三排
        # ----------------------------------------------------

        gr.Plot(
            value=fig_launch,
            label="上市年度"
        )


    # ========================================================
    # Tab 2：Product Agent
    # ========================================================

    with gr.Tab(
        "🤖 Product Agent"
    ):


        gr.Markdown(
            """
## 用自然語言查詢 Product Master

你可以輸入：

**請找高階咖啡機**

**請找 OBM 的中階氣炸鍋**

**有哪些新品？**

**有哪些停產除濕機？**
            """
        )


        question_input = gr.Textbox(

            label="💬 請輸入你的問題",

            placeholder=(
                "例如：請找 OBM 的中階咖啡機"
            ),

            lines=2

        )


        analyze_button = gr.Button(
            "🤖 開始分析",
            variant="primary"
        )


        agent_summary = gr.Textbox(

            label="Agent 分析",

            lines=12

        )


        product_table = gr.Dataframe(

            label="查詢結果",

            interactive=False

        )


        product_chart = gr.Plot(
            label="分析圖表"
        )


        analyze_button.click(

            fn=product_agent,

            inputs=[
                question_input
            ],

            outputs=[
                agent_summary,
                product_table,
                product_chart
            ]

        )


        # 按 Enter 也可以查詢
        question_input.submit(

            fn=product_agent,

            inputs=[
                question_input
            ],

            outputs=[
                agent_summary,
                product_table,
                product_chart
            ]

        )


    # ========================================================
    # Tab 3：資料探索
    # ========================================================

    with gr.Tab(
        "🔎 Product Master"
    ):

        gr.Markdown(
            """
## ERP Product Master

這就是目前 Agent 背後使用的資料。
            """
        )

        gr.Dataframe(
            value=product_df,
            interactive=False,
            label="SKU總表"
        )


    # ========================================================
    # Tab 4：學生挑戰
    # ========================================================

    with gr.Tab(
        "🎯 學生挑戰"
    ):

        gr.Markdown(
            """
# Lab 1 Challenge

請使用 Product Agent 回答：

1. 公司目前有哪些產品別？
2. 哪一個產品別 SKU 最多？
3. 找出所有高階產品。
4. 找出 OBM 中階咖啡機。
5. 找出 OEM 入門氣炸鍋。
6. 比較 OBM、OEM、ODM SKU 數量。
7. 找出所有新品。
8. 找出所有已停產產品。

---

## 進階挑戰

試著修改 Python，讓 Agent 可以理解：

- 「我要看新品咖啡機」
- 「有哪些停產除濕機？」
- 「我要看高階 ODM 吸塵器」
- 「有哪些中階 OBM 產品？」

---

### 今天的核心觀念

**Excel**
→ **Pandas DataFrame**
→ **Python Function**
→ **Visualization**
→ **Agent**
            """
        )


# ============================================================
# Step 23. 啟動 Dashboard
# ============================================================

demo.launch(
    share=True,
    debug=True
)
