# ============================================================
# YUNFA MES - Method 04 Rule-Based Scheduling
# AI 智慧工單排程與生產優先順序分析
# Day 2 - Practice 4
# 適用檔案：YUNFA_MES.xlsx
# Google Colab CPU
#
# 核心流程：
# MES 未完工工單
# → CNC 工藝路由標準工時
# → 可排產條件判斷
# → Rule-Based Priority Score
# → 8 台 CNC 負載平衡派工
# → FIFO Baseline
# → 效益量化比較
# → AI Agent 管理建議
#
# 作者：國立雲林科技大學電機系 林家仁
# ============================================================
# ============================================================
# STEP 0｜Import Packages
# ============================================================

# 如果 Colab 尚未安裝 openpyxl，可解除下一行註解
# !pip install -q openpyxl

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from google.colab import files


# ============================================================
# STEP 0-1｜Display Style
# ============================================================

COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#FF9DA7"
]

PASTEL_BLUE   = "#DCE8F5"
PASTEL_GREEN  = "#DCEBD8"
PASTEL_ORANGE = "#FBE4CD"
PASTEL_RED    = "#F7D8D8"
PASTEL_YELLOW = "#F8EFC8"
PASTEL_PURPLE = "#E9DDF0"
PASTEL_PINK   = "#F9DDE7"
PASTEL_TEAL   = "#D9EEEC"
PASTEL_GRAY   = "#E8E8E8"
PASTEL_BROWN  = "#EADFD6"

TEXT_DARK = "#3F4650"

plt.rcParams["figure.figsize"] = (13, 6.1)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.style"] = "normal"


def style_table(
    df_show,
    header_color=PASTEL_BLUE,
    formats=None
):

    styler = (
        df_show.style
        .set_properties(**{
            "color": TEXT_DARK,
            "text-align": "center",
            "font-style": "normal",
            "padding": "7px",
            "border": "1px solid #E3E3E3"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", header_color),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("padding", "8px"),
                    ("border", "1px solid white")
                ]
            },
            {
                "selector": "tbody tr:nth-child(odd) td",
                "props": [
                    ("background-color", "#FFFFFF")
                ]
            },
            {
                "selector": "tbody tr:nth-child(even) td",
                "props": [
                    ("background-color", "#F8FAFC")
                ]
            }
        ])
    )

    if formats:
        styler = styler.format(
            formats,
            na_rep="-"
        )

    return styler


# ============================================================
# STEP 1｜Upload YUNFA MES Excel
# ============================================================

uploaded = files.upload()

if len(uploaded) == 0:
    raise ValueError("尚未上傳 Excel 檔案。")

file_name = next(iter(uploaded))

excel_bytes = io.BytesIO(
    uploaded[file_name]
)

print(f"已載入檔案：{file_name}")


# ============================================================
# STEP 2｜Read MES Sheets
# ============================================================

xls = pd.ExcelFile(
    excel_bytes
)

required_sheets = [
    "機台主檔",
    "工藝路由主檔",
    "生產工單",
    "生產報工",
    "機台稼動日報",
    "品質異常"
]

missing_sheets = [
    sheet
    for sheet in required_sheets
    if sheet not in xls.sheet_names
]

if missing_sheets:

    raise ValueError(
        f"找不到必要 Sheet：{missing_sheets}\n"
        f"目前 Excel 內 Sheet：{xls.sheet_names}"
    )


machine_df = pd.read_excel(
    xls,
    sheet_name="機台主檔"
)

route_df = pd.read_excel(
    xls,
    sheet_name="工藝路由主檔"
)

wo_df = pd.read_excel(
    xls,
    sheet_name="生產工單"
)

report_df = pd.read_excel(
    xls,
    sheet_name="生產報工"
)

equip_df = pd.read_excel(
    xls,
    sheet_name="機台稼動日報"
)

quality_abn_df = pd.read_excel(
    xls,
    sheet_name="品質異常"
)


print()
print("=" * 90)
print("MES 資料讀取完成")
print("=" * 90)

print(
    "※ 程式已完整讀取各工作表；"
    "以下僅顯示前 5 筆供資料檢視。"
)


preview_tables = [
    ("機台主檔", machine_df, PASTEL_BLUE),
    ("工藝路由主檔", route_df, PASTEL_GREEN),
    ("生產工單", wo_df, PASTEL_ORANGE),
    ("生產報工", report_df, PASTEL_YELLOW)
]

for title, df_preview, color in preview_tables:

    print(f"\n【{title}｜前 5 筆】")

    display(
        style_table(
            df_preview.head(),
            color
        )
    )


# ============================================================
# STEP 3｜Column Validation
# ============================================================

required_wo_cols = [
    "工單ID",
    "SKU",
    "產品線",
    "計畫數量",
    "開單日期",
    "交期",
    "急單",
    "優先等級",
    "材料齊套",
    "品質風險",
    "狀態"
]

required_route_cols = [
    "SKU",
    "路由明細ID",
    "工序序號",
    "工序名稱",
    "需求工作中心",
    "標準換線工時_hr",
    "標準加工工時_hr每100件"
]

required_machine_cols = [
    "機台ID",
    "工作中心",
    "每日可用工時",
    "效率目標",
    "狀態"
]


for col in required_wo_cols:

    if col not in wo_df.columns:

        raise ValueError(
            f"生產工單缺少必要欄位：{col}"
        )


for col in required_route_cols:

    if col not in route_df.columns:

        raise ValueError(
            f"工藝路由主檔缺少必要欄位：{col}"
        )


for col in required_machine_cols:

    if col not in machine_df.columns:

        raise ValueError(
            f"機台主檔缺少必要欄位：{col}"
        )


# ============================================================
# STEP 4｜Data Cleaning
# ============================================================

date_cols_wo = [
    "開單日期",
    "計畫開工日期",
    "計畫完工日期",
    "交期",
    "實際開工日期",
    "實際完工日期"
]

for col in date_cols_wo:

    if col in wo_df.columns:

        wo_df[col] = pd.to_datetime(
            wo_df[col],
            errors="coerce"
        )


if "生效日期" in route_df.columns:

    route_df["生效日期"] = pd.to_datetime(
        route_df["生效日期"],
        errors="coerce"
    )


numeric_route_cols = [
    "工序序號",
    "標準換線工時_hr",
    "標準加工工時_hr每100件"
]

for col in numeric_route_cols:

    route_df[col] = pd.to_numeric(
        route_df[col],
        errors="coerce"
    )


wo_df["計畫數量"] = pd.to_numeric(
    wo_df["計畫數量"],
    errors="coerce"
)


for col in [
    "每日可用工時",
    "效率目標"
]:

    machine_df[col] = pd.to_numeric(
        machine_df[col],
        errors="coerce"
    )


# ============================================================
# STEP 5｜Customer Level Protection
# ============================================================

# 若 MES 本身有「客戶等級」，直接使用。
# 若沒有，則只給 0 分，不自行製造假客戶資料。

if "客戶等級" not in wo_df.columns:

    wo_df["客戶等級"] = "未設定"


# ============================================================
# STEP 6｜Define Unfinished Work Orders
# ============================================================

active_df = (
    wo_df[
        wo_df["狀態"] != "已完工"
    ]
    .copy()
    .reset_index(drop=True)
)


print()
print("=" * 90)
print("【MES 未完工工單盤點】")
print("=" * 90)

print(
    f"生產工單總數："
    f"{len(wo_df):,}"
)

print(
    f"已完工工單："
    f"{(wo_df['狀態'] == '已完工').sum():,}"
)

print(
    f"未完工工單："
    f"{len(active_df):,}"
)


status_summary = (
    active_df["狀態"]
    .value_counts()
    .rename_axis("工單狀態")
    .reset_index(
        name="工單數"
    )
)


display(
    style_table(
        status_summary,
        PASTEL_BLUE
    )
)


# ============================================================
# STEP 7｜Planning Reference Date
# ============================================================

# 為確保每次教學 Run 的結果一致，
# 不直接使用 datetime.today()。
#
# 使用本批未完工工單的最早開單日期
# 作為排程分析基準日。

if active_df["開單日期"].notna().sum() == 0:

    raise ValueError(
        "未完工工單中找不到有效開單日期。"
    )


PLANNING_DATE = (
    active_df["開單日期"]
    .dropna()
    .min()
    .normalize()
)


active_df["距離交期天數"] = (
    active_df["交期"]
    -
    PLANNING_DATE
).dt.days


active_df["距離交期天數"] = (
    active_df[
        "距離交期天數"
    ]
    .fillna(999)
    .astype(int)
)


print(
    f"\n排程分析基準日："
    f"{PLANNING_DATE.date()}"
)


# ============================================================
# STEP 8｜Build CNC Routing Standard Hours
# ============================================================

cnc_route = (
    route_df[
        route_df[
            "需求工作中心"
        ] == "CNC"
    ]
    .copy()
)


if len(cnc_route) == 0:

    raise ValueError(
        "工藝路由主檔中找不到需求工作中心 = CNC 的製程。"
    )


cnc_standard = (
    cnc_route
    .groupby(
        "SKU",
        as_index=False
    )
    .agg(

        CNC工序數=(
            "路由明細ID",
            "count"
        ),

        CNC標準換線工時_hr=(
            "標準換線工時_hr",
            "sum"
        ),

        CNC標準加工工時_hr每100件=(
            "標準加工工時_hr每100件",
            "sum"
        )
    )
)


active_df = active_df.merge(
    cnc_standard,
    on="SKU",
    how="left"
)


active_df["有CNC路由"] = np.where(
    active_df[
        "CNC工序數"
    ].notna(),
    "是",
    "否"
)


# ============================================================
# STEP 9｜Estimate CNC Production Hours
# ============================================================

# CNC 加工工時 =
#
# 每 100 件標準加工工時
# ×
# 計畫數量 / 100


active_df["CNC加工工時_hr"] = (

    active_df[
        "CNC標準加工工時_hr每100件"
    ]

    *

    active_df[
        "計畫數量"
    ]

    / 100.0
)


# CNC 排程總工時 =
#
# 標準換線工時
# +
# 標準加工工時


active_df["CNC排程總工時_hr"] = (

    active_df[
        "CNC標準換線工時_hr"
    ]

    +

    active_df[
        "CNC加工工時_hr"
    ]
)


active_df[
    "CNC排程總工時_hr"
] = (

    active_df[
        "CNC排程總工時_hr"
    ]
    .round(2)
)


# ============================================================
# STEP 10｜Rule-Based Priority Score
# ============================================================

def score_due(days):

    if days <= 1:
        return 30

    elif days <= 2:
        return 25

    elif days <= 4:
        return 15

    elif days <= 7:
        return 5

    else:
        return 0


# ------------------------------------------------------------
# 急單
# ------------------------------------------------------------

active_df["急單分數"] = np.where(
    active_df["急單"] == "是",
    30,
    0
)


# ------------------------------------------------------------
# 交期
# ------------------------------------------------------------

active_df["交期分數"] = (
    active_df[
        "距離交期天數"
    ]
    .apply(
        score_due
    )
)


# ------------------------------------------------------------
# 客戶
# ------------------------------------------------------------

active_df["客戶分數"] = (
    active_df[
        "客戶等級"
    ]
    .map({

        "VIP": 20,

        "重要客戶": 10,

        "重要": 10,

        "一般客戶": 0,

        "一般": 0,

        "未設定": 0
    })
    .fillna(0)
)


# ------------------------------------------------------------
# 材料
# ------------------------------------------------------------

active_df["材料分數"] = np.where(

    active_df[
        "材料齊套"
    ] == "是",

    15,

    -35
)


# ------------------------------------------------------------
# MES 優先等級
# ------------------------------------------------------------

active_df[
    "工單優先等級分數"
] = (

    active_df[
        "優先等級"
    ]
    .map({

        "高": 10,

        "中": 5,

        "一般": 0,

        "低": 0
    })
    .fillna(0)
)


# ------------------------------------------------------------
# 換線
# ------------------------------------------------------------

active_df[
    "換線分數"
] = np.select(

    [

        active_df[
            "CNC標準換線工時_hr"
        ] <= 1.5,

        active_df[
            "CNC標準換線工時_hr"
        ] >= 3.5

    ],

    [

        10,

        -5

    ],

    default=0
)


# ------------------------------------------------------------
# 品質風險
# ------------------------------------------------------------

active_df["品質分數"] = (

    active_df[
        "品質風險"
    ]
    .map({

        "高": -15,

        "中": -5,

        "低": 0
    })
    .fillna(0)
)


# ------------------------------------------------------------
# 排程總分
# ------------------------------------------------------------

score_cols = [

    "急單分數",

    "交期分數",

    "客戶分數",

    "材料分數",

    "工單優先等級分數",

    "換線分數",

    "品質分數"
]


active_df["排程分數"] = (
    active_df[
        score_cols
    ]
    .sum(axis=1)
)


# ============================================================
# STEP 11｜Schedulable Decision
# ============================================================

# 可立即排產：
#
# 1. 材料齊套
# 2. 品質風險不是高
# 3. 有 CNC Routing
# 4. 標準 CNC 工時可計算


active_df["可立即排產"] = np.where(

    (
        active_df[
            "材料齊套"
        ] == "是"
    )

    &

    (
        active_df[
            "品質風險"
        ] != "高"
    )

    &

    (
        active_df[
            "有CNC路由"
        ] == "是"
    )

    &

    (
        active_df[
            "CNC排程總工時_hr"
        ].notna()
    ),

    "是",

    "否"
)


# ============================================================
# STEP 12｜Rule Explanation
# ============================================================

def explain_rule(row):

    reasons = []

    if row["急單分數"] > 0:

        reasons.append(
            "急單 +30"
        )

    if row["交期分數"] > 0:

        reasons.append(
            f"交期壓力 "
            f"+{int(row['交期分數'])}"
        )

    if row["客戶分數"] > 0:

        reasons.append(
            f"{row['客戶等級']} "
            f"+{int(row['客戶分數'])}"
        )

    if row["材料齊套"] == "是":

        reasons.append(
            "材料到齊 +15"
        )

    else:

        reasons.append(
            "材料未到 -35"
        )

    if (
        row[
            "工單優先等級分數"
        ] > 0
    ):

        reasons.append(

            f"工單優先等級 "
            f"+{int(row['工單優先等級分數'])}"
        )

    if row["換線分數"] != 0:

        reasons.append(
            f"換線 "
            f"{int(row['換線分數']):+d}"
        )

    if row["品質分數"] != 0:

        reasons.append(
            f"品質風險 "
            f"{int(row['品質分數']):+d}"
        )

    return "；".join(
        reasons
    )


active_df["規則說明"] = (
    active_df.apply(
        explain_rule,
        axis=1
    )
)


# ============================================================
# STEP 13｜Priority Ranking
# ============================================================

priority_df = (

    active_df

    .sort_values(

        [

            "可立即排產",

            "排程分數",

            "距離交期天數",

            "CNC標準換線工時_hr",

            "開單日期"

        ],

        ascending=[

            False,

            False,

            True,

            True,

            True
        ]
    )

    .reset_index(
        drop=True
    )
)


priority_df["排程順位"] = (
    np.arange(
        1,
        len(priority_df) + 1
    )
)


# ============================================================
# STEP 14｜Get Available CNC Machines
# ============================================================

cnc_machine_df = (

    machine_df[
        machine_df[
            "工作中心"
        ] == "CNC"
    ]
    .copy()
)


# 若狀態欄有可用設備，優先使用可用設備
if (
    "可用"
    in
    cnc_machine_df[
        "狀態"
    ].astype(str).unique()
):

    cnc_machine_df = (
        cnc_machine_df[
            cnc_machine_df[
                "狀態"
            ] == "可用"
        ]
        .copy()
    )


MACHINE_NAMES = (
    cnc_machine_df[
        "機台ID"
    ]
    .astype(str)
    .tolist()
)


if len(MACHINE_NAMES) == 0:

    raise ValueError(
        "找不到 CNC 機台。"
    )


print()
print("=" * 90)
print("【CNC 機台資源】")
print("=" * 90)

print(
    f"CNC 機台數量："
    f"{len(MACHINE_NAMES)}"
)

print(
    "機台："
    +
    "、".join(
        MACHINE_NAMES
    )
)


# ============================================================
# STEP 15｜Schedulable Work Orders
# ============================================================

schedulable = (

    priority_df[
        priority_df[
            "可立即排產"
        ] == "是"
    ]

    .copy()

    .reset_index(
        drop=True
    )
)


material_hold_df = (

    priority_df[
        priority_df[
            "材料齊套"
        ] == "否"
    ]

    .copy()
)


quality_df = (

    priority_df[
        priority_df[
            "品質風險"
        ] == "高"
    ]

    .copy()
)


print()
print("=" * 90)
print("【MES 可排產分析】")
print("=" * 90)

print(
    f"未完工工單："
    f"{len(active_df):,}"
)

print(
    f"材料齊套："
    f"{(active_df['材料齊套'] == '是').sum():,}"
)

print(
    f"待料工單："
    f"{len(material_hold_df):,}"
)

print(
    f"高品質風險："
    f"{len(quality_df):,}"
)

print(
    f"可立即排產："
    f"{len(schedulable):,}"
)


if len(schedulable) == 0:

    raise ValueError(
        "目前沒有符合可立即排產條件的 CNC 工單。"
    )


# ============================================================
# STEP 16｜Rule-Based Machine Assignment
# ============================================================

available = {

    machine: 0.0

    for machine
    in MACHINE_NAMES
}


schedule_rows = []


for _, row in schedulable.iterrows():

    # 選擇目前累積排程工時最少的機台
    machine = min(
        available,
        key=available.get
    )

    start_hour = (
        available[
            machine
        ]
    )

    duration = float(
        row[
            "CNC排程總工時_hr"
        ]
    )

    finish_hour = (
        start_hour
        +
        duration
    )

    schedule_rows.append({

        **row.to_dict(),

        "機台": machine,

        "開始工時":
            round(
                start_hour,
                2
            ),

        "排程總工時":
            round(
                duration,
                2
            ),

        "完成工時":
            round(
                finish_hour,
                2
            )
    })

    available[
        machine
    ] = finish_hour


schedule = pd.DataFrame(
    schedule_rows
)


schedule[
    "RuleBased順位"
] = np.arange(
    1,
    len(schedule) + 1
)


# ============================================================
# STEP 17｜FIFO Baseline
# ============================================================

# FIFO 與 Rule-Based 使用完全相同的可排產工單。
#
# 唯一差異：
#
# FIFO：
# 依開單日期
#
# Rule-Based：
# 依排程分數


fifo_source = (

    schedulable

    .sort_values(

        [
            "開單日期",
            "工單ID"
        ],

        ascending=[
            True,
            True
        ]
    )

    .copy()

    .reset_index(
        drop=True
    )
)


fifo_available = {

    machine: 0.0

    for machine
    in MACHINE_NAMES
}


fifo_rows = []


for _, row in fifo_source.iterrows():

    machine = min(
        fifo_available,
        key=fifo_available.get
    )

    start_hour = (
        fifo_available[
            machine
        ]
    )

    duration = float(
        row[
            "CNC排程總工時_hr"
        ]
    )

    finish_hour = (
        start_hour
        +
        duration
    )

    fifo_rows.append({

        **row.to_dict(),

        "機台": machine,

        "開始工時":
            round(
                start_hour,
                2
            ),

        "排程總工時":
            round(
                duration,
                2
            ),

        "完成工時":
            round(
                finish_hour,
                2
            )
    })

    fifo_available[
        machine
    ] = finish_hour


fifo = pd.DataFrame(
    fifo_rows
)


fifo["FIFO順位"] = np.arange(
    1,
    len(fifo) + 1
)


# ============================================================
# STEP 18｜FIFO vs Rule-Based Rank Comparison
# ============================================================

rank_compare = (

    fifo[
        [
            "工單ID",
            "FIFO順位"
        ]
    ]

    .merge(

        schedule[
            [
                "工單ID",
                "RuleBased順位"
            ]
        ],

        on="工單ID",

        how="inner"
    )
)


rank_compare[
    "順位提前"
] = (

    rank_compare[
        "FIFO順位"
    ]

    -

    rank_compare[
        "RuleBased順位"
    ]
)


# ============================================================
# STEP 19｜Benefit Metrics
# ============================================================

def average_rank(
    rank_df,
    source_df,
    filter_column,
    filter_value,
    rank_column
):

    target_ids = source_df.loc[

        source_df[
            filter_column
        ] == filter_value,

        "工單ID"
    ]

    sub = rank_df[
        rank_df[
            "工單ID"
        ].isin(
            target_ids
        )
    ]

    if len(sub) == 0:

        return np.nan

    return (
        sub[
            rank_column
        ]
        .mean()
    )


# ------------------------------------------------------------
# 急單順位
# ------------------------------------------------------------

urgent_fifo_rank = average_rank(

    rank_compare,

    schedulable,

    "急單",

    "是",

    "FIFO順位"
)


urgent_rule_rank = average_rank(

    rank_compare,

    schedulable,

    "急單",

    "是",

    "RuleBased順位"
)


# ------------------------------------------------------------
# VIP 順位
# ------------------------------------------------------------

vip_fifo_rank = average_rank(

    rank_compare,

    schedulable,

    "客戶等級",

    "VIP",

    "FIFO順位"
)


vip_rule_rank = average_rank(

    rank_compare,

    schedulable,

    "客戶等級",

    "VIP",

    "RuleBased順位"
)


# ------------------------------------------------------------
# Makespan
# ------------------------------------------------------------

fifo_makespan = (

    fifo[
        "完成工時"
    ].max()

    if len(fifo) > 0

    else 0
)


rule_makespan = (

    schedule[
        "完成工時"
    ].max()

    if len(schedule) > 0

    else 0
)


# ------------------------------------------------------------
# Machine Hours
# ------------------------------------------------------------

fifo_machine_hours = (

    fifo

    .groupby(
        "機台"
    )[
        "排程總工時"
    ]

    .sum()

    .reindex(
        MACHINE_NAMES,
        fill_value=0
    )
)


rule_machine_hours = (

    schedule

    .groupby(
        "機台"
    )[
        "排程總工時"
    ]

    .sum()

    .reindex(
        MACHINE_NAMES,
        fill_value=0
    )
)


# ------------------------------------------------------------
# Machine Utilization
# ------------------------------------------------------------

if fifo_makespan > 0:

    fifo_util = (
        fifo_machine_hours
        /
        fifo_makespan
        *
        100
    )

else:

    fifo_util = (
        fifo_machine_hours
        *
        0
    )


if rule_makespan > 0:

    rule_util = (
        rule_machine_hours
        /
        rule_makespan
        *
        100
    )

else:

    rule_util = (
        rule_machine_hours
        *
        0
    )


fifo_avg_util = (
    fifo_util.mean()
)

rule_avg_util = (
    rule_util.mean()
)


# ============================================================
# STEP 20｜Improvement Calculation
# ============================================================

def safe_improvement(
    before,
    after,
    lower_is_better=False
):

    if (
        pd.isna(before)
        or
        pd.isna(after)
    ):

        return np.nan

    if before == 0:

        return np.nan

    if lower_is_better:

        return (
            before - after
        ) / before * 100

    else:

        return (
            after - before
        ) / before * 100


urgent_rank_improve = (
    safe_improvement(
        urgent_fifo_rank,
        urgent_rule_rank,
        lower_is_better=True
    )
)


vip_rank_improve = (
    safe_improvement(
        vip_fifo_rank,
        vip_rule_rank,
        lower_is_better=True
    )
)


makespan_improve = (
    safe_improvement(
        fifo_makespan,
        rule_makespan,
        lower_is_better=True
    )
)


util_relative_improve = (
    safe_improvement(
        fifo_avg_util,
        rule_avg_util,
        lower_is_better=False
    )
)


util_ppt_improve = (
    rule_avg_util
    -
    fifo_avg_util
)


# ============================================================
# STEP 21｜Operational Screening KPI
# ============================================================

schedulable_rate = (

    len(schedulable)
    /
    len(active_df)
    *
    100

    if len(active_df) > 0

    else 0
)


material_hold_rate = (

    len(material_hold_df)
    /
    len(active_df)
    *
    100

    if len(active_df) > 0

    else 0
)


quality_risk_rate = (

    len(quality_df)
    /
    len(active_df)
    *
    100

    if len(active_df) > 0

    else 0
)


# ============================================================
# TABLE 1｜MES Unfinished Work Orders
# ============================================================

print(
    "\n【表 1｜MES 未完工工單－前 20 筆】"
)


table1_cols = [
    "工單ID",
    "SKU",
    "產品線",
    "計畫數量",
    "客戶等級",
    "急單",
    "優先等級",
    "材料齊套",
    "品質風險",
    "狀態",
    "交期"
]


display(

    style_table(

        active_df
        .head(20)[
            table1_cols
        ],

        PASTEL_BLUE
    )
)


# ============================================================
# TABLE 2｜Rule-Based Scheduling Rules
# ============================================================

rules_df = pd.DataFrame({

    "規則": [

        "急單",

        "交期 ≤ 1 天",

        "交期 ≤ 2 天",

        "交期 ≤ 4 天",

        "交期 ≤ 7 天",

        "VIP 客戶",

        "重要客戶",

        "材料到齊",

        "材料未到",

        "優先等級高",

        "優先等級中",

        "CNC 換線 ≤ 1.5 hr",

        "CNC 換線 ≥ 3.5 hr",

        "品質風險高",

        "品質風險中"
    ],

    "加扣分": [

        "+30",

        "+30",

        "+25",

        "+15",

        "+5",

        "+20",

        "+10",

        "+15",

        "-35",

        "+10",

        "+5",

        "+10",

        "-5",

        "-15",

        "-5"
    ],

    "管理意義": [

        "提高急單生產優先度",

        "最高交期壓力",

        "交期即將到期",

        "提前安排生產",

        "一般交期提醒",

        "核心客戶優先",

        "重要客戶加權",

        "具備生產條件",

        "暫緩等待材料",

        "高優先工單",

        "中優先工單",

        "降低換線負擔",

        "避免長時間換線",

        "生產前先確認品質",

        "適度降低優先度"
    ]
})


print(
    "\n【表 2｜Rule-Based Scheduling 規則】"
)


display(

    style_table(
        rules_df,
        PASTEL_GREEN
    )
)


# ============================================================
# TABLE 3｜Priority Ranking
# ============================================================

print(
    "\n【表 3｜未完工工單完整優先排序－前 30 筆】"
)


priority_cols = [

    "排程順位",

    "工單ID",

    "SKU",

    "客戶等級",

    "距離交期天數",

    "急單",

    "優先等級",

    "材料齊套",

    "品質風險",

    "狀態",

    "CNC工序數",

    "CNC標準換線工時_hr",

    "CNC加工工時_hr",

    "CNC排程總工時_hr",

    "排程分數",

    "可立即排產"
]


display(

    style_table(

        priority_df
        .head(30)[
            priority_cols
        ],

        PASTEL_TEAL,

        {

            "CNC標準換線工時_hr":
                "{:.2f}",

            "CNC加工工時_hr":
                "{:.2f}",

            "CNC排程總工時_hr":
                "{:.2f}",

            "排程分數":
                "{:.0f}"
        }
    )
)


# ============================================================
# TABLE 4｜Top 10 Production Schedule
# ============================================================

print(
    "\n【表 4｜今日優先生產 Top 10】"
)


display(

    style_table(

        schedule
        .head(10)[
            [

                "RuleBased順位",

                "工單ID",

                "SKU",

                "機台",

                "客戶等級",

                "急單",

                "材料齊套",

                "品質風險",

                "排程分數",

                "開始工時",

                "排程總工時",

                "完成工時"
            ]
        ],

        PASTEL_ORANGE,

        {

            "排程分數":
                "{:.0f}",

            "開始工時":
                "{:.2f}",

            "排程總工時":
                "{:.2f}",

            "完成工時":
                "{:.2f}"
        }
    )
)


# ============================================================
# TABLE 5｜Top 10 Score Breakdown
# ============================================================

print(
    "\n【表 5｜Top 10 排程分數來源拆解】"
)


display(

    style_table(

        schedulable
        .head(10)[
            [

                "工單ID",

                "急單分數",

                "交期分數",

                "客戶分數",

                "材料分數",

                "工單優先等級分數",

                "換線分數",

                "品質分數",

                "排程分數"
            ]
        ],

        PASTEL_YELLOW,

        {

            "急單分數":
                "{:+.0f}",

            "交期分數":
                "{:+.0f}",

            "客戶分數":
                "{:+.0f}",

            "材料分數":
                "{:+.0f}",

            "工單優先等級分數":
                "{:+.0f}",

            "換線分數":
                "{:+.0f}",

            "品質分數":
                "{:+.0f}",

            "排程分數":
                "{:.0f}"
        }
    )
)


# ============================================================
# TABLE 6｜Top 10 Rule Explanation
# ============================================================

print(
    "\n【表 6｜Top 10 排程原因說明】"
)


display(

    style_table(

        schedulable
        .head(10)[
            [

                "工單ID",

                "排程分數",

                "規則說明"
            ]
        ],

        PASTEL_PINK,

        {

            "排程分數":
                "{:.0f}"
        }
    )
)


# ============================================================
# TABLE 7｜Material Hold
# ============================================================

print(
    "\n【表 7｜待料工單－前 30 筆】"
)


display(

    style_table(

        material_hold_df
        .head(30)[
            [

                "工單ID",

                "SKU",

                "客戶等級",

                "急單",

                "交期",

                "品質風險",

                "狀態",

                "排程分數"
            ]
        ],

        PASTEL_BROWN,

        {

            "排程分數":
                "{:.0f}"
        }
    )
)


# ============================================================
# TABLE 8｜High Quality Risk
# ============================================================

print(
    "\n【表 8｜高品質風險工單】"
)


display(

    style_table(

        quality_df[
            [

                "工單ID",

                "SKU",

                "客戶等級",

                "急單",

                "材料齊套",

                "狀態",

                "排程分數"
            ]
        ],

        PASTEL_PURPLE,

        {

            "排程分數":
                "{:.0f}"
        }
    )
)


# ============================================================
# TABLE 9｜FIFO vs Rule-Based Rank Comparison
# ============================================================

print(
    "\n【表 9｜FIFO vs Rule-Based 派工順位比較】"
)


rank_show = (

    rank_compare

    .sort_values(
        "RuleBased順位"
    )

    .head(20)
)


display(

    style_table(

        rank_show,

        PASTEL_BLUE,

        {

            "FIFO順位":
                "{:.0f}",

            "RuleBased順位":
                "{:.0f}",

            "順位提前":
                "{:+.0f}"
        }
    )
)


# ============================================================
# TABLE 10｜FIFO vs Rule-Based Benefit Analysis
# ============================================================

def show_number(value):

    if pd.isna(value):
        return "-"

    return f"{value:.2f}"


benefit_analysis_df = pd.DataFrame({

    "效益指標": [

        "急單平均派工順位",

        "VIP 平均派工順位",

        "總排程工期(hr)",

        "平均 CNC 利用率(%)",

        "可立即排產工單",

        "待料工單隔離",

        "高品質風險工單辨識"
    ],

    "FIFO / 原始方式": [

        urgent_fifo_rank,

        vip_fifo_rank,

        fifo_makespan,

        fifo_avg_util,

        len(active_df),

        0,

        0
    ],

    "Rule-Based / AI 排程": [

        urgent_rule_rank,

        vip_rule_rank,

        rule_makespan,

        rule_avg_util,

        len(schedulable),

        len(material_hold_df),

        len(quality_df)
    ],

    "改善幅度": [

        (
            f"{urgent_rank_improve:.1f}%"
            if not pd.isna(
                urgent_rank_improve
            )
            else "-"
        ),

        (
            f"{vip_rank_improve:.1f}%"
            if not pd.isna(
                vip_rank_improve
            )
            else "-"
        ),

        (
            f"{makespan_improve:.1f}%"
            if not pd.isna(
                makespan_improve
            )
            else "-"
        ),

        (
            f"{util_ppt_improve:+.1f} ppt"
            if not pd.isna(
                util_ppt_improve
            )
            else "-"
        ),

        (
            f"{schedulable_rate:.1f}% "
            f"可立即排產"
        ),

        (
            f"{material_hold_rate:.1f}% "
            f"自動隔離"
        ),

        (
            f"{quality_risk_rate:.1f}% "
            f"風險辨識"
        )
    ],

    "效益方向": [

        "越低越好",

        "越低越好",

        "越低越好",

        "越高越好",

        "自動篩選",

        "避免錯誤派工",

        "降低投產風險"
    ],

    "管理效益": [

        "急單優先進入生產，降低因 FIFO 排隊造成的急單延誤風險",

        "核心客戶工單優先處理，提高重要客戶服務優先度",

        "利用 8 台 CNC 負載平衡，降低整體生產完成時間",

        "降低機台工作量過度集中，提高 CNC 產能使用效率",

        "從全部未完工工單中，自動辨識目前真正具備生產條件的工單",

        "材料未齊工單先隔離，避免排入設備後才發現無法生產",

        "高品質風險工單先提醒確認，避免直接投入正式生產"
    ]
})


print()
print("=" * 110)
print("【表 10｜FIFO vs Rule-Based Scheduling 效益分析表】")
print("=" * 110)


benefit_display = (
    benefit_analysis_df
    .copy()
)


benefit_display[
    "FIFO / 原始方式"
] = benefit_display[
    "FIFO / 原始方式"
].apply(
    show_number
)


benefit_display[
    "Rule-Based / AI 排程"
] = benefit_display[
    "Rule-Based / AI 排程"
].apply(
    show_number
)


display(

    benefit_display.style

    .set_properties(**{

        "background-color":
            "#FFFDFC",

        "color":
            TEXT_DARK,

        "text-align":
            "center",

        "border":
            "1px solid #E2DED8",

        "padding":
            "8px"
    })

    .set_table_styles([

        {

            "selector":
                "th",

            "props": [

                (
                    "background-color",
                    "#DCEBD8"
                ),

                (
                    "color",
                    TEXT_DARK
                ),

                (
                    "font-weight",
                    "bold"
                ),

                (
                    "text-align",
                    "center"
                ),

                (
                    "border",
                    "1px solid white"
                ),

                (
                    "padding",
                    "9px"
                )
            ]
        },

        {

            "selector":
                "tbody tr:nth-child(odd) td",

            "props": [

                (
                    "background-color",
                    "#FFFFFF"
                )
            ]
        },

        {

            "selector":
                "tbody tr:nth-child(even) td",

            "props": [

                (
                    "background-color",
                    "#F8FAFC"
                )
            ]
        }
    ])
)


# ============================================================
# TABLE 11｜Machine Utilization
# ============================================================

machine_util_df = pd.DataFrame({

    "CNC機台":
        MACHINE_NAMES,

    "FIFO工時":
        fifo_machine_hours.values,

    "Rule-Based工時":
        rule_machine_hours.values,

    "FIFO利用率(%)":
        fifo_util.values,

    "Rule-Based利用率(%)":
        rule_util.values
})


print(
    "\n【表 11｜CNC 機台利用率比較】"
)


display(

    style_table(

        machine_util_df,

        PASTEL_TEAL,

        {

            "FIFO工時":
                "{:.2f}",

            "Rule-Based工時":
                "{:.2f}",

            "FIFO利用率(%)":
                "{:.1f}",

            "Rule-Based利用率(%)":
                "{:.1f}"
        }
    )
)


# ============================================================
# CHART 1｜Top 15 Work Orders by Priority Score
# ============================================================

top15 = (
    schedulable
    .head(15)
    .copy()
)


fig, ax = plt.subplots(
    figsize=(13, 6.1)
)


bars = ax.bar(

    np.arange(
        len(top15)
    ),

    top15[
        "排程分數"
    ],

    width=0.55,

    color=[

        COLORS[
            i % len(COLORS)
        ]

        for i
        in range(
            len(top15)
        )
    ]
)


ax.set_xticks(
    np.arange(
        len(top15)
    )
)


ax.set_xticklabels(
    top15[
        "工單ID"
    ],
    rotation=0
)


ax.set_title(
    "Top 15 Work Orders by Priority Score"
)

ax.set_xlabel(
    "Work Order"
)

ax.set_ylabel(
    "Priority Score"
)

ax.grid(
    axis="y",
    alpha=0.22
)


for bar, value in zip(
    bars,
    top15[
        "排程分數"
    ]
):

    ax.text(

        bar.get_x()
        +
        bar.get_width() / 2,

        value + 1,

        f"{int(value)}",

        ha="center",

        fontsize=9
    )


plt.tight_layout()
plt.show()


# ============================================================
# CHART 2｜Rule-Based CNC Gantt Chart
# ============================================================

machine_y = {

    machine: i

    for i, machine
    in enumerate(
        MACHINE_NAMES
    )
}


fig, ax = plt.subplots(
    figsize=(13, 6.1)
)


for i, row in schedule.iterrows():

    y = machine_y[
        row["機台"]
    ]

    ax.barh(

        y,

        row[
            "排程總工時"
        ],

        left=row[
            "開始工時"
        ],

        height=0.60,

        color=COLORS[
            i % len(COLORS)
        ],

        edgecolor="white"
    )

    if (
        row[
            "排程總工時"
        ] >= 2
    ):

        ax.text(

            row[
                "開始工時"
            ]
            +
            row[
                "排程總工時"
            ] / 2,

            y,

            row[
                "工單ID"
            ].replace(
                "WO",
                ""
            ),

            ha="center",

            va="center",

            fontsize=7
        )


ax.set_yticks(
    range(
        len(MACHINE_NAMES)
    )
)

ax.set_yticklabels(
    MACHINE_NAMES
)

ax.set_xlabel(
    "Scheduled CNC Hours"
)

ax.set_ylabel(
    "CNC Machine"
)

ax.set_title(
    "MES Rule-Based CNC Production Schedule"
)

ax.grid(
    axis="x",
    alpha=0.20
)


# 每 8 小時一條 Day Boundary
max_schedule_hour = (
    schedule[
        "完成工時"
    ].max()
)

for hour in np.arange(
    8,
    max_schedule_hour + 8,
    8
):

    ax.axvline(
        hour,
        linewidth=0.8,
        linestyle="--",
        alpha=0.25
    )


plt.tight_layout()
plt.show()


# ============================================================
# CHART 3｜CNC Utilization
# ============================================================

fig, ax = plt.subplots(
    figsize=(13, 6.1)
)


bars = ax.bar(

    MACHINE_NAMES,

    rule_util.values,

    width=0.55,

    color=COLORS[
        :len(MACHINE_NAMES)
    ]
)


ax.set_title(
    "CNC Utilization after Rule-Based Scheduling"
)

ax.set_ylabel(
    "Utilization (%)"
)

ax.set_ylim(
    0,
    105
)

ax.grid(
    axis="y",
    alpha=0.22
)


for bar, value in zip(
    bars,
    rule_util.values
):

    ax.text(

        bar.get_x()
        +
        bar.get_width() / 2,

        value + 1,

        f"{value:.1f}%",

        ha="center",

        fontsize=9
    )


plt.tight_layout()
plt.show()


# ============================================================
# CHART 4｜FIFO vs Rule-Based Dispatch Priority
# ============================================================

rank_metrics = pd.DataFrame({

    "Metric": [

        "Urgent Avg Rank",

        "VIP Avg Rank"
    ],

    "FIFO": [

        urgent_fifo_rank,

        vip_fifo_rank
    ],

    "Rule-Based": [

        urgent_rule_rank,

        vip_rule_rank
    ]
})


# NaN 只用於圖表顯示時轉 0
rank_plot = (
    rank_metrics
    .fillna(0)
)


x = np.arange(
    len(rank_plot)
)

width = 0.34


fig, ax = plt.subplots(
    figsize=(11, 6)
)


fifo_bars = ax.bar(

    x - width / 2,

    rank_plot[
        "FIFO"
    ],

    width,

    label="FIFO",

    color="#BAB0AC"
)


rule_bars = ax.bar(

    x + width / 2,

    rank_plot[
        "Rule-Based"
    ],

    width,

    label="Rule-Based",

    color="#4E79A7"
)


ax.set_xticks(x)

ax.set_xticklabels(
    rank_plot[
        "Metric"
    ]
)

ax.set_ylabel(
    "Average Dispatch Rank"
)

ax.set_title(
    "FIFO vs Rule-Based Dispatch Priority"
)

ax.grid(
    axis="y",
    alpha=0.22
)

ax.legend(
    frameon=False
)


for bars in [
    fifo_bars,
    rule_bars
]:

    for bar in bars:

        value = (
            bar.get_height()
        )

        ax.text(

            bar.get_x()
            +
            bar.get_width() / 2,

            value + 0.2,

            f"{value:.1f}",

            ha="center",

            fontsize=9
        )


plt.tight_layout()
plt.show()


# ============================================================
# CHART 5｜FIFO vs Rule-Based Efficiency
# ============================================================

efficiency_df = pd.DataFrame({

    "Metric": [

        "Makespan (hr)",

        "Avg CNC Utilization (%)"
    ],

    "FIFO": [

        fifo_makespan,

        fifo_avg_util
    ],

    "Rule-Based": [

        rule_makespan,

        rule_avg_util
    ]
})


x = np.arange(
    len(efficiency_df)
)

width = 0.34


fig, ax = plt.subplots(
    figsize=(11, 6)
)


fifo_bars = ax.bar(

    x - width / 2,

    efficiency_df[
        "FIFO"
    ],

    width,

    label="FIFO",

    color="#BAB0AC"
)


rule_bars = ax.bar(

    x + width / 2,

    efficiency_df[
        "Rule-Based"
    ],

    width,

    label="Rule-Based",

    color="#59A14F"
)


ax.set_xticks(x)

ax.set_xticklabels(
    efficiency_df[
        "Metric"
    ]
)

ax.set_ylabel(
    "Metric Value"
)

ax.set_title(
    "FIFO vs Rule-Based Scheduling Efficiency"
)

ax.grid(
    axis="y",
    alpha=0.22
)

ax.legend(
    frameon=False
)


for bars in [
    fifo_bars,
    rule_bars
]:

    for bar in bars:

        value = (
            bar.get_height()
        )

        ax.text(

            bar.get_x()
            +
            bar.get_width() / 2,

            value + 0.5,

            f"{value:.1f}",

            ha="center",

            fontsize=9
        )


plt.tight_layout()
plt.show()


# ============================================================
# CHART 6｜MES Scheduling Screening
# ============================================================

screening_labels = [

    "Unfinished",

    "Ready",

    "Material Hold",

    "Quality Risk"
]


screening_values = [

    len(active_df),

    len(schedulable),

    len(material_hold_df),

    len(quality_df)
]


fig, ax = plt.subplots(
    figsize=(11, 6)
)


bars = ax.bar(

    screening_labels,

    screening_values,

    width=0.55,

    color=[

        "#4E79A7",

        "#59A14F",

        "#F28E2B",

        "#E15759"
    ]
)


ax.set_title(
    "MES Work Order Screening for Production Scheduling"
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
    screening_values
):

    ax.text(

        bar.get_x()
        +
        bar.get_width() / 2,

        value + 1,

        f"{value}",

        ha="center"
    )


plt.tight_layout()
plt.show()


# ============================================================
# STEP 22｜AI Agent Benefit Summary
# ============================================================

urgent_count = (
    schedulable[
        "急單"
    ] == "是"
).sum()


vip_count = (
    schedulable[
        "客戶等級"
    ] == "VIP"
).sum()


print()
print("=" * 100)
print("AI Agent－MES 智慧排程效益摘要")
print("=" * 100)


print(
    f"1. MES 共辨識 "
    f"{len(active_df)} 筆未完工工單。"
)


print(
    f"2. 其中 {len(schedulable)} 筆"
    f"符合材料齊套、品質條件與 CNC 路由，"
    f"可立即投入生產排程，"
    f"占未完工工單 {schedulable_rate:.1f}%。"
)


print(
    f"3. 系統自動隔離 "
    f"{len(material_hold_df)} 筆待料工單，"
    f"避免無法執行工單占用 CNC 排程。"
)


print(
    f"4. 系統辨識 "
    f"{len(quality_df)} 筆高品質風險工單，"
    f"建議品質確認後再投入生產。"
)


if (
    not pd.isna(
        urgent_fifo_rank
    )
    and
    not pd.isna(
        urgent_rule_rank
    )
):

    print(
        f"5. 可排產工單中共有 "
        f"{urgent_count} 筆急單；"
        f"急單平均派工順位由 FIFO 的 "
        f"{urgent_fifo_rank:.1f} "
        f"提升至 Rule-Based 的 "
        f"{urgent_rule_rank:.1f}。"
    )


if (
    not pd.isna(
        vip_fifo_rank
    )
    and
    not pd.isna(
        vip_rule_rank
    )
):

    print(
        f"6. 可排產工單中共有 "
        f"{vip_count} 筆 VIP 工單；"
        f"VIP 平均派工順位由 FIFO 的 "
        f"{vip_fifo_rank:.1f} "
        f"調整為 Rule-Based 的 "
        f"{vip_rule_rank:.1f}。"
    )


print(
    f"7. FIFO 總排程工期約 "
    f"{fifo_makespan:.2f} 小時；"
    f"Rule-Based 為 "
    f"{rule_makespan:.2f} 小時。"
)


if not pd.isna(
    makespan_improve
):

    print(
        f"   總工期改善約 "
        f"{makespan_improve:.1f}%。"
    )


print(
    f"8. 平均 CNC 利用率由 "
    f"{fifo_avg_util:.1f}% "
    f"調整為 "
    f"{rule_avg_util:.1f}%，"
    f"變化 "
    f"{util_ppt_improve:+.1f} 個百分點。"
)


print()
print("【AI Agent 管理建議】")

print(
    "● 優先處理急單、交期壓力較高與高優先等級工單。"
)

print(
    "● 待料工單暫不投入 CNC 排程，材料到齊後重新計算優先順位。"
)

print(
    "● 高品質風險工單先進行品質確認，再重新加入排程。"
)

print(
    "● 依 8 台 CNC 即時累積負載進行派工，避免單一設備形成瓶頸。"
)

print(
    "● 將排程結果與 MES 生產報工持續更新，可形成動態智慧排程 AI Agent。"
)


# ============================================================
# STEP 23｜Export CSV
# ============================================================

priority_df.to_csv(

    "YUNFA_MES_RuleBased_完整優先排序.csv",

    index=False,

    encoding="utf-8-sig"
)


schedule.to_csv(

    "YUNFA_MES_RuleBased_CNC排程結果.csv",

    index=False,

    encoding="utf-8-sig"
)


rank_compare.to_csv(

    "YUNFA_MES_FIFO_vs_RuleBased順位比較.csv",

    index=False,

    encoding="utf-8-sig"
)


benefit_analysis_df.to_csv(

    "YUNFA_MES_RuleBased_效益分析.csv",

    index=False,

    encoding="utf-8-sig"
)


machine_util_df.to_csv(

    "YUNFA_MES_CNC機台利用率比較.csv",

    index=False,

    encoding="utf-8-sig"
)


print()
print("=" * 100)
print("分析完成")
print("=" * 100)

print(
    "已輸出：YUNFA_MES_RuleBased_完整優先排序.csv"
)

print(
    "已輸出：YUNFA_MES_RuleBased_CNC排程結果.csv"
)

print(
    "已輸出：YUNFA_MES_FIFO_vs_RuleBased順位比較.csv"
)

print(
    "已輸出：YUNFA_MES_RuleBased_效益分析.csv"
)

print(
    "已輸出：YUNFA_MES_CNC機台利用率比較.csv"
)
