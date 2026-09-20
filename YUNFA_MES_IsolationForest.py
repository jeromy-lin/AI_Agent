# ============================================================
# YUNFA MES - Method 03
# Isolation Forest：多來源、多變數製程異常偵測
# YUNFA MES - Isolation Forest：多來源、多變數製程異常偵測
# Day 2 - Practice 3 
# 適用檔案：YUNFA_MES.xlsx
# 作者：國立雲林科技大學電機系 林家仁
# ============================================================

# 如 Colab 尚未安裝套件，可先執行：
# !pip install -q openpyxl scikit-learn

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from google.colab import files

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# 參數設定
# ============================================================
HISTORY_YEARS = [2023, 2024, 2025]
DETECTION_YEAR = 2026
CONTAMINATION = 0.05
RANDOM_STATE = 42
TOP_N = 15

TEXT_DARK = "#3F3F3F"

PALETTE = {
    "orange": "#F2C6A0",
    "green": "#A8C69F",
    "blue": "#AFC8E6",
    "purple": "#D7C4E8",
    "red": "#D9534F",
    "gold": "#E8D58A",
    "mint": "#B7D7D0",
    "gray": "#D9D9D9",
}

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

# ============================================================
# STEP 1｜Upload YUNFA MES Excel
# ============================================================
uploaded = files.upload()
file_name = next(iter(uploaded))
excel_bytes = io.BytesIO(uploaded[file_name])

print(f"已載入檔案：{file_name}")

# ============================================================
# STEP 2｜Read MES Sheets
# 完全依照 YUNFA_MES.xlsx 目前實際 Sheet 名稱
# ============================================================
xls = pd.ExcelFile(excel_bytes)

required_sheets = [
    "機台主檔",
    "工藝路由主檔",
    "生產工單",
    "生產報工",
    "機台稼動日報",
    "品質異常",
]

missing = [s for s in required_sheets if s not in xls.sheet_names]
if missing:
    raise ValueError(
        f"找不到必要 Sheet：{missing}\n"
        f"目前 Excel 內 Sheet：{xls.sheet_names}"
    )

machine_df = pd.read_excel(xls, sheet_name="機台主檔")
route_df = pd.read_excel(xls, sheet_name="工藝路由主檔")
wo_df = pd.read_excel(xls, sheet_name="生產工單")
report_df = pd.read_excel(xls, sheet_name="生產報工")
equip_df = pd.read_excel(xls, sheet_name="機台稼動日報")
quality_abn_df = pd.read_excel(xls, sheet_name="品質異常")

def preview(df, title, header_color):
    print(f"\n【{title}】")
    print(f"※ 程式已讀取整張「{title}」工作表；以下僅顯示前 5 筆供資料檢視。")
    display(
        df.head().style
        .set_properties(**{
            "background-color": "#FFFDFC",
            "color": TEXT_DARK,
            "border": "1px solid #E5E1DC",
            "padding": "5px"
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", header_color),
                ("color", TEXT_DARK),
                ("font-weight", "bold"),
                ("font-style", "normal"),
                ("border", "1px solid #DDD8D0")
            ]}
        ])
    )

preview(machine_df, "機台主檔", "#F7E3CF")
preview(route_df, "工藝路由主檔", "#DDEAD9")
preview(wo_df, "生產工單", "#E7E1F2")
preview(report_df, "生產報工", "#F7E8C9")
preview(equip_df, "機台稼動日報", "#DDEAF5")
preview(quality_abn_df, "品質異常", "#F4CCCC")

# ============================================================
# STEP 3｜Data Cleaning
# ============================================================
def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# 日期
for c in ["計畫開始時間", "計畫結束時間", "實際開始時間", "實際結束時間"]:
    report_df[c] = pd.to_datetime(report_df[c], errors="coerce")

for c in ["開單日期", "計畫開工日期", "計畫完工日期", "交期",
          "實際開工日期", "實際完工日期"]:
    if c in wo_df.columns:
        wo_df[c] = pd.to_datetime(wo_df[c], errors="coerce")

equip_df["日期"] = pd.to_datetime(equip_df["日期"], errors="coerce").dt.normalize()

if "生效日期" in route_df.columns:
    route_df["生效日期"] = pd.to_datetime(route_df["生效日期"], errors="coerce")

if "發現日期" in quality_abn_df.columns:
    quality_abn_df["發現日期"] = pd.to_datetime(
        quality_abn_df["發現日期"], errors="coerce"
    )

# 數值
report_df = to_num(report_df, [
    "工序序號",
    "標準換線工時_hr",
    "實際換線工時_hr",
    "標準加工工時_hr",
    "實際加工工時_hr",
    "停機工時_hr",
    "良品數量",
    "報廢數量",
    "重工數量",
    "不良率",
])

machine_df = to_num(machine_df, [
    "每日可用工時",
    "效率目標",
    "設備年齡_年",
])

equip_df = to_num(equip_df, [
    "可用工時_hr",
    "運轉工時_hr",
    "換線工時_hr",
    "停機工時_hr",
    "稼動率",
    "品質良率",
    "OEE",
    "故障次數",
])

# ============================================================
# STEP 4｜Build Event-Level MES Dataset
# 分析單位：工單 × 工序 × 機台 × 一次實際報工
# ============================================================
events = report_df.copy()

# 以「生產工單開單年份」作為年度切分依據
# 這樣 2026 工單即使跨年度完工，仍屬於 2026 偵測批次
wo_keep = [
    "工單ID",
    "產品線",
    "開單日期",
    "計畫開工日期",
    "計畫完工日期",
    "交期",
    "計畫數量",
    "急單",
    "優先等級",
    "材料齊套",
    "品質風險",
    "狀態",
]

events = events.merge(
    wo_df[wo_keep],
    on="工單ID",
    how="left"
)

events["工單年度"] = events["開單日期"].dt.year
events["報工日期"] = events["實際開始時間"].dt.normalize()

# Merge 機台主檔
machine_keep = [
    "機台ID",
    "機台類型",
    "工作中心",
    "能力群組",
    "每日可用工時",
    "效率目標",
    "設備年齡_年",
    "狀態",
]

machine_merge = machine_df[machine_keep].copy()
machine_merge = machine_merge.rename(
    columns={"工作中心": "機台工作中心"}
)

events = events.merge(
    machine_merge,
    on="機台ID",
    how="left"
)

# Merge 2026 機台稼動日報
# 此 Sheet 目前只有 2026，因此用於「偵測後的設備情境解讀」，
# 不放入 2023-2025 Isolation Forest 訓練特徵。
equip_keep = [
    "日期",
    "機台ID",
    "可用工時_hr",
    "運轉工時_hr",
    "換線工時_hr",
    "停機工時_hr",
    "稼動率",
    "品質良率",
    "OEE",
    "故障次數",
]

equip_merge = equip_df[equip_keep].rename(columns={
    "日期": "報工日期",
    "停機工時_hr": "設備日停機工時_hr",
})

events = events.merge(
    equip_merge,
    on=["報工日期", "機台ID"],
    how="left"
)

# 品質異常僅作為模型結果的事後驗證，不作 Isolation Forest 輸入特徵
quality_flag = (
    quality_abn_df.groupby(["工單ID", "機台ID"])
    .agg(
        品質異常單數=("異常單ID", "count"),
        品質異常影響數量=("受影響數量", "sum"),
    )
    .reset_index()
)

events = events.merge(
    quality_flag,
    on=["工單ID", "機台ID"],
    how="left"
)

events["品質異常單數"] = events["品質異常單數"].fillna(0)
events["品質異常影響數量"] = events["品質異常影響數量"].fillna(0)

print("\n【MES 整合結果】")
print(f"生產報工原始筆數：{len(report_df):,}")
print(f"整合後事件筆數  ：{len(events):,}")

# ============================================================
# STEP 5｜Context-Aware Feature Engineering
# ============================================================
def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.replace(0, np.nan)

# 相對於製程標準
events["換線超時率"] = safe_divide(
    events["實際換線工時_hr"] - events["標準換線工時_hr"],
    events["標準換線工時_hr"]
)

events["加工超時率"] = safe_divide(
    events["實際加工工時_hr"] - events["標準加工工時_hr"],
    events["標準加工工時_hr"]
)

# 停機相對於機台每日可用工時
events["停機占比"] = safe_divide(
    events["停機工時_hr"],
    events["每日可用工時"]
)

# 品質
events["總產出數量"] = (
    events["良品數量"].fillna(0)
    + events["報廢數量"].fillna(0)
    + events["重工數量"].fillna(0)
)

events["報廢率"] = safe_divide(
    events["報廢數量"],
    events["總產出數量"]
).fillna(0)

events["重工率"] = safe_divide(
    events["重工數量"],
    events["總產出數量"]
).fillna(0)

events["良品產出率_件每時"] = safe_divide(
    events["良品數量"],
    events["實際加工工時_hr"]
)

# ============================================================
# STEP 6｜Historical Context Baselines
# 2023-2025 建立同 SKU / 工序 / 工作中心的歷史比較基準
# ============================================================
history_mask = events["工單年度"].isin(HISTORY_YEARS)
detect_mask = events["工單年度"].eq(DETECTION_YEAR)

if history_mask.sum() == 0:
    raise ValueError(f"找不到歷史年度 {HISTORY_YEARS} 的工單資料。")

if detect_mask.sum() == 0:
    available_years = sorted(
        events["工單年度"].dropna().astype(int).unique().tolist()
    )
    raise ValueError(
        f"找不到 {DETECTION_YEAR} 年工單資料。"
        f"目前可用年度：{available_years}"
    )

history_events = events.loc[history_mask].copy()

def add_context_z(
    full_df,
    history_df,
    value_col,
    primary_group,
    fallback_group,
    output_col
):
    global_mean = history_df[value_col].mean()
    global_std = history_df[value_col].std()

    p = (
        history_df.groupby(primary_group)[value_col]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "_p_mean", "std": "_p_std"})
    )

    f = (
        history_df.groupby(fallback_group)[value_col]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "_f_mean", "std": "_f_std"})
    )

    out = full_df.merge(p, on=primary_group, how="left")
    out = out.merge(f, on=fallback_group, how="left")

    mean_ref = out["_p_mean"].copy()
    std_ref = out["_p_std"].copy()

    weak = (
        mean_ref.isna()
        | std_ref.isna()
        | (std_ref.abs() < 1e-9)
    )
    mean_ref = mean_ref.where(~weak, out["_f_mean"])
    std_ref = std_ref.where(~weak, out["_f_std"])

    weak2 = (
        mean_ref.isna()
        | std_ref.isna()
        | (std_ref.abs() < 1e-9)
    )

    if pd.isna(global_std) or abs(global_std) < 1e-9:
        global_std = 1.0

    mean_ref = mean_ref.where(~weak2, global_mean)
    std_ref = std_ref.where(~weak2, global_std)

    out[output_col] = (
        (out[value_col] - mean_ref) / std_ref
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    return out.drop(
        columns=["_p_mean", "_p_std", "_f_mean", "_f_std"],
        errors="ignore"
    )

# 以同 SKU + 工序為主要比較情境，工作中心作 fallback
for value_col, output_col in [
    ("換線超時率", "換線情境偏差_Z"),
    ("加工超時率", "加工情境偏差_Z"),
    ("停機占比", "停機情境偏差_Z"),
    ("報廢率", "報廢情境偏差_Z"),
    ("重工率", "重工情境偏差_Z"),
    ("良品產出率_件每時", "產出效率偏差_Z"),
]:
    events = add_context_z(
        full_df=events,
        history_df=events.loc[history_mask].copy(),
        value_col=value_col,
        primary_group=["SKU", "工序序號"],
        fallback_group=["工作中心"],
        output_col=output_col
    )

# ============================================================
# STEP 7｜Isolation Forest Features
# 多變數聯合異常，不使用品質異常標籤
# ============================================================
FEATURES = [
    "換線超時率",
    "加工超時率",
    "停機占比",
    "報廢率",
    "重工率",
    "換線情境偏差_Z",
    "加工情境偏差_Z",
    "停機情境偏差_Z",
    "報廢情境偏差_Z",
    "重工情境偏差_Z",
    "產出效率偏差_Z",
]

FEATURE_EN = {
    "換線超時率": "Setup Overrun",
    "加工超時率": "Processing Overrun",
    "停機占比": "Downtime Ratio",
    "報廢率": "Scrap Rate",
    "重工率": "Rework Rate",
    "換線情境偏差_Z": "Context Setup Deviation",
    "加工情境偏差_Z": "Context Processing Deviation",
    "停機情境偏差_Z": "Context Downtime Deviation",
    "報廢情境偏差_Z": "Context Scrap Deviation",
    "重工情境偏差_Z": "Context Rework Deviation",
    "產出效率偏差_Z": "Throughput Deviation",
}

feature_desc = pd.DataFrame({
    "變數": FEATURES,
    "商業意義": [
        "實際換線時間相對標準值的超出比例",
        "實際加工時間相對標準值的超出比例",
        "停機工時占機台每日可用工時比例",
        "報廢數量占總產出比例",
        "重工數量占總產出比例",
        "換線表現相對同 SKU / 工序歷史基準的偏差",
        "加工表現相對同 SKU / 工序歷史基準的偏差",
        "停機表現相對同 SKU / 工序歷史基準的偏差",
        "報廢表現相對同 SKU / 工序歷史基準的偏差",
        "重工表現相對同 SKU / 工序歷史基準的偏差",
        "每小時良品產出相對同 SKU / 工序歷史基準的偏差",
    ]
})

print("\n【Isolation Forest 多變數特徵】")
display(feature_desc)

events[FEATURES] = (
    events[FEATURES]
    .replace([np.inf, -np.inf], np.nan)
)

history_medians = (
    events.loc[history_mask, FEATURES]
    .median(numeric_only=True)
)

events[FEATURES] = (
    events[FEATURES]
    .fillna(history_medians)
    .fillna(0)
)

# ============================================================
# STEP 8｜Timeline Split
# ============================================================
train_df = events.loc[history_mask].copy().reset_index(drop=True)
detect_df = events.loc[detect_mask].copy().reset_index(drop=True)

X_train = train_df[FEATURES].copy()
X_detect = detect_df[FEATURES].copy()

print("\n【時間切分】")
print(f"歷史基準年度：{HISTORY_YEARS}")
print(f"歷史資料筆數：{len(train_df):,}")
print(f"偵測年度    ：{DETECTION_YEAR}")
print(f"偵測資料筆數：{len(detect_df):,}")

# ============================================================
# STEP 9｜Standardization
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_detect_scaled = scaler.transform(X_detect)

# ============================================================
# STEP 10｜Isolation Forest
# ============================================================
iso = IsolationForest(
    n_estimators=400,
    max_samples="auto",
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

iso.fit(X_train_scaled)

train_score_raw = -iso.score_samples(X_train_scaled)
detect_score_raw = -iso.score_samples(X_detect_scaled)

detect_df["IF判定"] = np.where(
    iso.predict(X_detect_scaled) == -1,
    "異常",
    "正常"
)

sorted_train_scores = np.sort(train_score_raw)

detect_df["Anomaly Score"] = [
    100.0
    * np.searchsorted(
        sorted_train_scores,
        score,
        side="right"
    )
    / len(sorted_train_scores)
    for score in detect_score_raw
]

detect_df["Anomaly Rank"] = (
    detect_df["Anomaly Score"]
    .rank(method="first", ascending=False)
    .astype(int)
)

detect_df["異常層級"] = np.select(
    [
        detect_df["IF判定"].eq("異常"),
        detect_df["Anomaly Score"].ge(95),
    ],
    [
        "High Anomaly",
        "Review",
    ],
    default="Normal"
)

# ============================================================
# STEP 11｜Rule-Based Baseline
# 歷史單變數 95 / 5 百分位門檻
# ============================================================
rule_specs = {
    "換線超時率": ("high", 0.95),
    "加工超時率": ("high", 0.95),
    "停機占比": ("high", 0.95),
    "報廢率": ("high", 0.95),
    "重工率": ("high", 0.95),
    "產出效率偏差_Z": ("low", 0.05),
}

rule_flags = pd.DataFrame(index=detect_df.index)

for feature, (direction, q) in rule_specs.items():
    threshold = train_df[feature].quantile(q)

    if direction == "high":
        rule_flags[feature] = detect_df[feature] > threshold
    else:
        rule_flags[feature] = detect_df[feature] < threshold

detect_df["Rule Flag Count"] = rule_flags.sum(axis=1)

detect_df["Rule判定"] = np.where(
    detect_df["Rule Flag Count"] >= 1,
    "Warning",
    "Normal"
)

detect_df["方法對照"] = np.select(
    [
        detect_df["Rule判定"].eq("Warning")
        & detect_df["IF判定"].eq("異常"),

        detect_df["Rule判定"].eq("Normal")
        & detect_df["IF判定"].eq("異常"),

        detect_df["Rule判定"].eq("Warning")
        & detect_df["IF判定"].eq("正常"),
    ],
    [
        "共同警示",
        "多變數隱性異常",
        "單變數規則警示",
    ],
    default="共同正常"
)

# ============================================================
# STEP 12｜2026 Equipment Context
# 機台稼動日報目前只有 2026，因此只作異常事件的情境解讀
# ============================================================
for c in ["OEE", "稼動率", "品質良率"]:
    detect_df[c] = pd.to_numeric(detect_df[c], errors="coerce")

detect_df["設備日資料可用"] = np.where(
    detect_df["OEE"].notna(),
    "是",
    "否"
)

# ============================================================
# STEP 13｜Post-hoc Validation
# 品質異常不作模型輸入，只檢查模型偵測結果是否與既有異常紀錄重疊
# ============================================================
detect_df["有品質異常紀錄"] = np.where(
    detect_df["品質異常單數"] > 0,
    "是",
    "否"
)

if_detected_quality = detect_df.loc[
    detect_df["IF判定"].eq("異常"),
    "有品質異常紀錄"
].eq("是").mean()

# ============================================================
# STEP 14｜Model Summary
# ============================================================
print("\n" + "=" * 78)
print("YUNFA MES－Isolation Forest 多來源製程異常偵測")
print("=" * 78)
print(f"歷史基準年度                  ：{HISTORY_YEARS}")
print(f"偵測年度                      ：{DETECTION_YEAR}")
print(f"歷史基準事件數                ：{len(train_df):,}")
print(f"2026 偵測事件數               ：{len(detect_df):,}")
print(f"Isolation Forest contamination：{CONTAMINATION:.2%}")
print(f"模型判定異常事件數            ：{detect_df['IF判定'].eq('異常').sum():,}")
print(f"多變數隱性異常事件數          ：{detect_df['方法對照'].eq('多變數隱性異常').sum():,}")
print(f"異常事件具有品質異常紀錄比例  ：{if_detected_quality*100:.2f}%")
print("=" * 78)

print("\n【分析意義】")
print("分析單位為：工單 × 工序 × 機台 × 一次實際生產報工。")
print("2023-2025 用於建立歷史製造行為基準，2026 工單作為異常偵測對象。")
print("Isolation Forest 同時分析工時、停機、報廢、重工與同類製程歷史偏差。")
print("機台稼動日報與品質異常資料僅用於結果解讀與事後驗證，不參與模型訓練。")

# ============================================================
# STEP 15｜Top Anomaly Table
# ============================================================
top_anomaly = (
    detect_df
    .sort_values(
        ["Anomaly Score", "Rule Flag Count"],
        ascending=[False, False]
    )
    .head(TOP_N)
    .copy()
)

output_cols = [
    "報工ID",
    "工單ID",
    "SKU",
    "工序序號",
    "工序名稱",
    "工作中心",
    "機台ID",
    "Anomaly Score",
    "Anomaly Rank",
    "異常層級",
    "Rule Flag Count",
    "Rule判定",
    "IF判定",
    "方法對照",
    "換線超時率",
    "加工超時率",
    "停機占比",
    "報廢率",
    "重工率",
    "OEE",
    "稼動率",
    "品質良率",
    "故障次數",
    "有品質異常紀錄",
]

output_cols = [c for c in output_cols if c in top_anomaly.columns]

def highlight_level(v):
    if v == "High Anomaly":
        return "background-color: #F4CCCC; color: #8B0000; font-weight: bold"
    if v == "Review":
        return "background-color: #FFF2CC; color: #7F6000; font-weight: bold"
    return "background-color: #EAF4E4; color: #3F3F3F"

def highlight_compare(v):
    if v == "多變數隱性異常":
        return "background-color: #E7DDF2; color: #4C3B63; font-weight: bold"
    if v == "共同警示":
        return "background-color: #F4CCCC; color: #8B0000; font-weight: bold"
    if v == "單變數規則警示":
        return "background-color: #FFF2CC; color: #7F6000"
    return "background-color: #F7FBF5; color: #3F3F3F"

print(f"\n【Top {TOP_N} MES 異常事件】")

display(
    top_anomaly[output_cols].style
    .format({
        "Anomaly Score": "{:.1f}",
        "換線超時率": "{:.1%}",
        "加工超時率": "{:.1%}",
        "停機占比": "{:.1%}",
        "報廢率": "{:.2%}",
        "重工率": "{:.2%}",
        "OEE": "{:.2%}",
        "稼動率": "{:.2%}",
        "品質良率": "{:.2%}",
    }, na_rep="-")
    .map(highlight_level, subset=["異常層級"])
    .map(highlight_compare, subset=["方法對照"])
    .set_properties(**{
        "background-color": "#FFFDFC",
        "color": TEXT_DARK,
        "border": "1px solid #E4DFD9",
        "padding": "5px"
    })
    .set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#F7E8C9"),
            ("color", TEXT_DARK),
            ("font-weight", "bold"),
            ("font-style", "normal"),
            ("border", "1px solid #DED4C4")
        ]}
    ])
)

# ============================================================
# STEP 16｜Rule-Based vs Isolation Forest
# ============================================================
comparison_table = pd.crosstab(
    detect_df["Rule判定"],
    detect_df["IF判定"]
).reindex(
    index=["Normal", "Warning"],
    columns=["正常", "異常"],
    fill_value=0
)

print("\n【Rule-Based vs Isolation Forest】")
display(comparison_table)

# ============================================================
# STEP 17｜Top Anomaly Multi-Factor Profile
# ============================================================
top_row = top_anomaly.iloc[0]
top_pos = top_anomaly.index[0]

detect_position = detect_df.index.get_loc(top_pos)
top_scaled = X_detect_scaled[detect_position]

profile_df = pd.DataFrame({
    "特徵": FEATURES,
    "英文名稱": [FEATURE_EN[f] for f in FEATURES],
    "原始值": [top_row[f] for f in FEATURES],
    "相對歷史標準化偏差": top_scaled
})

profile_df["絕對偏差程度"] = (
    profile_df["相對歷史標準化偏差"].abs()
)

profile_df = profile_df.sort_values(
    "絕對偏差程度",
    ascending=False
)

print("\n【最高異常事件－多因素偏差輪廓】")
print("※ 此輪廓描述與歷史基準的偏離程度，不代表因果關係。")

display(
    profile_df[
        ["特徵", "原始值", "相對歷史標準化偏差", "絕對偏差程度"]
    ].style
    .format({
        "原始值": "{:.3f}",
        "相對歷史標準化偏差": "{:.2f}",
        "絕對偏差程度": "{:.2f}",
    })
)

# ============================================================
# CHART 1｜Top Anomaly Score Ranking
# ============================================================
plot_top = top_anomaly.sort_values(
    "Anomaly Score",
    ascending=True
).copy()

# TOP 15 圖表標籤全部使用英文／代碼，避免中文方塊字
plot_top["EventLabel"] = (
    "WO " + plot_top["工單ID"].astype(str)
    + " | OP " + plot_top["工序序號"].astype(str)
    + " | MC " + plot_top["機台ID"].astype(str)
)

plt.figure(figsize=(12, 7))

colors = [
    PALETTE["red"] if x == "High Anomaly"
    else PALETTE["gold"]
    for x in plot_top["異常層級"]
]

bars = plt.barh(
    plot_top["EventLabel"],
    plot_top["Anomaly Score"],
    color=colors,
    edgecolor="white"
)

for bar, value in zip(bars, plot_top["Anomaly Score"]):
    plt.text(
        value + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}",
        va="center",
        fontsize=9
    )

plt.xlabel("Anomaly Score (Historical Percentile)")
plt.ylabel("Work Order | Operation | Machine")
plt.title(
    f"Top {TOP_N} MES Anomaly Events - {DETECTION_YEAR}",
    fontweight="bold"
)
plt.xlim(0, 105)
plt.grid(axis="x")
plt.tight_layout()
plt.show()

# ============================================================
# CHART 2｜PCA 2D MES Anomaly Map
# ============================================================
pca = PCA(n_components=2)

all_scaled = np.vstack([
    X_train_scaled,
    X_detect_scaled
])

all_pca = pca.fit_transform(all_scaled)

train_pca = all_pca[:len(X_train_scaled)]
detect_pca = all_pca[len(X_train_scaled):]

plt.figure(figsize=(11, 7))

plt.scatter(
    train_pca[:, 0],
    train_pca[:, 1],
    s=28,
    alpha=0.22,
    color=PALETTE["gray"],
    label="Historical 2023-2025"
)

normal_mask = detect_df["IF判定"].eq("正常").values
anomaly_mask = detect_df["IF判定"].eq("異常").values

plt.scatter(
    detect_pca[normal_mask, 0],
    detect_pca[normal_mask, 1],
    s=50,
    alpha=0.72,
    color=PALETTE["green"],
    edgecolors="white",
    linewidths=0.6,
    label="2026 Normal"
)

plt.scatter(
    detect_pca[anomaly_mask, 0],
    detect_pca[anomaly_mask, 1],
    s=95,
    alpha=0.95,
    color=PALETTE["red"],
    edgecolors="white",
    linewidths=0.8,
    label="2026 Anomaly"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(
    "MES Multi-Variable Anomaly Map - PCA",
    fontweight="bold"
)
plt.legend(frameon=False, prop={"style": "normal"})
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# CHART 3｜Top Anomaly Multi-Factor Deviation Profile
# ============================================================
plot_profile = profile_df.sort_values(
    "相對歷史標準化偏差",
    ascending=True
).copy()

profile_colors = [
    PALETTE["red"] if abs(v) >= 2
    else PALETTE["orange"]
    for v in plot_profile["相對歷史標準化偏差"]
]

plt.figure(figsize=(12, 6.5))

bars = plt.barh(
    plot_profile["英文名稱"],
    plot_profile["相對歷史標準化偏差"],
    color=profile_colors,
    edgecolor="white"
)

plt.axvline(
    0,
    color="#888888",
    linestyle="--",
    linewidth=1
)

for bar, value in zip(
    bars,
    plot_profile["相對歷史標準化偏差"]
):
    offset = 0.08
    plt.text(
        value + (offset if value >= 0 else -offset),
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}",
        va="center",
        ha="left" if value >= 0 else "right",
        fontsize=9
    )

plt.xlabel("Standardized Deviation from Historical Baseline")
plt.ylabel("MES Feature")
plt.title(
    "Top Anomaly - Multi-Factor Deviation Profile",
    fontweight="bold"
)
plt.grid(axis="x")
plt.tight_layout()
plt.show()

# ============================================================
# CHART 4｜Rule-Based vs Isolation Forest Matrix
# ============================================================
matrix = comparison_table.values.astype(float)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

ax.set_xticks([0, 1])
ax.set_xticklabels(["IF Normal", "IF Anomaly"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Rule Normal", "Rule Warning"])

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(
            j,
            i,
            f"{int(matrix[i, j])}",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=TEXT_DARK
        )

ax.set_title(
    "Rule-Based vs Isolation Forest",
    fontweight="bold"
)

plt.colorbar(im, ax=ax, label="Number of Events")
plt.tight_layout()
plt.show()

# ============================================================
# STEP 18｜AI Agent Decision Meaning
# ============================================================
print("\n【後續 AI Agent 應用意義】")
print("Isolation Forest 用來辨識多變數聯合行為中相對少見的製造事件。")
print("模型提供 Anomaly Score、異常排序與多因素偏差輪廓，不直接宣告故障原因。")
print("AI Agent 可再結合 2026 機台稼動日報與品質異常紀錄，提供優先檢查與追蹤建議。")

if len(top_anomaly) > 0:
    top = top_anomaly.iloc[0]
    top_dev = profile_df.head(3)["特徵"].tolist()

    print("\n【最高優先事件】")
    print(f"工單ID        ：{top['工單ID']}")
    print(f"工序          ：{top['工序名稱']}")
    print(f"機台          ：{top['機台ID']}")
    print(f"Anomaly Score ：{top['Anomaly Score']:.1f}")
    print(f"主要偏差特徵  ：{'、'.join(top_dev)}")
    print("建議：優先檢視製程執行、停機紀錄、設備日績效與品質異常紀錄。")
