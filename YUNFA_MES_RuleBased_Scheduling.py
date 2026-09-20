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
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display, HTML
from google.colab import files

warnings.filterwarnings("ignore")

# ============================================================
# STEP 0-1｜Matplotlib 字體與圖表規格
# ============================================================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.style"] = "normal"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.unicode_minus"] = False

def format_axes(ax):
    """固定圖表 Tick 字體與角度規格"""
    for label in ax.get_xticklabels():
        label.set_fontstyle("normal")
        label.set_fontfamily("DejaVu Sans")
        label.set_rotation(0)
    for label in ax.get_yticklabels():
        label.set_fontstyle("normal")
        label.set_fontfamily("DejaVu Sans")

def display_styled_table(df, title=""):
    """柔和暖色調白底 HTML 表格渲染"""
    if title:
        display(HTML(f"<h4 style='color: #2c3e50; font-family: sans-serif; margin-top: 18px; margin-bottom: 8px;'>{title}</h4>"))
    styled = df.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f8f9f9'), ('color', '#2c3e50'), ('font-weight', 'bold'), ('border', '1px solid #d5dbdb'), ('padding', '8px'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('background-color', '#ffffff'), ('border', '1px solid #e5e8e8'), ('padding', '8px'), ('text-align', 'center')]},
        {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%'), ('margin-bottom', '15px')]}
    ]).hide(axis='index')
    display(styled)

# ============================================================
# STEP 1｜Upload YUNFA MES Excel
# ============================================================
print("請上傳 YUNFA_MES.xlsx 檔案：")
uploaded = files.upload()
file_name = next(iter(uploaded))
excel_bytes = io.BytesIO(uploaded[file_name])
print(f"\n已載入檔案：{file_name}\n")

# ============================================================
# STEP 2｜Read MES Sheets
# ============================================================
excel_file = pd.ExcelFile(excel_bytes)

required_sheets = ["機台主檔", "工藝路由主檔", "生產工單", "生產報工", "機台稼動日報", "品質異常"]
missing_sheets = [s for s in required_sheets if s not in excel_file.sheet_names]
if missing_sheets:
    raise ValueError(f"缺少必要工作表：{missing_sheets}")

machine_df = pd.read_excel(excel_file, sheet_name="機台主檔")
route_df = pd.read_excel(excel_file, sheet_name="工藝路由主檔")
wo_df = pd.read_excel(excel_file, sheet_name="生產工單")
report_df = pd.read_excel(excel_file, sheet_name="生產報工")
equip_df = pd.read_excel(excel_file, sheet_name="機台稼動日報")
quality_abn_df = pd.read_excel(excel_file, sheet_name="品質異常")

print("程式已完整讀取工作表，以下僅顯示前 5 筆供資料檢視。")

display_styled_table(machine_df.head(), "機台主檔 (前 5 筆)")
display_styled_table(route_df.head(), "工藝路由主檔 (前 5 筆)")
display_styled_table(wo_df.head(), "生產工單 (前 5 筆)")
display_styled_table(report_df.head(), "生產報工 (前 5 筆)")
display_styled_table(equip_df.head(), "機台稼動日報 (前 5 筆)")
display_styled_table(quality_abn_df.head(), "品質異常 (前 5 筆)")

# ============================================================
# STEP 3｜Column Validation
# ============================================================
wo_req_cols = ["工單ID", "SKU", "產品線", "計畫數量", "開單日期", "交期", "急單", "優先等級", "材料齊套", "品質風險", "狀態"]
route_req_cols = ["SKU", "路由明細ID", "工序序號", "工序名稱", "需求工作中心", "標準換線工時_hr", "標準加工工時_hr每100件"]
machine_req_cols = ["機台ID", "工作中心", "每日可用工時", "效率目標", "狀態"]

missing_wo_cols = [c for c in wo_req_cols if c not in wo_df.columns]
missing_route_cols = [c for c in route_req_cols if c not in route_df.columns]
missing_machine_cols = [c for c in machine_req_cols if c not in machine_df.columns]

if missing_wo_cols or missing_route_cols or missing_machine_cols:
    raise ValueError(f"欄位缺失，無法進行排程：\n工單欄位缺失: {missing_wo_cols}\n路由欄位缺失: {missing_route_cols}\n機台欄位缺失: {missing_machine_cols}")

# ============================================================
# STEP 4｜Data Cleaning
# ============================================================
date_cols_wo = ["開單日期", "計畫開工日期", "計畫完工日期", "交期", "實際開工日期", "實際完工日期"]
for col in date_cols_wo:
    if col in wo_df.columns:
        wo_df[col] = pd.to_datetime(wo_df[col], errors="coerce")

if "生效日期" in route_df.columns:
    route_df["生效日期"] = pd.to_datetime(route_df["生效日期"], errors="coerce")

route_df["工序序號"] = pd.to_numeric(route_df["工序序號"], errors="coerce")
route_df["標準換線工時_hr"] = pd.to_numeric(route_df["標準換線工時_hr"], errors="coerce")
route_df["標準加工工時_hr每100件"] = pd.to_numeric(route_df["標準加工工時_hr每100件"], errors="coerce")

wo_df["計畫數量"] = pd.to_numeric(wo_df["計畫數量"], errors="coerce")
machine_df["每日可用工時"] = pd.to_numeric(machine_df["每日可用工時"], errors="coerce")
machine_df["效率目標"] = pd.to_numeric(machine_df["效率目標"], errors="coerce")

# ============================================================
# STEP 5｜Define Unfinished Work Orders
# ============================================================
active_df = wo_df[wo_df["狀態"] != "已完工"].copy()

total_wo_count = len(wo_df)
completed_wo_count = len(wo_df[wo_df["狀態"] == "已完工"])
active_wo_count = len(active_df)

print(f"【工單狀態統計】")
print(f"生產工單總數: {total_wo_count}")
print(f"已完工工單數: {completed_wo_count}")
print(f"未完工工單數: {active_wo_count}")
print("未完工工單狀態分布：")
print(active_df["狀態"].value_counts().to_string())
print("-" * 50)

# ============================================================
# STEP 6｜Planning Reference Date
# ============================================================
PLANNING_DATE = active_df["開單日期"].min()
active_df["距離交期天數"] = (active_df["交期"] - PLANNING_DATE).dt.days
active_df["距離交期天數"] = active_df["距離交期天數"].fillna(999).astype(int)

# ============================================================
# STEP 7｜Build CNC Routing Standard Hours
# ============================================================
cnc_route = route_df[route_df["需求工作中心"] == "CNC"].copy()

cnc_agg = cnc_route.groupby("SKU").agg(
    CNC工序數=("工序序號", "count"),
    CNC標準換線工時_hr=("標準換線工時_hr", "sum"),
    CNC標準加工工時_hr每100件=("標準加工工時_hr每100件", "sum")
).reset_index()

active_df = pd.merge(active_df, cnc_agg, on="SKU", how="left")
active_df["有CNC路由"] = active_df["CNC工序數"].notna() & (active_df["CNC工序數"] > 0)
active_df["有CNC路由"] = active_df["有CNC路由"].map({True: "是", False: "否"})

active_df["CNC工序數"] = active_df["CNC工序數"].fillna(0).astype(int)
active_df["CNC標準換線工時_hr"] = active_df["CNC標準換線工時_hr"].fillna(0.0)
active_df["CNC標準加工工時_hr每100件"] = active_df["CNC標準加工工時_hr每100件"].fillna(0.0)

# ============================================================
# STEP 8｜Estimate CNC Production Hours
# ============================================================
active_df["CNC加工工時_hr"] = (active_df["CNC標準加工工時_hr每100件"] * active_df["計畫數量"] / 100.0).round(2)
active_df["CNC排程總工時_hr"] = (active_df["CNC標準換線工時_hr"] + active_df["CNC加工工時_hr"]).round(2)

# ============================================================
# STEP 9｜Rule-Based Priority Score
# ============================================================
# Rule 1: 急單
active_df["急單分數"] = active_df["急單"].apply(lambda x: 30 if x == "是" else 0)

# Rule 2: 交期
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

active_df["交期分數"] = active_df["距離交期天數"].apply(score_due)

# Rule 3: 客戶
if "客戶等級" in active_df.columns:
    def score_cust(c):
        c_str = str(c)
        if "VIP" in c_str:
            return 20
        elif "重要" in c_str:
            return 10
        else:
            return 0
    active_df["客戶分數"] = active_df["客戶等級"].apply(score_cust)
else:
    active_df["客戶等級"] = "未設定"
    active_df["客戶分數"] = 0

# Rule 4: 材料
active_df["材料分數"] = active_df["材料齊套"].apply(lambda x: 15 if x == "是" else -35)

# Rule 5: MES 優先等級
def score_priority(p):
    p_str = str(p)
    if p_str == "高":
        return 10
    elif p_str == "中":
        return 5
    else:
        return 0

active_df["工單優先等級分數"] = active_df["優先等級"].apply(score_priority)

# Rule 6: 換線
def score_changeover(hours):
    if hours <= 1.5:
        return 10
    elif hours >= 3.5:
        return -5
    else:
        return 0

active_df["換線分數"] = active_df["CNC標準換線工時_hr"].apply(score_changeover)

# Rule 7: 品質風險
def score_quality(q):
    q_str = str(q)
    if q_str == "高":
        return -15
    elif q_str == "中":
        return -5
    else:
        return 0

active_df["品質分數"] = active_df["品質風險"].apply(score_quality)

# ============================================================
# STEP 10｜Priority Score
# ============================================================
score_cols = ["急單分數", "交期分數", "客戶分數", "材料分數", "工單優先等級分數", "換線分數", "品質分數"]
active_df["排程分數"] = active_df[score_cols].sum(axis=1)

# ============================================================
# STEP 11｜Schedulable Decision
# ============================================================
active_df["可立即排產"] = np.where(
    (active_df["材料齊套"] == "是") &
    (active_df["品質風險"] != "高") &
    (active_df["有CNC路由"] == "是") &
    (active_df["CNC排程總工時_hr"].notna()),
    "是", "否"
)

# ============================================================
# STEP 12｜Rule Explanation
# ============================================================
def build_explanation(row):
    items = []
    if row["急單分數"] != 0:
        items.append(f"急單 {row['急單分數']:+d}")
    if row["交期分數"] != 0:
        items.append(f"交期壓力 {row['交期分數']:+d}")
    if row["客戶分數"] != 0:
        items.append(f"客戶等級 {row['客戶分數']:+d}")
    if row["材料分數"] != 0:
        items.append(f"材料狀態 {row['材料分數']:+d}")
    if row["工單優先等級分數"] != 0:
        items.append(f"優先級 {row['工單優先等級分數']:+d}")
    if row["換線分數"] != 0:
        items.append(f"換線效率 {row['換線分數']:+d}")
    if row["品質分數"] != 0:
        items.append(f"品質風險 {row['品質分數']:+d}")
    return "；".join(items) if items else "基礎排程"

active_df["規則說明"] = active_df.apply(build_explanation, axis=1)

# ============================================================
# STEP 13｜Priority Ranking
# ============================================================
active_df["schedulable_bool"] = active_df["可立即排產"] == "是"

priority_df = active_df.sort_values(
    by=["schedulable_bool", "排程分數", "距離交期天數", "CNC標準換線工時_hr", "開單日期"],
    ascending=[False, False, True, True, True]
).reset_index(drop=True)

priority_df.drop(columns=["schedulable_bool"], inplace=True)
priority_df["排程順位"] = range(1, len(priority_df) + 1)

# ============================================================
# STEP 14｜CNC Machine Pool
# ============================================================
cnc_machines = machine_df[machine_df["工作中心"] == "CNC"].copy()
if "可用" in cnc_machines["狀態"].values:
    cnc_machines = cnc_machines[cnc_machines["狀態"] == "可用"]

MACHINE_NAMES = sorted(cnc_machines["機台ID"].unique().tolist())
print(f"【CNC 可用機台池 ({len(MACHINE_NAMES)} 台)】: {MACHINE_NAMES}")

# ============================================================
# STEP 15｜Schedulable Work Orders
# ============================================================
schedulable = priority_df[priority_df["可立即排產"] == "是"].copy().reset_index(drop=True)
material_hold_df = priority_df[priority_df["材料齊套"] == "否"].copy().reset_index(drop=True)
quality_df = priority_df[priority_df["品質風險"] == "高"].copy().reset_index(drop=True)

print(f"未完工工單數: {len(priority_df)}")
print(f"材料齊套數: {len(priority_df[priority_df['材料齊套'] == '是'])}")
print(f"待料工單數: {len(material_hold_df)}")
print(f"高品質風險數: {len(quality_df)}")
print(f"可立即排產數: {len(schedulable)}")
print("-" * 50)

# ============================================================
# STEP 16｜Rule-Based Machine Assignment
# ============================================================
available = {m: 0.0 for m in MACHINE_NAMES}

assigned_list = []
for idx, row in schedulable.iterrows():
    m = min(available, key=available.get)
    start_h = available[m]
    dur = float(row["CNC排程總工時_hr"])
    finish_h = round(start_h + dur, 2)
    available[m] = finish_h

    row_dict = row.to_dict()
    row_dict["機台"] = m
    row_dict["開始工時"] = round(start_h, 2)
    row_dict["排程總工時"] = dur
    row_dict["完成工時"] = finish_h
    row_dict["RuleBased順位"] = idx + 1
    assigned_list.append(row_dict)

schedule = pd.DataFrame(assigned_list)

# ============================================================
# STEP 17｜FIFO Baseline
# ============================================================
fifo_schedulable = schedulable.sort_values(
    by=["開單日期", "工單ID"],
    ascending=[True, True]
).reset_index(drop=True)

fifo_available = {m: 0.0 for m in MACHINE_NAMES}
fifo_assigned_list = []

for idx, row in fifo_schedulable.iterrows():
    m = min(fifo_available, key=fifo_available.get)
    start_h = fifo_available[m]
    dur = float(row["CNC排程總工時_hr"])
    finish_h = round(start_h + dur, 2)
    fifo_available[m] = finish_h

    row_dict = row.to_dict()
    row_dict["機台"] = m
    row_dict["開始工時"] = round(start_h, 2)
    row_dict["排程總工時"] = dur
    row_dict["完成工時"] = finish_h
    row_dict["FIFO順位"] = idx + 1
    fifo_assigned_list.append(row_dict)

fifo = pd.DataFrame(fifo_assigned_list)

# ============================================================
# STEP 18｜FIFO vs Rule-Based Rank Comparison
# ============================================================
rank_compare = pd.merge(
    schedule[["工單ID", "RuleBased順位"]],
    fifo[["工單ID", "FIFO順位"]],
    on="工單ID"
)
rank_compare["順位提前"] = rank_compare["FIFO順位"] - rank_compare["RuleBased順位"]
rank_compare = rank_compare.sort_values(by="RuleBased順位").reset_index(drop=True)

# ============================================================
# STEP 19｜Benefit Metrics
# ============================================================
schedule_merged = pd.merge(schedule, rank_compare[["工單ID", "FIFO順位", "順位提前"]], on="工單ID")

# 1 & 2: 急單平均順位
urgent_fifo = schedule_merged[schedule_merged["急單"] == "是"]["FIFO順位"]
urgent_rb = schedule_merged[schedule_merged["急單"] == "是"]["RuleBased順位"]
urgent_avg_fifo = urgent_fifo.mean() if len(urgent_fifo) > 0 else 0.0
urgent_avg_rb = urgent_rb.mean() if len(urgent_rb) > 0 else 0.0

# 3 & 4: VIP 平均順位
vip_mask = schedule_merged["客戶等級"].astype(str).str.contains("VIP")
vip_fifo = schedule_merged[vip_mask]["FIFO順位"]
vip_rb = schedule_merged[vip_mask]["RuleBased順位"]
vip_avg_fifo = vip_fifo.mean() if len(vip_fifo) > 0 else 0.0
vip_avg_rb = vip_rb.mean() if len(vip_rb) > 0 else 0.0

# 5 & 6: Makespan
fifo_makespan = fifo["完成工時"].max() if len(fifo) > 0 else 0.0
rb_makespan = schedule["完成工時"].max() if len(schedule) > 0 else 0.0

# 7 & 8: 各 CNC 排程工時
fifo_mach_hours = fifo.groupby("機台")["排程總工時"].sum().reindex(MACHINE_NAMES, fill_value=0.0)
rb_mach_hours = schedule.groupby("機台")["排程總工時"].sum().reindex(MACHINE_NAMES, fill_value=0.0)

# 9 & 10: Utilization
fifo_util = (fifo_mach_hours / fifo_makespan * 100) if fifo_makespan > 0 else fifo_mach_hours * 0
rb_util = (rb_mach_hours / rb_makespan * 100) if rb_makespan > 0 else rb_mach_hours * 0

# 11 & 12: 平均 CNC 利用率
fifo_avg_util = fifo_util.mean()
rb_avg_util = rb_util.mean()

# ============================================================
# STEP 20｜Improvement Calculation
# ============================================================
def safe_improvement(before, after, lower_is_better=True):
    if before == 0:
        return 0.0
    if lower_is_better:
        return ((before - after) / before) * 100
    else:
        return ((after - before) / before) * 100

urgent_rank_improve = safe_improvement(urgent_avg_fifo, urgent_avg_rb, lower_is_better=True)
vip_rank_improve = safe_improvement(vip_avg_fifo, vip_avg_rb, lower_is_better=True)
makespan_improve = safe_improvement(fifo_makespan, rb_makespan, lower_is_better=True)
util_relative_improve = safe_improvement(fifo_avg_util, rb_avg_util, lower_is_better=False)
util_ppt_improve = rb_avg_util - fifo_avg_util

# ============================================================
# STEP 21｜Operational Screening KPI
# ============================================================
schedulable_rate = (len(schedulable) / active_wo_count * 100) if active_wo_count > 0 else 0.0
material_hold_rate = (len(material_hold_df) / active_wo_count * 100) if active_wo_count > 0 else 0.0
quality_risk_rate = (len(quality_df) / active_wo_count * 100) if active_wo_count > 0 else 0.0

# ============================================================
# TABLES DISPLAY
# ============================================================

# TABLE 1｜MES 未完工工單
table1_cols = ["工單ID", "SKU", "產品線", "計畫數量", "客戶等級", "急單", "優先等級", "材料齊套", "品質風險", "狀態", "交期"]
display_cols_1 = [c for c in table1_cols if c in active_df.columns]
display_styled_table(active_df[display_cols_1].head(20), "TABLE 1｜MES 未完工工單 (前 20 筆)")

# TABLE 2｜Rule-Based Scheduling 規則
rule_table_data = [
    {"規則": "急單", "加扣分": "+30", "管理意義": "優先處置急單，確保客戶交期不延誤"},
    {"規則": "交期", "加扣分": "+5 ~ +30", "管理意義": "距離交期天數越短得分越高，階梯式防範逾期"},
    {"規則": "VIP", "加扣分": "+20", "管理意義": "保障重要客戶服務水準 (SLA)"},
    {"規則": "重要客戶", "加扣分": "+10", "管理意義": "提升重點客戶滿意度與排程優先度"},
    {"規則": "材料", "加扣分": "+15 / -35", "管理意義": "齊套給予獎勵，缺料大幅扣分並隔離排產"},
    {"規則": "優先等級", "加扣分": "+5 ~ +10", "管理意義": "尊重 MES 系統既有高/中優先級設定"},
    {"規則": "換線", "加扣分": "+10 / -5", "管理意義": "鼓勵短換線工時工單，減少機台無效等待"},
    {"規則": "品質風險", "加扣分": "-5 / -15", "管理意義": "高風險品質工單扣分隔離，降減廢品率"}
]
display_styled_table(pd.DataFrame(rule_table_data), "TABLE 2｜Rule-Based Scheduling 規則表")

# TABLE 3｜未完工工單完整優先排序
table3_cols = [
    "排程順位", "工單ID", "SKU", "客戶等級", "距離交期天數", "急單", "優先等級",
    "材料齊套", "品質風險", "狀態", "CNC工序數", "CNC標準換線工時_hr",
    "CNC加工工時_hr", "CNC排程總工時_hr", "排程分數", "可立即排產"
]
display_cols_3 = [c for c in table3_cols if c in priority_df.columns]
display_styled_table(priority_df[display_cols_3].head(30), "TABLE 3｜未完工工單完整優先排序 (前 30 筆)")

# TABLE 4｜今日優先生產 Top 10
table4_cols = [
    "RuleBased順位", "工單ID", "SKU", "機台", "客戶等級", "急單",
    "材料齊套", "品質風險", "排程分數", "開始工時", "排程總工時", "完成工時"
]
display_cols_4 = [c for c in table4_cols if c in schedule.columns]
display_styled_table(schedule[display_cols_4].head(10), "TABLE 4｜今日優先生產 Top 10")

# TABLE 5｜Top 10 排程分數來源拆解
top10_ids = schedule.head(10)["工單ID"].tolist()
top10_score_df = priority_df[priority_df["工單ID"].isin(top10_ids)].copy()
table5_cols = ["工單ID", "急單分數", "交期分數", "客戶分數", "材料分數", "工單優先等級分數", "換線分數", "品質分數", "排程分數"]
display_styled_table(top10_score_df[table5_cols], "TABLE 5｜Top 10 排程分數來源拆解")

# TABLE 6｜Top 10 排程原因說明
table6_cols = ["工單ID", "排程分數", "規則說明"]
display_styled_table(top10_score_df[table6_cols], "TABLE 6｜Top 10 排程原因說明")

# TABLE 7｜待料工單
table7_cols = ["工單ID", "SKU", "客戶等級", "急單", "交期", "品質風險", "狀態", "排程分數"]
display_cols_7 = [c for c in table7_cols if c in material_hold_df.columns]
display_styled_table(material_hold_df[display_cols_7].head(30), "TABLE 7｜待料工單 (前 30 筆)")

# TABLE 8｜高品質風險工單
table8_cols = ["工單ID", "SKU", "客戶等級", "急單", "材料齊套", "狀態", "排程分數"]
display_cols_8 = [c for c in table8_cols if c in quality_df.columns]
display_styled_table(quality_df[display_cols_8], "TABLE 8｜高品質風險工單")

# TABLE 9｜FIFO vs Rule-Based 派工順位比較
display_styled_table(rank_compare, "TABLE 9｜FIFO vs Rule-Based 派工順位比較")

# TABLE 10｜FIFO vs Rule-Based Scheduling 效益分析表
benefit_data = [
    {
        "效益指標": "急單平均派工順位",
        "FIFO / 原始方式": f"{urgent_avg_fifo:.2f}",
        "Rule-Based / AI 排程": f"{urgent_avg_rb:.2f}",
        "改善幅度": f"{urgent_rank_improve:+.1f}%",
        "效益方向": "越低越好",
        "管理效益": "急單大幅提前派工，防範交期延誤"
    },
    {
        "效益指標": "VIP 平均派工順位",
        "FIFO / 原始方式": f"{vip_avg_fifo:.2f}",
        "Rule-Based / AI 排程": f"{vip_avg_rb:.2f}",
        "改善幅度": f"{vip_rank_improve:+.1f}%",
        "效益方向": "越低越好",
        "管理效益": "VIP 客戶工單優先處理，提升服務滿意度"
    },
    {
        "效益指標": "總排程工期(hr)",
        "FIFO / 原始方式": f"{fifo_makespan:.2f}",
        "Rule-Based / AI 排程": f"{rb_makespan:.2f}",
        "改善幅度": f"{makespan_improve:+.1f}%",
        "效益方向": "越低越好",
        "管理效益": "總製造週期 (Makespan) 縮短"
    },
    {
        "效益指標": "平均 CNC 利用率(%)",
        "FIFO / 原始方式": f"{fifo_avg_util:.1f}%",
        "Rule-Based / AI 排程": f"{rb_avg_util:.1f}%",
        "改善幅度": f"+{util_ppt_improve:.1f} ppt",
        "效益方向": "越高越好",
        "管理效益": "機台負載更均勻，平均利用率提升"
    },
    {
        "效益指標": "可立即排產工單",
        "FIFO / 原始方式": f"{len(schedulable)} 筆",
        "Rule-Based / AI 排程": f"{len(schedulable)} 筆",
        "改善幅度": f"{schedulable_rate:.1f}% (占比)",
        "效益方向": "精準篩選",
        "管理效益": "自動辨識具備開工條件之工單"
    },
    {
        "效益指標": "待料工單隔離",
        "FIFO / 原始方式": "混入排程",
        "Rule-Based / AI 排程": f"{len(material_hold_df)} 筆隔離",
        "改善幅度": f"{material_hold_rate:.1f}% (占比)",
        "效益方向": "防呆隔離",
        "管理效益": "防止缺料工單佔用現場產線"
    },
    {
        "效益指標": "高品質風險工單辨識",
        "FIFO / 原始方式": "未隔離",
        "Rule-Based / AI 排程": f"{len(quality_df)} 筆隔離",
        "改善幅度": f"{quality_risk_rate:.1f}% (占比)",
        "效益方向": "風險控管",
        "管理效益": "隔離高品質風險工單，先行品質確認"
    }
]
benefit_analysis_df = pd.DataFrame(benefit_data)
display_styled_table(benefit_analysis_df, "TABLE 10｜FIFO vs Rule-Based Scheduling 效益分析表")

# TABLE 11｜CNC 機台利用率比較
util_comp_df = pd.DataFrame({
    "CNC機台": MACHINE_NAMES,
    "FIFO工時": [round(fifo_mach_hours[m], 2) for m in MACHINE_NAMES],
    "Rule-Based工時": [round(rb_mach_hours[m], 2) for m in MACHINE_NAMES],
    "FIFO利用率(%)": [round(fifo_util[m], 1) for m in MACHINE_NAMES],
    "Rule-Based利用率(%)": [round(rb_util[m], 1) for m in MACHINE_NAMES]
})
display_styled_table(util_comp_df, "TABLE 11｜CNC 機台利用率比較")

# ============================================================
# CHARTS (1 ~ 6)
# ============================================================

# CHART 1｜Top 15 Work Orders by Priority Score
fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
ax.set_facecolor('white')
top15_chart = priority_df.head(15)
bars = ax.bar(top15_chart["工單ID"].astype(str), top15_chart["排程分數"], color="#3498db", width=0.6)

for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2, yval + 1, f"{int(yval)}",
        ha='center', va='bottom', fontsize=9, fontstyle="normal", fontfamily="DejaVu Sans"
    )

ax.set_title("Top 15 Work Orders by Priority Score", fontweight="bold", fontstyle="normal", fontfamily="DejaVu Sans")
ax.set_xlabel("Work Order", fontstyle="normal", fontfamily="DejaVu Sans")
ax.set_ylabel("Priority Score", fontstyle="normal", fontfamily="DejaVu Sans")
format_axes(ax)
plt.tight_layout()
plt.show()

# CHART 2｜MES Rule-Based CNC Production Schedule (Gantt Chart)
fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
ax.set_facecolor('white')

colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#e74c3c", "#34495e", "#f1c40f"]
for idx, row in schedule.iterrows():
    m_idx = MACHINE_NAMES.index(row["機台"])
    color = colors[idx % len(colors)]
    ax.barh(m_idx, row["排程總工時"], left=row["開始工時"], height=0.55, align='center', color=color, edgecolor='black', alpha=0.85)
    mid_x = row["開始工時"] + row["排程總工時"] / 2
    ax.text(mid_x, m_idx, str(row["工單ID"]), ha='center', va='center', fontsize=8, color='black', fontstyle="normal", fontfamily="DejaVu Sans")

max_h = int(np.ceil(rb_makespan)) + 8
for h in range(8, max_h, 8):
    ax.axvline(x=h, color='gray', linestyle='--', alpha=0.5)

ax.set_yticks(range(len(MACHINE_NAMES)))
ax.set_yticklabels(MACHINE_NAMES)
ax.set_xlabel("Scheduled CNC Hours", fontstyle="normal", fontfamily="DejaVu Sans")
ax.set_ylabel("CNC Machine", fontstyle="normal", fontfamily="DejaVu Sans")
ax.set_title("MES Rule-Based CNC Production Schedule", fontweight="bold", fontstyle="normal", fontfamily="DejaVu Sans")
format_axes(ax)
plt.tight_layout()
plt.show()

# CHART 3｜CNC Utilization after Rule-Based Scheduling
fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
ax.set_facecolor('white')
bars = ax.bar(util_comp_df["CNC機台"], util_comp_df["Rule-Based利用率(%)"], color="#2ecc71", width=0.55)

for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%",
        ha='center', va='bottom', fontsize=9, fontstyle="normal", fontfamily="DejaVu Sans"
    )

ax.set_ylim(0, max(util_comp_df["Rule-Based利用率(%)"].max() + 15, 100))
ax.set_title("CNC Utilization after Rule-Based Scheduling", fontweight="bold", fontstyle="normal", fontfamily="DejaVu Sans")
ax.set_xlabel("CNC Machine", fontstyle="normal", fontfamily="DejaVu Sans")
ax.set_ylabel("Utilization (%)", fontstyle="normal", fontfamily="DejaVu Sans")
format_axes(ax)
plt.tight_layout()
plt.show()

# CHART 4｜FIFO vs Rule-Based Dispatch Priority
fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
ax.set_facecolor('white')

x_indices = np.arange(2)
width = 0.35

fifo_ranks = [urgent_avg_fifo, vip_avg_fifo]
rb_ranks = [urgent_avg_rb, vip_avg_rb]

bars1 = ax.bar(x_indices - width/2, fifo_ranks, width, label='FIFO', color='#95a5a6')
bars2 = ax.bar(x_indices + width/2, rb_ranks, width, label='Rule-Based', color='#3498db')

for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.1f}", ha='center', va='bottom', fontstyle="normal", fontfamily="DejaVu Sans")

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.1f}", ha='center', va='bottom', fontstyle="normal", fontfamily="DejaVu Sans")

ax.set_xticks(x_indices)
ax.set_xticklabels(['Urgent Avg Rank', 'VIP Avg Rank'])
ax.set_ylabel("Average Dispatch Rank", fontstyle="normal", fontfamily="DejaVu Sans")
ax.set_title("FIFO vs Rule-Based Dispatch Priority", fontweight="bold", fontstyle="normal", fontfamily="DejaVu Sans")
ax.legend(frameon=False, prop={"family": "DejaVu Sans", "style": "normal"})
format_axes(ax)
plt.tight_layout()
plt.show()

# CHART 5｜FIFO vs Rule-Based Scheduling Efficiency
fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
ax.set_facecolor('white')

x_indices = np.arange(2)
width = 0.35

fifo_eff = [fifo_makespan, fifo_avg_util]
rb_eff = [rb_makespan, rb_avg_util]

bars1 = ax.bar(x_indices - width/2, fifo_eff, width, label='FIFO', color='#95a5a6')
bars2 = ax.bar(x_indices + width/2, rb_eff, width, label='Rule-Based', color='#2ecc71')

for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}", ha='center', va='bottom', fontstyle="normal", fontfamily="DejaVu Sans")

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}", ha='center', va='bottom', fontstyle="normal", fontfamily="DejaVu Sans")

ax.set_xticks(x_indices)
ax.set_xticklabels(['Makespan (hr)', 'Avg CNC Utilization (%)'])
ax.set_ylabel("Metric Value", fontstyle="normal", fontfamily="DejaVu Sans")
ax.set_title("FIFO vs Rule-Based Scheduling Efficiency", fontweight="bold", fontstyle="normal", fontfamily="DejaVu Sans")
ax.legend(frameon=False, prop={"family": "DejaVu Sans", "style": "normal"})
format_axes(ax)
plt.tight_layout()
plt.show()

# CHART 6｜MES Work Order Screening for Production Scheduling
fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')
ax.set_facecolor('white')

categories = ['Unfinished', 'Ready', 'Material Hold', 'Quality Risk']
counts = [active_wo_count, len(schedulable), len(material_hold_df), len(quality_df)]
colors_chart6 = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']

bars = ax.bar(categories, counts, color=colors_chart6, width=0.55)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{int(yval)}", ha='center', va='bottom', fontstyle="normal", fontfamily="DejaVu Sans")

ax.set_ylabel("Number of Work Orders", fontstyle="normal", fontfamily="DejaVu Sans")
ax.set_title("MES Work Order Screening for Production Scheduling", fontweight="bold", fontstyle="normal", fontfamily="DejaVu Sans")
format_axes(ax)
plt.tight_layout()
plt.show()

# ============================================================
# STEP 22｜AI Agent Benefit Summary
# ============================================================
print("=" * 70)
print(" AI Agent－MES 智慧排程效益摘要")
print("=" * 70)
print(f"1. 未完工工單總數：{active_wo_count} 筆")
print(f"2. 可立即排產工單數：{len(schedulable)} 筆 (可立即排產率：{schedulable_rate:.1f}%)")
print(f"3. 待料隔離工單數：{len(material_hold_df)} 筆 (待料率：{material_hold_rate:.1f}%)")
print(f"4. 高品質風險隔離工單數：{len(quality_df)} 筆 (風險率：{quality_risk_rate:.1f}%)")
print(f"5. 急單平均派工順位：FIFO {urgent_avg_fifo:.2f} 順位 ➔ Rule-Based {urgent_avg_rb:.2f} 順位 (提前 {urgent_rank_improve:+.1f}%)")
print(f"6. VIP 平均派工順位：FIFO {vip_avg_fifo:.2f} 順位 ➔ Rule-Based {vip_avg_rb:.2f} 順位 (提前 {vip_rank_improve:+.1f}%)")
print(f"7. 製造總工期 (Makespan)：FIFO {fifo_makespan:.2f} hr ➔ Rule-Based {rb_makespan:.2f} hr (改善率：{makespan_improve:+.1f}%)")
print(f"8. 平均 CNC 利用率：FIFO {fifo_avg_util:.1f}% ➔ Rule-Based {rb_avg_util:.1f}% (增加 +{util_ppt_improve:.1f} percentage points)")
print("=" * 70)

# ============================================================
# STEP 23｜AI Agent Management Recommendation
# ============================================================
print("\n【AI Agent 管理建議】")
print("1. 優先處理急單、交期壓力較高與高優先等級工單，確保重點客戶服務水準 (SLA) 與交期達成率。")
print("2. 待料工單暫不投入 CNC 排程，材料到齊後重新計算優先順位，避免現場塞車與停工待料。")
print("3. 高品質風險工單先進行品質確認與異常排除，再重新加入排程，降低現場廢品率與重工成本。")
print("4. 依 8 台 CNC 累積負載進行動態負載平衡派工，避免單一設備形成瓶頸與工時過度集中。")
print("5. 若 MES 生產報工持續更新，可定期自動重新執行排程，形成動態智慧排程 AI Agent 自適應機制。")
print("=" * 70)

# ============================================================
# STEP 24｜CSV Output
# ============================================================
priority_df.to_csv("YUNFA_MES_RuleBased_完整優先排序.csv", index=False, encoding="utf-8-sig")
schedule.to_csv("YUNFA_MES_RuleBased_CNC排程結果.csv", index=False, encoding="utf-8-sig")
rank_compare.to_csv("YUNFA_MES_FIFO_vs_RuleBased順位比較.csv", index=False, encoding="utf-8-sig")
benefit_analysis_df.to_csv("YUNFA_MES_RuleBased_效益分析.csv", index=False, encoding="utf-8-sig")
util_comp_df.to_csv("YUNFA_MES_CNC機台利用率比較.csv", index=False, encoding="utf-8-sig")

print("\n已成功導出 5 份 CSV 排程與效益分析報表檔案：")
print("- YUNFA_MES_RuleBased_完整優先排序.csv")
print("- YUNFA_MES_RuleBased_CNC排程結果.csv")
print("- YUNFA_MES_FIFO_vs_RuleBased順位比較.csv")
print("- YUNFA_MES_RuleBased_效益分析.csv")
print("- YUNFA_MES_CNC機台利用率比較.csv")
