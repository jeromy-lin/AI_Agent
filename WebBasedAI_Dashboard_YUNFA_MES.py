# ===========================================================
# 主題：WebBased AI 數據戰情室應用實作－MES 生產管理
# 目標：利用 MES 生產資料建立網頁戰情室，
#       協助學員理解工單排程與製程異常分析的實務應用。
# 方法一：Rule-Based Scheduling（規則式排程）
#         綜合交期、急單、材料齊套與品質風險，
#         決定工單優先順序，並安排 CNC 機台作業。
# 方法二：Isolation Forest（異常偵測）
#         比對歷史與當期製程資料，辨識加工、換線、停機及品質表現異常的事件。
#
# 應用流程：上傳 MES 資料 → 執行分析 → 檢視 Web 圖表與工單明細 → 支援現場決策。
# 作者：國立雲林科技大學電機工程系 林家仁
# ===========================================================

import os
import io
import sys
import time
import importlib
import subprocess
import warnings
from html import escape

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Minimal dependency check
# No Transformers / Torch / GPU required.
# ------------------------------------------------------------
def _ensure_import(import_name, pip_spec=None):
    try:
        return importlib.import_module(import_name)
    except ImportError:
        if not pip_spec:
            return None
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pip_spec],
            check=False
        )
        try:
            return importlib.import_module(import_name)
        except Exception:
            return None

if _ensure_import("gradio", "gradio>=5,<7") is None:
    raise RuntimeError("Gradio 無法安裝 / 載入。")
if _ensure_import("openpyxl", "openpyxl") is None:
    raise RuntimeError("openpyxl 無法安裝 / 載入。")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    silhouette_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(f"✅ Gradio {gr.__version__}")
print("✅ CPU Analytics Environment Ready")
print("ℹ️ 本 Lab 不使用語言模型，不需要 GPU。")

def normalize_file_path(file_value):
    if file_value is None:
        return None
    if isinstance(file_value, (str, os.PathLike)):
        return os.fspath(file_value)
    if isinstance(file_value, dict):
        return file_value.get("path") or file_value.get("name")
    return getattr(file_value, "name", None)

def df_to_dark_html(df, empty_text="尚無資料"):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return f"<div class='yunfa-table-empty'>{empty_text}</div>"

    safe = df.copy()
    for col in safe.select_dtypes(include="number").columns:
        if pd.api.types.is_integer_dtype(safe[col]) or col in ("排程順位", "RuleBased順位", "Anomaly Rank", "工序序號"):
            safe[col] = safe[col].map(lambda v: "—" if pd.isna(v) else f"{v:,.0f}")
        else:
            safe[col] = safe[col].map(lambda v: "—" if pd.isna(v) else f"{v:,.1f}")
    html = safe.to_html(
        index=False,
        escape=True,
        border=0,
        classes="yunfa-data-table",
    )
    return "<div class='yunfa-table-wrap'>" + html + "</div>"

def kpi_cards(kpis):
    html = "<div class='kpi-wrap'>"
    for key, value in kpis.items():
        html += (
            "<div class='kpi-card'>"
            f"<div class='kpi-label'>{escape(str(key))}</div>"
            f"<div class='kpi-value'>{escape(str(value))}</div>"
            "</div>"
        )
    html += "</div>"
    return html

CUSTOM_CSS = """
:root { color-scheme: light; }
.gradio-container {
    max-width: 1440px !important;
    margin: auto !important;
    background: #fffdf9 !important;
    color: #273346 !important;
    font-family: "Noto Sans TC", "Microsoft JhengHei", sans-serif !important;
}
.yunfa-hero {
    padding: 24px 30px; margin: 4px 0 20px;
    border: 1px solid #e9ded1; border-radius: 18px;
    background: linear-gradient(115deg,#fff6e9,#f4f8f5);
}
.yunfa-hero h1 { margin: 0 0 8px; font-size: clamp(26px,3vw,36px); color: #263746; }
.yunfa-hero p { margin: 0; color: #63717d; font-size: 16px; }
.yunfa-section-title { font-size: 19px; font-weight: 800; margin: 16px 0 8px; color: #314555; }
.yunfa-unit-note { color: #64748b; margin: 5px 0 13px; font-size: 14px; }
.chart-meaning {
    margin: 12px 0 22px; padding: 17px 21px;
    border: 1px solid #dce7df; border-left: 5px solid #82b29b;
    border-radius: 12px; background: #f7fbf8; color: #334155;
    line-height: 1.75; font-size: 15px;
}
.chart-meaning strong { color: #2e5548; }
.chart-view img { object-fit: contain !important; }
.upload-instruction {
    border: 1px solid #e9ded1; border-radius: 14px;
    padding: 16px 20px; margin-bottom: 10px; background: #fff9f1;
}
.upload-instruction .file-kind {
    font-size: 19px; color: #594637;
    font-weight: 800;
    line-height: 1.35;
}
.upload-instruction .file-name {
    font-size: 17px;
    font-weight: 750;
    line-height: 1.4;
    margin-top: 4px;
}
.upload-instruction .file-note {
    font-size: 15px;
    color: #667484;
    margin-top: 4px;
}
.method-card {
    border: 1px solid #eadfd5; border-radius: 14px;
    padding: 18px 20px; min-height: 165px; background: #ffffff;
    box-shadow: 0 4px 16px rgba(83, 70, 56, .05);
}
.method-title {
    font-size: 19px; color: #36505b;
    font-weight: 800;
    margin-bottom: 6px;
}
.method-note {
    font-size: 15px;
    line-height: 1.7; color: #596978;
}
.yunfa-table-wrap {
    width: 100%;
    overflow-x: auto;
    border: 1px solid #e6e5e1;
    border-radius: 12px;
    background: #ffffff !important;
}
.yunfa-data-table {
    width: 100%;
    border-collapse: collapse;
    background: #ffffff !important;
    color: #334155 !important;
    font-size: 14px;
}
.yunfa-data-table thead th {
    background: #f5eee7 !important;
    color: #344454 !important;
    font-weight: 800 !important;
    border-right: 1px solid #eee8e1 !important;
    border-bottom: 1px solid #e3d8ca !important;
    padding: 10px 12px !important;
    white-space: nowrap;
    text-align: left;
}
.yunfa-data-table tbody td {
    background: #ffffff !important;
    color: #334155 !important;
    border-right: 1px solid #eeeef0 !important;
    border-bottom: 1px solid #eeeef0 !important;
    padding: 9px 12px !important;
    white-space: nowrap;
}
.yunfa-data-table tbody tr:nth-child(even) td {
    background: #fafbf9 !important;
}
.yunfa-data-table tbody tr:hover td {
    background: #fff3e4 !important;
    color: #263746 !important;
}
.yunfa-table-empty {
    border: 1px dashed #d9c9b9;
    border-radius: 10px;
    padding: 18px;
    background: #fffdf9 !important;
    color: #64748b !important;
    font-size: 15px;
}
.kpi-wrap {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 8px 0 14px 0;
}
.kpi-card {
    border: 1px solid #eadfd5;
    border-radius: 12px;
    padding: 14px 18px;
    background: #fff9f2 !important;
    color: #0f172a !important;
    min-width: 155px; flex: 1 1 155px;
}
.kpi-card * {
    color: #0f172a !important;
}
.kpi-label {
    font-size: 13px;
    font-weight: 700;
}
.kpi-value {
    font-size: 22px;
    font-weight: 850;
    margin-top: 3px;
}

.metric-help {
    border: 1px solid #e2e8e2;
    border-radius: 12px;
    padding: 14px 16px;
    background: #f6faf7 !important;
    color: #0f172a !important;
    line-height: 1.55;
}
.metric-help * { color: #0f172a !important; }
.metric-help-title {
    font-size: 18px;
    font-weight: 850;
    margin-bottom: 8px;
}
.metric-help-row { margin: 7px 0; }
.metric-help-name {
    font-weight: 800;
    color: #0f3b78 !important;
}
.metric-help-current {
    display: inline-block;
    margin-left: 6px;
    padding: 2px 8px;
    border-radius: 999px;
    background: #e9f0eb !important;
    color: #0f172a !important;
    font-weight: 800;
}
.metric-help-note {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #dbe4ee;
    font-size: 13px;
    color: #475569 !important;
}
@media (max-width: 700px) {
    .yunfa-hero { padding: 18px; }
    .kpi-card { min-width: 130px; }
    .chart-view { max-height: 370px; }
}
"""

def _safe_num(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _fig_to_gallery(fig, caption):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=135, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img = Image.open(buf).convert("RGB").copy()
    plt.close(fig)
    return (img, caption)

SCHED_CHART_CHOICES = [
    ("圖 1｜工單優先順序", 0),
    ("圖 2｜CNC 機台排程", 1),
    ("圖 3｜CNC 機台利用率", 2),
    ("圖 4｜急單與 VIP 排序比較", 3),
    ("圖 5｜FIFO 與規則排程效率", 4),
    ("圖 6｜工單排產條件篩選", 5),
]
ANOMALY_CHART_CHOICES = [
    ("圖 1｜異常事件優先檢視", 0),
    ("圖 2｜PCA 異常位置", 1),
    ("圖 3｜異常事件特徵偏差", 2),
    ("圖 4｜規則與 Isolation Forest 對照", 3),
]

def chart_meaning(content):
    return '<div class="chart-meaning"><strong>圖表意義與實務判讀</strong><br>' + escape(str(content)) + '</div>'

def select_chart(charts, choice):
    if not charts:
        return None, chart_meaning("請先上傳 MES 資料並執行分析。")
    try:
        index = int(choice)
    except (ValueError, TypeError):
        index = 0
    if not 0 <= index < len(charts):
        index = 0
    img, _title, meaning = charts[index]
    return img, chart_meaning(meaning)

def build_scheduling(mes_path):
    xls = pd.ExcelFile(mes_path)
    machine_df = pd.read_excel(xls, "機台主檔")
    route_df = pd.read_excel(xls, "工藝路由主檔")
    wo_df = pd.read_excel(xls, "生產工單")

    for c in ["開單日期","計畫開工日期","計畫完工日期","交期","實際開工日期","實際完工日期"]:
        if c in wo_df.columns: wo_df[c] = pd.to_datetime(wo_df[c], errors="coerce")
    for c in ["工序序號","標準換線工時_hr","標準加工工時_hr每100件"]:
        if c in route_df.columns: route_df[c] = pd.to_numeric(route_df[c], errors="coerce")
    for c in ["計畫數量"]:
        if c in wo_df.columns: wo_df[c] = pd.to_numeric(wo_df[c], errors="coerce")
    machine_df = _safe_num(machine_df, ["每日可用工時","效率目標"])

    active = wo_df[wo_df["狀態"] != "已完工"].copy()
    if len(active)==0: raise ValueError("MES 沒有未完工工單。")
    planning_date = active["開單日期"].min()
    active["距離交期天數"] = (active["交期"] - planning_date).dt.days.fillna(999).astype(int)

    cnc_route = route_df[route_df["需求工作中心"] == "CNC"].copy()
    cnc_agg = cnc_route.groupby("SKU").agg(
        CNC工序數=("工序序號","count"),
        CNC標準換線工時_hr=("標準換線工時_hr","sum"),
        CNC標準加工工時_hr每100件=("標準加工工時_hr每100件","sum")
    ).reset_index()
    active = active.merge(cnc_agg, on="SKU", how="left")
    active["有CNC路由"] = np.where(active["CNC工序數"].fillna(0)>0,"是","否")
    active["CNC工序數"] = active["CNC工序數"].fillna(0).astype(int)
    active["CNC標準換線工時_hr"] = active["CNC標準換線工時_hr"].fillna(0.)
    active["CNC標準加工工時_hr每100件"] = active["CNC標準加工工時_hr每100件"].fillna(0.)
    active["CNC加工工時_hr"] = active["CNC標準加工工時_hr每100件"] * active["計畫數量"] / 100
    active["CNC排程總工時_hr"] = active["CNC標準換線工時_hr"] + active["CNC加工工時_hr"]

    active["急單分數"] = active["急單"].apply(lambda x:30 if x=="是" else 0)
    def due(d): return 30 if d<=1 else 25 if d<=2 else 15 if d<=4 else 5 if d<=7 else 0
    active["交期分數"] = active["距離交期天數"].apply(due)
    if "客戶等級" not in active.columns: active["客戶等級"]="未設定"
    active["客戶分數"] = active["客戶等級"].astype(str).apply(lambda s:20 if "VIP" in s else 10 if "重要" in s else 0)
    active["材料分數"] = active["材料齊套"].apply(lambda x:15 if x=="是" else -35)
    active["工單優先等級分數"] = active["優先等級"].astype(str).map({"高":10,"中":5}).fillna(0)
    active["換線分數"] = active["CNC標準換線工時_hr"].apply(lambda h:10 if h<=1.5 else -5 if h>=3.5 else 0)
    active["品質分數"] = active["品質風險"].astype(str).map({"高":-15,"中":-5}).fillna(0)
    score_cols=["急單分數","交期分數","客戶分數","材料分數","工單優先等級分數","換線分數","品質分數"]
    active["排程分數"] = active[score_cols].sum(axis=1)
    active["可立即排產"] = np.where((active["材料齊套"]=="是")&(active["品質風險"]!="高")&(active["有CNC路由"]=="是"),"是","否")
    active["_ok"] = active["可立即排產"].eq("是")
    priority = active.sort_values(["_ok","排程分數","距離交期天數","CNC標準換線工時_hr","開單日期"], ascending=[False,False,True,True,True]).drop(columns="_ok").reset_index(drop=True)
    priority["排程順位"] = np.arange(1,len(priority)+1)

    cnc = machine_df[machine_df["工作中心"]=="CNC"].copy()
    if "可用" in cnc["狀態"].astype(str).values: cnc=cnc[cnc["狀態"]=="可用"]
    machines=sorted(cnc["機台ID"].astype(str).unique().tolist())
    if not machines: raise ValueError("找不到可用 CNC 機台。")

    sched=priority[priority["可立即排產"]=="是"].copy().reset_index(drop=True)
    def assign(df, rank_name):
        available={m:0.0 for m in machines}; rows=[]
        for i,(_,r) in enumerate(df.iterrows(),1):
            m=min(available,key=available.get); start=available[m]; dur=float(r["CNC排程總工時_hr"] or 0); finish=start+dur; available[m]=finish
            z=r.to_dict(); z.update({"機台":m,"開始工時":round(start,2),"排程總工時":round(dur,2),"完成工時":round(finish,2),rank_name:i}); rows.append(z)
        return pd.DataFrame(rows, columns=[*df.columns, "機台", "開始工時", "排程總工時", "完成工時", rank_name])
    schedule=assign(sched,"RuleBased順位")
    fifo_base=sched.sort_values(["開單日期","工單ID"]).reset_index(drop=True)
    fifo=assign(fifo_base,"FIFO順位")
    if len(schedule):
        rank_compare=schedule[["工單ID","RuleBased順位"]].merge(fifo[["工單ID","FIFO順位"]],on="工單ID")
        rank_compare["順位提前"]=rank_compare["FIFO順位"]-rank_compare["RuleBased順位"]
    else: rank_compare=pd.DataFrame(columns=["工單ID","RuleBased順位","FIFO順位","順位提前"])

    fifo_makespan=float(fifo["完成工時"].max()) if len(fifo) else 0
    rb_makespan=float(schedule["完成工時"].max()) if len(schedule) else 0
    fifo_hours=fifo.groupby("機台")["排程總工時"].sum().reindex(machines,fill_value=0.) if len(fifo) else pd.Series(0.,index=machines)
    rb_hours=schedule.groupby("機台")["排程總工時"].sum().reindex(machines,fill_value=0.) if len(schedule) else pd.Series(0.,index=machines)
    fifo_util=fifo_hours/fifo_makespan*100 if fifo_makespan>0 else fifo_hours*0
    rb_util=rb_hours/rb_makespan*100 if rb_makespan>0 else rb_hours*0
    util_df=pd.DataFrame({"CNC機台":machines,"FIFO利用率(%)":fifo_util.values,"Rule-Based利用率(%)":rb_util.values})

    urgent_rb=schedule.loc[schedule.get("急單",pd.Series(dtype=str)).eq("是"),"RuleBased順位"].mean() if len(schedule) else np.nan
    urgent_fifo=fifo.loc[fifo.get("急單",pd.Series(dtype=str)).eq("是"),"FIFO順位"].mean() if len(fifo) else np.nan
    material_hold=priority[priority["材料齊套"]=="否"].copy()
    quality_hold=priority[priority["品質風險"]=="高"].copy()

    return {"priority":priority,"schedule":schedule,"fifo":fifo,"rank_compare":rank_compare,"util":util_df,
            "material_hold":material_hold,"quality_hold":quality_hold,
            "kpis":{"未完工工單":len(priority),"可立即排產":len(sched),"待料工單":len(material_hold),"高品質風險":len(quality_hold),
                    "Rule-Based Makespan(hr)":round(rb_makespan,2),"FIFO Makespan(hr)":round(fifo_makespan,2),
                    "Rule-Based平均利用率(%)":round(float(rb_util.mean()),1),"FIFO平均利用率(%)":round(float(fifo_util.mean()),1),
                    "急單Rule-Based平均順位":None if pd.isna(urgent_rb) else round(float(urgent_rb),2),"急單FIFO平均順位":None if pd.isna(urgent_fifo) else round(float(urgent_fifo),2)}}

def build_anomaly(mes_path):
    """
    與原始 YUNFA_MES_IsolationForest.py 對齊：
    - 2023~2025 歷史基準、2026 偵測
    - SKU + 工序作 primary context，工作中心作 fallback
    - 11 個原始特徵
    - Isolation Forest contamination=0.05, n_estimators=400
    - Rule-Based baseline 僅供對照
    """
    HISTORY_YEARS = [2023, 2024, 2025]
    DETECTION_YEAR = 2026
    CONTAMINATION = 0.05
    RANDOM_STATE = 42

    xls = pd.ExcelFile(mes_path)
    machine_df = pd.read_excel(xls, "機台主檔")
    wo_df = pd.read_excel(xls, "生產工單")
    report_df = pd.read_excel(xls, "生產報工")
    equip_df = pd.read_excel(xls, "機台稼動日報")
    quality_abn_df = pd.read_excel(xls, "品質異常")

    for c in ["計畫開始時間","計畫結束時間","實際開始時間","實際結束時間"]:
        if c in report_df.columns:
            report_df[c] = pd.to_datetime(report_df[c], errors="coerce")
    for c in ["開單日期","計畫開工日期","計畫完工日期","交期","實際開工日期","實際完工日期"]:
        if c in wo_df.columns:
            wo_df[c] = pd.to_datetime(wo_df[c], errors="coerce")
    equip_df["日期"] = pd.to_datetime(equip_df["日期"], errors="coerce").dt.normalize()
    if "發現日期" in quality_abn_df.columns:
        quality_abn_df["發現日期"] = pd.to_datetime(quality_abn_df["發現日期"], errors="coerce")

    report_df = _safe_num(report_df, [
        "工序序號","標準換線工時_hr","實際換線工時_hr","標準加工工時_hr",
        "實際加工工時_hr","停機工時_hr","良品數量","報廢數量","重工數量","不良率"
    ])
    machine_df = _safe_num(machine_df, ["每日可用工時","效率目標","設備年齡_年"])
    equip_df = _safe_num(equip_df, [
        "可用工時_hr","運轉工時_hr","換線工時_hr","停機工時_hr",
        "稼動率","品質良率","OEE","故障次數"
    ])

    events = report_df.copy()
    wo_keep = [
        "工單ID","產品線","開單日期","計畫開工日期","計畫完工日期","交期",
        "計畫數量","急單","優先等級","材料齊套","品質風險","狀態"
    ]
    events = events.merge(wo_df[wo_keep], on="工單ID", how="left")
    events["工單年度"] = events["開單日期"].dt.year
    events["報工日期"] = events["實際開始時間"].dt.normalize()

    machine_keep = [
        "機台ID","機台類型","工作中心","能力群組","每日可用工時",
        "效率目標","設備年齡_年","狀態"
    ]
    machine_merge = machine_df[machine_keep].copy().rename(columns={"工作中心":"機台工作中心"})
    events = events.merge(machine_merge, on="機台ID", how="left")

    equip_keep = [
        "日期","機台ID","可用工時_hr","運轉工時_hr","換線工時_hr","停機工時_hr",
        "稼動率","品質良率","OEE","故障次數"
    ]
    equip_merge = equip_df[equip_keep].rename(
        columns={"日期":"報工日期","停機工時_hr":"設備日停機工時_hr"}
    )
    events = events.merge(equip_merge, on=["報工日期","機台ID"], how="left")

    quality_flag = (
        quality_abn_df.groupby(["工單ID","機台ID"])
        .agg(
            品質異常單數=("異常單ID","count"),
            品質異常影響數量=("受影響數量","sum"),
        )
        .reset_index()
    )
    events = events.merge(quality_flag, on=["工單ID","機台ID"], how="left")
    events["品質異常單數"] = events["品質異常單數"].fillna(0)
    events["品質異常影響數量"] = events["品質異常影響數量"].fillna(0)

    def safe_divide(a, b):
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        return a / b.replace(0, np.nan)

    events["換線超時率"] = safe_divide(
        events["實際換線工時_hr"] - events["標準換線工時_hr"],
        events["標準換線工時_hr"]
    )
    events["加工超時率"] = safe_divide(
        events["實際加工工時_hr"] - events["標準加工工時_hr"],
        events["標準加工工時_hr"]
    )
    events["停機占比"] = safe_divide(events["停機工時_hr"], events["每日可用工時"])
    events["總產出數量"] = (
        events["良品數量"].fillna(0) + events["報廢數量"].fillna(0) + events["重工數量"].fillna(0)
    )
    events["報廢率"] = safe_divide(events["報廢數量"], events["總產出數量"]).fillna(0)
    events["重工率"] = safe_divide(events["重工數量"], events["總產出數量"]).fillna(0)
    events["良品產出率_件每時"] = safe_divide(events["良品數量"], events["實際加工工時_hr"])

    history_mask = events["工單年度"].isin(HISTORY_YEARS)
    detect_mask = events["工單年度"].eq(DETECTION_YEAR)
    if history_mask.sum() == 0:
        raise ValueError(f"找不到歷史年度 {HISTORY_YEARS} 的工單資料。")
    if detect_mask.sum() == 0:
        raise ValueError(f"找不到 {DETECTION_YEAR} 年工單資料。")

    def add_context_z(full_df, history_df, value_col, primary_group, fallback_group, output_col):
        global_mean = history_df[value_col].mean()
        global_std = history_df[value_col].std()

        p = (
            history_df.groupby(primary_group)[value_col]
            .agg(["mean","std"]).reset_index()
            .rename(columns={"mean":"_p_mean","std":"_p_std"})
        )
        f = (
            history_df.groupby(fallback_group)[value_col]
            .agg(["mean","std"]).reset_index()
            .rename(columns={"mean":"_f_mean","std":"_f_std"})
        )

        out = full_df.merge(p, on=primary_group, how="left")
        out = out.merge(f, on=fallback_group, how="left")

        mean_ref = out["_p_mean"].copy()
        std_ref = out["_p_std"].copy()
        weak = mean_ref.isna() | std_ref.isna() | (std_ref.abs() < 1e-9)
        mean_ref = mean_ref.where(~weak, out["_f_mean"])
        std_ref = std_ref.where(~weak, out["_f_std"])

        weak2 = mean_ref.isna() | std_ref.isna() | (std_ref.abs() < 1e-9)
        if pd.isna(global_std) or abs(global_std) < 1e-9:
            global_std = 1.0
        mean_ref = mean_ref.where(~weak2, global_mean)
        std_ref = std_ref.where(~weak2, global_std)

        out[output_col] = (
            (out[value_col] - mean_ref) / std_ref
        ).replace([np.inf,-np.inf], np.nan).fillna(0)
        return out.drop(columns=["_p_mean","_p_std","_f_mean","_f_std"], errors="ignore")

    for value_col, output_col in [
        ("換線超時率","換線情境偏差_Z"),
        ("加工超時率","加工情境偏差_Z"),
        ("停機占比","停機情境偏差_Z"),
        ("報廢率","報廢情境偏差_Z"),
        ("重工率","重工情境偏差_Z"),
        ("良品產出率_件每時","產出效率偏差_Z"),
    ]:
        events = add_context_z(
            full_df=events,
            history_df=events.loc[history_mask].copy(),
            value_col=value_col,
            primary_group=["SKU","工序序號"],
            fallback_group=["工作中心"],
            output_col=output_col,
        )

    FEATURES = [
        "換線超時率","加工超時率","停機占比","報廢率","重工率",
        "換線情境偏差_Z","加工情境偏差_Z","停機情境偏差_Z",
        "報廢情境偏差_Z","重工情境偏差_Z","產出效率偏差_Z"
    ]
    FEATURE_EN = {
        "換線超時率":"Setup Overrun",
        "加工超時率":"Processing Overrun",
        "停機占比":"Downtime Ratio",
        "報廢率":"Scrap Rate",
        "重工率":"Rework Rate",
        "換線情境偏差_Z":"Context Setup Deviation",
        "加工情境偏差_Z":"Context Processing Deviation",
        "停機情境偏差_Z":"Context Downtime Deviation",
        "報廢情境偏差_Z":"Context Scrap Deviation",
        "重工情境偏差_Z":"Context Rework Deviation",
        "產出效率偏差_Z":"Throughput Deviation",
    }

    events[FEATURES] = events[FEATURES].replace([np.inf,-np.inf], np.nan)
    history_medians = events.loc[history_mask, FEATURES].median(numeric_only=True)
    events[FEATURES] = events[FEATURES].fillna(history_medians).fillna(0)

    train_df = events.loc[history_mask].copy().reset_index(drop=True)
    detect_df = events.loc[detect_mask].copy().reset_index(drop=True)
    X_train = train_df[FEATURES].copy()
    X_detect = detect_df[FEATURES].copy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_detect_scaled = scaler.transform(X_detect)

    iso = IsolationForest(
        n_estimators=400,
        max_samples="auto",
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_train_scaled)

    train_score_raw = -iso.score_samples(X_train_scaled)
    detect_score_raw = -iso.score_samples(X_detect_scaled)
    detect_df["IF判定"] = np.where(iso.predict(X_detect_scaled) == -1, "異常", "正常")
    sorted_train_scores = np.sort(train_score_raw)
    detect_df["Anomaly Score"] = [
        100.0 * np.searchsorted(sorted_train_scores, score, side="right") / len(sorted_train_scores)
        for score in detect_score_raw
    ]
    detect_df["Anomaly Rank"] = (
        detect_df["Anomaly Score"].rank(method="first", ascending=False).astype(int)
    )
    detect_df["異常層級"] = np.select(
        [
            detect_df["IF判定"].eq("異常"),
            detect_df["Anomaly Score"].ge(95),
        ],
        ["High Anomaly","Review"],
        default="Normal",
    )

    rule_specs = {
        "換線超時率":("high",0.95),
        "加工超時率":("high",0.95),
        "停機占比":("high",0.95),
        "報廢率":("high",0.95),
        "重工率":("high",0.95),
        "產出效率偏差_Z":("low",0.05),
    }
    rule_flags = pd.DataFrame(index=detect_df.index)
    for feature, (direction, q) in rule_specs.items():
        threshold = train_df[feature].quantile(q)
        rule_flags[feature] = (
            detect_df[feature] > threshold if direction == "high"
            else detect_df[feature] < threshold
        )
    detect_df["Rule Flag Count"] = rule_flags.sum(axis=1)
    detect_df["Rule判定"] = np.where(detect_df["Rule Flag Count"] >= 1, "Warning", "Normal")
    detect_df["方法對照"] = np.select(
        [
            detect_df["Rule判定"].eq("Warning") & detect_df["IF判定"].eq("異常"),
            detect_df["Rule判定"].eq("Normal") & detect_df["IF判定"].eq("異常"),
            detect_df["Rule判定"].eq("Warning") & detect_df["IF判定"].eq("正常"),
        ],
        ["共同警示","多變數隱性異常","單變數規則警示"],
        default="共同正常",
    )
    detect_df["有品質異常紀錄"] = np.where(detect_df["品質異常單數"] > 0, "是", "否")

    # 原始程式 PCA：歷史 + 2026 一起投影
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(np.vstack([X_train_scaled, X_detect_scaled]))
    train_pca = pd.DataFrame(all_pca[:len(X_train_scaled)], columns=["PC1","PC2"])
    detect_pca = all_pca[len(X_train_scaled):]
    detect_df["PC1"] = detect_pca[:,0]
    detect_df["PC2"] = detect_pca[:,1]

    # 每筆事件前三大偏差特徵
    top_features = []
    for row_scaled in X_detect_scaled:
        inds = np.argsort(np.abs(row_scaled))[::-1][:3]
        top_features.append("、".join([FEATURES[j] for j in inds]))
    detect_df["主要偏差特徵"] = top_features

    comparison_table = (
        pd.crosstab(detect_df["Rule判定"], detect_df["IF判定"])
        .reindex(index=["Normal","Warning"], columns=["正常","異常"], fill_value=0)
    )

    top_idx = detect_df["Anomaly Score"].idxmax()
    top_scaled = X_detect_scaled[top_idx]
    profile_df = pd.DataFrame({
        "特徵": FEATURES,
        "英文名稱": [FEATURE_EN[f] for f in FEATURES],
        "原始值": [detect_df.loc[top_idx, f] for f in FEATURES],
        "相對歷史標準化偏差": top_scaled,
    })
    profile_df["絕對偏差程度"] = profile_df["相對歷史標準化偏差"].abs()
    profile_df = profile_df.sort_values("絕對偏差程度", ascending=False)

    detect_sorted = detect_df.sort_values("Anomaly Score", ascending=False).reset_index(drop=True)
    high_count = int((detect_sorted["IF判定"] == "異常").sum())
    review_count = int((detect_sorted["異常層級"] == "Review").sum())

    return {
        "data": detect_sorted,
        "features": FEATURES,
        "feature_en": FEATURE_EN,
        "detect_year": DETECTION_YEAR,
        "history_years": HISTORY_YEARS,
        "train_pca": train_pca,
        "comparison_table": comparison_table,
        "profile_df": profile_df,
        "kpis": {
            "歷史基準事件": len(train_df),
            "偵測年度": DETECTION_YEAR,
            "偵測事件": len(detect_sorted),
            "High Anomaly": high_count,
            "Review": review_count,
            "多變數隱性異常": int((detect_sorted["方法對照"] == "多變數隱性異常").sum()),
        },
    }

def scheduling_original_charts(x):
    charts = []
    priority_df = x["priority"]
    schedule = x["schedule"]
    fifo = x["fifo"]
    util = x["util"]
    k = x["kpis"]
    machines = util["CNC機台"].astype(str).tolist()

    # Original CHART 1
    top15 = priority_df.head(15)
    fig, ax = plt.subplots(figsize=(10,7.5), facecolor="white")
    ordered = top15.iloc[::-1]
    bars = ax.barh(ordered["工單ID"].astype(str), ordered["排程分數"], color="#638fad", height=0.62)
    ax.set_xlim(0, max(100, float(top15["排程分數"].max()) * 1.14))
    for bar in bars:
        ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                f"{bar.get_width():.0f}", ha="left", va="center", fontsize=9)
    ax.set_title("Top 15 Work Orders by Priority Score", fontweight="bold")
    ax.set_xlabel("Priority Score"); ax.set_ylabel("Work Order")
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    charts.append(_fig_to_gallery(fig, "Scheduling 1｜Top 15 Work Orders by Priority Score"))
    first = top15.iloc[0] if len(top15) else None
    charts[-1] += (f"本圖依排程分數顯示前 {len(top15)} 張未完工工單；本次最優先檢視的是工單 {first['工單ID']}（{first['排程分數']:.0f} 分）。分數綜合交期、急單、客戶等級、材料與品質條件；實際派工仍須確認工單可立即排產。後續 AI Agent 可依順位彙整交期壓力與工單阻礙，提供生管人員確認。" if first is not None else "目前沒有可供排序的未完工工單。",)

    # Original CHART 2 - Gantt
    fig, ax = plt.subplots(figsize=(12,6), facecolor="white")
    colors = ["#2ecc71","#3498db","#9b59b6","#e67e22","#1abc9c","#e74c3c","#34495e","#f1c40f"]
    shown_schedule = schedule.head(18)
    for idx, row in shown_schedule.iterrows():
        m_idx = machines.index(str(row["機台"]))
        color = colors[idx % len(colors)]
        ax.barh(m_idx, row["排程總工時"], left=row["開始工時"], height=0.55,
                align="center", color=color, edgecolor="black", alpha=0.85)
        ax.text(row["開始工時"] + row["排程總工時"]/2, m_idx, str(row["工單ID"]),
                ha="center", va="center", fontsize=8, color="black")
    rb_makespan = float(k["Rule-Based Makespan(hr)"] or 0)
    for h in range(8, int(np.ceil(rb_makespan))+8, 8):
        ax.axvline(x=h, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(machines))); ax.set_yticklabels(machines)
    ax.set_xlabel("Scheduled CNC Hours"); ax.set_ylabel("CNC Machine")
    ax.set_title("MES Rule-Based CNC Production Schedule", fontweight="bold")
    fig.tight_layout()
    charts.append(_fig_to_gallery(fig, "Scheduling 2｜Rule-Based CNC Production Schedule"))
    charts[-1] += (f"本圖展示前 {len(shown_schedule)} 張可排產工單在各 CNC 機台上的開始時間及預計加工時段；全部可排產工單共 {len(schedule)} 張，規則排程總完成工時為 {rb_makespan:.1f} 小時。後續 AI Agent 可比對交期與機台負荷，提醒生管人員檢查可能衝突的機台與工單。橫軸為累計排程工時，並非實際日曆時間。",)

    # Original CHART 3
    fig, ax = plt.subplots(figsize=(10,5), facecolor="white")
    bars = ax.bar(util["CNC機台"], util["Rule-Based利用率(%)"], color="#2ecc71", width=0.55)
    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, y+1, f"{y:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(float(util["Rule-Based利用率(%)"].max())+15,100))
    ax.set_title("CNC Utilization after Rule-Based Scheduling", fontweight="bold")
    ax.set_xlabel("CNC Machine"); ax.set_ylabel("Utilization (%)")
    fig.tight_layout()
    charts.append(_fig_to_gallery(fig, "Scheduling 3｜CNC Utilization after Rule-Based Scheduling"))
    busiest = util.loc[util["Rule-Based利用率(%)"].idxmax()]
    charts[-1] += (f"本圖比較各 CNC 機台在規則排程下的工時利用率；本次 {busiest['CNC機台']} 最高，為 {busiest['Rule-Based利用率(%)']:.1f}%。AI Agent 可依機台負荷提出分流與維護時段檢視項目；此利用率以排程總完成工時為基準，並非機台實際稼動率。",)

    # Original CHART 4
    rank_compare = schedule[["工單ID","RuleBased順位"]].merge(
        fifo[["工單ID","FIFO順位"]], on="工單ID", how="inner"
    )
    merged = schedule.merge(rank_compare[["工單ID","FIFO順位"]], on="工單ID", how="left")
    urgent = merged["急單"].eq("是")
    vip = merged["客戶等級"].astype(str).str.contains("VIP", na=False)
    urgent_fifo = float(merged.loc[urgent,"FIFO順位"].mean()) if urgent.any() else 0.0
    urgent_rb = float(merged.loc[urgent,"RuleBased順位"].mean()) if urgent.any() else 0.0
    vip_fifo = float(merged.loc[vip,"FIFO順位"].mean()) if vip.any() else 0.0
    vip_rb = float(merged.loc[vip,"RuleBased順位"].mean()) if vip.any() else 0.0
    fig, ax = plt.subplots(figsize=(8,5), facecolor="white")
    xx=np.arange(2); width=.35
    b1=ax.bar(xx-width/2,[urgent_fifo,vip_fifo],width,label="FIFO",color="#95a5a6")
    b2=ax.bar(xx+width/2,[urgent_rb,vip_rb],width,label="Rule-Based",color="#3498db")
    for bar in list(b1)+list(b2):
        y=bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2,y+.2,f"{y:.1f}",ha="center",va="bottom")
    ax.set_xticks(xx); ax.set_xticklabels(["Urgent Avg Rank","VIP Avg Rank"])
    ax.set_ylabel("Average Dispatch Rank")
    ax.set_title("FIFO vs Rule-Based Dispatch Priority", fontweight="bold")
    ax.legend(frameon=False); fig.tight_layout()
    charts.append(_fig_to_gallery(fig, "Scheduling 4｜FIFO vs Rule-Based Dispatch Priority"))
    charts[-1] += (f"本圖比較先進先出（FIFO）與規則排程對急單及 VIP 工單的平均派工順位；急單平均順位由 {urgent_fifo:.1f} 變為 {urgent_rb:.1f}，VIP 工單由 {vip_fifo:.1f} 變為 {vip_rb:.1f}。數字越小代表越早派工。AI Agent 可據此核對重要工單是否獲得符合交期及客戶承諾的排序；若沒有相應工單，圖中的 0 僅為空組占位值。",)

    # Original CHART 5
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="white")
    for ax, values, title, unit in (
        (axes[0], [float(k["FIFO Makespan(hr)"]), float(k["Rule-Based Makespan(hr)"])], "Total Completion Time", "Hours"),
        (axes[1], [float(k["FIFO平均利用率(%)"]), float(k["Rule-Based平均利用率(%)"])], "Average CNC Utilization", "Utilization (%)"),
    ):
        bars = ax.bar(["FIFO", "Rule-Based"], values, color=["#a9b9bf", "#86b3a0"], width=.54)
        ax.set_ylim(0, max(values + [1]) * 1.22)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(unit)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(values+[1])*.02,
                    f"{bar.get_height():.1f}", ha="center", fontsize=10)
    fig.suptitle("FIFO vs Rule-Based Scheduling Efficiency", fontweight="bold")
    fig.tight_layout()
    charts.append(_fig_to_gallery(fig, "Scheduling 5｜FIFO vs Rule-Based Scheduling Efficiency"))
    charts[-1] += (f"左圖以小時比較完成全部可排工單所需的排程時間：FIFO 為 {float(k['FIFO Makespan(hr)']):.1f} 小時，規則排程為 {float(k['Rule-Based Makespan(hr)']):.1f} 小時；右圖分別比較 CNC 平均利用率 {float(k['FIFO平均利用率(%)']):.1f}% 與 {float(k['Rule-Based平均利用率(%)']):.1f}%。AI Agent 可同時檢視交期與機台分配，提出需人工確認的派工調整；兩個指標單位不同，分圖判讀。",)

    # Original CHART 6
    fig, ax = plt.subplots(figsize=(9,5), facecolor="white")
    categories=["Unfinished","Ready","Material Hold","Quality Risk"]
    counts=[len(priority_df), int(k["可立即排產"]), len(x["material_hold"]), len(x["quality_hold"])]
    cols=["#3498db","#2ecc71","#e67e22","#e74c3c"]
    bars=ax.bar(categories,counts,color=cols,width=.55)
    for bar in bars:
        y=bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2,y+.5,f"{int(y)}",ha="center",va="bottom")
    ax.set_ylabel("Number of Work Orders")
    ax.set_title("MES Work Order Screening for Production Scheduling",fontweight="bold")
    fig.tight_layout()
    charts.append(_fig_to_gallery(fig, "Scheduling 6｜MES Work Order Screening"))
    charts[-1] += (f"本次未完工工單 {len(priority_df)} 張，其中 {int(k['可立即排產'])} 張符合材料齊套、品質條件且有 CNC 路由；待料 {len(x['material_hold'])} 張，高品質風險 {len(x['quality_hold'])} 張。AI Agent 可將無法排產的工單分別交由採購或品管檢視，待條件確認後再更新排程。待料與品質風險可能同時出現在同一張工單，柱數不可直接相加。",)
    return charts

def anomaly_original_charts(x):
    charts=[]
    d=x["data"]
    top=d.head(15).sort_values("Anomaly Score",ascending=True).copy()
    top["EventLabel"]="WO "+top["工單ID"].astype(str)+" | OP "+top["工序序號"].astype(str)+" | MC "+top["機台ID"].astype(str)

    # Original CHART 1
    fig,ax=plt.subplots(figsize=(12,7),facecolor="white")
    colors=["#D9534F" if z=="High Anomaly" else "#E8D58A" for z in top["異常層級"]]
    bars=ax.barh(top["EventLabel"],top["Anomaly Score"],color=colors,edgecolor="white")
    for bar,val in zip(bars,top["Anomaly Score"]):
        ax.text(val+.5,bar.get_y()+bar.get_height()/2,f"{val:.1f}",va="center",fontsize=9)
    ax.set_xlabel("Anomaly Score (Historical Percentile)")
    ax.set_ylabel("Work Order | Operation | Machine")
    ax.set_title(f"Top 15 MES Anomaly Events - {x['detect_year']}",fontweight="bold")
    ax.set_xlim(0,105); ax.grid(axis="x",alpha=.35); fig.tight_layout()
    charts.append(_fig_to_gallery(fig,"Anomaly 1｜Top 15 MES Anomaly Events"))
    highest = d.iloc[0]
    charts[-1] += (f"本圖依相對歷史異常分數列出前 {len(top)} 筆製程事件；最高的是工單 {highest['工單ID']} 的第 {highest['工序序號']} 道工序，機台 {highest['機台ID']}，分數 {highest['Anomaly Score']:.1f}。AI Agent 可依排序調閱報工、停機與品質紀錄並交由製造端查核。此分數是相對歷史資料的檢視順位，不是設備故障機率。",)

    # Original CHART 2
    fig,ax=plt.subplots(figsize=(11,7),facecolor="white")
    tr=x["train_pca"]
    ax.scatter(tr["PC1"],tr["PC2"],s=28,alpha=.22,color="#D9D9D9",label="Historical 2023-2025")
    normal=d["IF判定"].eq("正常"); abn=d["IF判定"].eq("異常")
    ax.scatter(d.loc[normal,"PC1"],d.loc[normal,"PC2"],s=50,alpha=.72,color="#A8C69F",
               edgecolors="white",linewidths=.6,label="2026 Normal")
    ax.scatter(d.loc[abn,"PC1"],d.loc[abn,"PC2"],s=95,alpha=.95,color="#D9534F",
               edgecolors="white",linewidths=.8,label="2026 Anomaly")
    ax.set_xlabel("Principal Component 1"); ax.set_ylabel("Principal Component 2")
    ax.set_title("MES Multi-Variable Anomaly Map - PCA",fontweight="bold")
    ax.legend(frameon=False); ax.grid(True,alpha=.35); fig.tight_layout()
    charts.append(_fig_to_gallery(fig,"Anomaly 2｜PCA 2D MES Anomaly Map"))
    flagged = int(abn.sum())
    charts[-1] += (f"灰色點為歷史製程事件，綠色點為本次一般事件，紅色點為 Isolation Forest 標記的 {flagged} 筆異常事件。相近位置代表多項製程特徵的投影較接近；AI Agent 可定位與歷史事件差異明顯的工單，再連結工序與機台紀錄供現場判讀。PCA 橫縱軸為綜合特徵，並非工時或良率的原始單位。",)

    # Original CHART 3
    prof=x["profile_df"].sort_values("相對歷史標準化偏差",ascending=True)
    fig,ax=plt.subplots(figsize=(12,6.5),facecolor="white")
    cols=["#D9534F" if abs(v)>=2 else "#F2C6A0" for v in prof["相對歷史標準化偏差"]]
    bars=ax.barh(prof["英文名稱"],prof["相對歷史標準化偏差"],color=cols,edgecolor="white")
    ax.axvline(0,color="#888888",linestyle="--",linewidth=1)
    for bar,val in zip(bars,prof["相對歷史標準化偏差"]):
        ax.text(val+(.08 if val>=0 else -.08),bar.get_y()+bar.get_height()/2,f"{val:.2f}",
                va="center",ha="left" if val>=0 else "right",fontsize=9)
    ax.set_xlabel("Standardized Deviation from Historical Baseline")
    ax.set_ylabel("MES Feature"); ax.set_title("Top Anomaly - Multi-Factor Deviation Profile",fontweight="bold")
    ax.grid(axis="x",alpha=.35); fig.tight_layout()
    charts.append(_fig_to_gallery(fig,"Anomaly 3｜Top Anomaly Multi-Factor Deviation Profile"))
    largest = x["profile_df"].sort_values("絕對偏差程度", ascending=False).iloc[0]
    charts[-1] += (f"本圖檢視異常分數最高事件的各項製程特徵，相對歷史基準的標準化偏差；偏離最大的是 {largest['特徵']}（{largest['相對歷史標準化偏差']:+.1f}）。AI Agent 可將偏差較大的換線、加工、停機或品質指標與該筆報工紀錄對照，整理現場應先核對的項目；正負號表示相對基準方向，不能單獨判斷故障原因。",)

    # Original CHART 4
    matrix=x["comparison_table"].values.astype(float)
    fig,ax=plt.subplots(figsize=(7.5,5.5),facecolor="white")
    im=ax.imshow(matrix,cmap="YlOrRd",aspect="auto")
    ax.set_xticks([0,1]); ax.set_xticklabels(["IF Normal","IF Anomaly"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["Rule Normal","Rule Warning"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j,i,f"{int(matrix[i,j])}",ha="center",va="center",fontsize=16,fontweight="bold",color="#3F3F3F")
    ax.set_title("Rule-Based vs Isolation Forest",fontweight="bold")
    fig.colorbar(im,ax=ax,label="Number of Events"); fig.tight_layout()
    charts.append(_fig_to_gallery(fig,"Anomaly 4｜Rule-Based vs Isolation Forest Matrix"))
    shared = int(matrix[1,1]); hidden = int(matrix[0,1]); rule_only = int(matrix[1,0])
    charts[-1] += (f"本圖比較單項規則與 Isolation Forest 的判定：共同警示 {shared} 筆、僅模型判為異常 {hidden} 筆、僅規則提出警示 {rule_only} 筆。AI Agent 可先檢視共同警示的報工事件，再整理兩種方法判定不一致的製程條件交由人員確認；判定不一致代表檢查角度不同，並不直接表示某一方法判錯。",)
    return charts


# ------------------------------------------------------------
# MES Manual Analytics UI
# ------------------------------------------------------------
def run_scheduling_web(mes_file, progress=gr.Progress()):
    path = normalize_file_path(mes_file)
    if not path:
        return (
            "❌ 請先上傳 YUNFA_MES.xlsx。",
            "",
            df_to_dark_html(pd.DataFrame(), "尚無排程結果"),
            [], gr.update(value=0), None, chart_meaning("請先上傳 MES 檔案並執行排程分析。")
        )

    t0 = time.time()
    try:
        progress(0.15, desc="讀取 MES 與工藝路由")
        result = build_scheduling(path)
        progress(0.60, desc="執行 Rule-Based Scheduling")

        table = result["schedule"].copy()
        cols = [
            c for c in [
                "RuleBased順位","工單ID","SKU","客戶等級","急單",
                "材料齊套","品質風險","距離交期天數","排程分數",
                "機台","開始工時","完成工時"
            ] if c in table.columns
        ]
        table = table[cols].head(20)

        progress(0.82, desc="產生 6 張排程圖表")
        charts = scheduling_original_charts(result)

        elapsed = time.time() - t0
        status = (
            "✅ Rule-Based Scheduling 完成  \n"
            f"CPU 執行時間：**{elapsed:.2f} 秒**"
        )
        progress(1.0, desc="完成")
        return (status, kpi_cards(result["kpis"]), df_to_dark_html(table),
                charts, gr.update(value=0), *select_chart(charts, 0))
    except Exception as e:
        return (
            f"❌ Scheduling 執行失敗：{type(e).__name__}: {e}",
            "",
            df_to_dark_html(pd.DataFrame(), "Scheduling 執行失敗"),
            [], gr.update(value=0), None, chart_meaning("排程未完成，請檢查上方錯誤訊息。")
        )

def run_anomaly_web(mes_file, progress=gr.Progress()):
    path = normalize_file_path(mes_file)
    if not path:
        return (
            "❌ 請先上傳 YUNFA_MES.xlsx。",
            "",
            df_to_dark_html(pd.DataFrame(), "尚無異常分析結果"),
            [], gr.update(value=0), None, chart_meaning("請先上傳 MES 檔案並執行異常分析。")
        )

    t0 = time.time()
    try:
        progress(0.15, desc="讀取 MES 歷史與 2026 資料")
        result = build_anomaly(path)
        progress(0.62, desc="執行 Isolation Forest")

        table = result["data"].copy()
        cols = [
            c for c in [
                "Anomaly Rank","工單ID","工序序號","工序名稱","機台ID",
                "Anomaly Score","異常層級","Rule判定","IF判定",
                "方法對照","主要偏差特徵","OEE","品質良率"
            ] if c in table.columns
        ]
        table = table[cols].head(15)

        progress(0.82, desc="產生 4 張異常分析圖表")
        charts = anomaly_original_charts(result)

        elapsed = time.time() - t0
        status = (
            "✅ Isolation Forest 完成  \n"
            f"CPU 執行時間：**{elapsed:.2f} 秒**  \n"
            "Anomaly Score 代表相對異常優先程度，不是設備故障機率。"
        )
        progress(1.0, desc="完成")
        return (status, kpi_cards(result["kpis"]), df_to_dark_html(table),
                charts, gr.update(value=0), *select_chart(charts, 0))
    except Exception as e:
        return (
            f"❌ Isolation Forest 執行失敗：{type(e).__name__}: {e}",
            "",
            df_to_dark_html(pd.DataFrame(), "Isolation Forest 執行失敗"),
            [], gr.update(value=0), None, chart_meaning("異常分析未完成，請檢查上方錯誤訊息。")
        )

def create_mes_app():
    with gr.Blocks(
        title="WebBased AI 數據戰情室應用實作",
        css=CUSTOM_CSS,
    ) as demo:
        gr.HTML("""<div class="yunfa-hero"><h1>WebBased AI 數據戰情室應用實作</h1>
                <p>YUNFA MES｜CNC 生產排程 · 製程異常檢視｜Rule-Based Scheduling + Isolation Forest</p></div>""")

        gr.HTML(
            """
            <div class="upload-instruction">
              <div class="file-kind">🏭 MES 生產資料</div>
              <div class="file-name">請上傳：YUNFA_MES.xlsx</div>
              <div class="file-note">包含工單、工藝路由、機台、報工、稼動與品質異常資料；上傳後自動分析，也可按下方按鈕重新執行。</div>
            </div>
            """
        )
        mes_file = gr.File(
            label="選擇 YUNFA_MES.xlsx",
            type="filepath",
            file_types=[".xlsx"],
        )
        upload_status = gr.Markdown("請上傳 YUNFA_MES.xlsx，分析將自動開始。")

        with gr.Row():
            with gr.Column():
                gr.HTML(
                    """
                    <div class="method-card">
                      <div class="method-title">方法 1｜Rule-Based Scheduling</div>
                      <div class="method-note">
                        問題：哪些工單應該先做？如何派到 CNC？<br>
                        使用：工單、材料齊套、品質風險、交期、CNC 工時。<br>
                        輸出：工單順位、機台排程、FIFO 比較、利用率與排產篩選，共 6 張圖。
                      </div>
                    </div>
                    """
                )
                run_sched = gr.Button("▶ 執行 Rule-Based Scheduling", variant="primary")

            with gr.Column():
                gr.HTML(
                    """
                    <div class="method-card">
                      <div class="method-title">方法 2｜Isolation Forest</div>
                      <div class="method-note">
                        問題：哪些製程事件最不尋常、應優先檢查？<br>
                        使用：加工、換線、停機、報廢、重工與歷史情境偏差。<br>
                        輸出：異常排序、PCA、特徵偏差與規則對照，共 4 張圖。
                      </div>
                    </div>
                    """
                )
                run_if = gr.Button("▶ 執行 Isolation Forest", variant="secondary")

        with gr.Tab("Rule-Based Scheduling 結果"):
            sched_status = gr.Markdown()
            sched_kpis = gr.HTML()
            gr.HTML('<div class="yunfa-section-title">工單排程明細</div><div class="yunfa-unit-note">工時：小時｜排程分數：分｜小數顯示至一位</div>')
            sched_table = gr.HTML(value=df_to_dark_html(pd.DataFrame(), "尚未執行 Scheduling"))
            gr.HTML('<div class="yunfa-section-title">生產排程圖表</div>')
            sched_chart_choice = gr.Dropdown(
                choices=SCHED_CHART_CHOICES, value=0, label="選擇要查看的分析圖表",
                info="一次顯示一張圖及其意義", interactive=True,
            )
            sched_charts = gr.State([])
            sched_chart_image = gr.Image(label="生產排程圖表", type="pil", interactive=False,
                                         height=720, elem_classes=["chart-view"])
            sched_chart_meaning = gr.HTML(value=chart_meaning("請先執行生產排程分析。"))

        with gr.Tab("Isolation Forest 結果"):
            anomaly_status = gr.Markdown()
            anomaly_kpis = gr.HTML()
            gr.HTML('<div class="yunfa-section-title">製程事件明細</div><div class="yunfa-unit-note">相對異常分數：歷史百分位｜其他數值顯示至小數一位</div>')
            anomaly_table = gr.HTML(value=df_to_dark_html(pd.DataFrame(), "尚未執行 Isolation Forest"))
            gr.HTML('<div class="yunfa-section-title">製程異常圖表</div>')
            anomaly_chart_choice = gr.Dropdown(
                choices=ANOMALY_CHART_CHOICES, value=0, label="選擇要查看的分析圖表",
                info="一次顯示一張圖及其意義", interactive=True,
            )
            anomaly_charts = gr.State([])
            anomaly_chart_image = gr.Image(label="製程異常圖表", type="pil", interactive=False,
                                           height=580, elem_classes=["chart-view"])
            anomaly_chart_meaning = gr.HTML(value=chart_meaning("請先執行製程異常分析。"))

        run_sched.click(
            run_scheduling_web,
            inputs=[mes_file],
            outputs=[sched_status, sched_kpis, sched_table, sched_charts,
                     sched_chart_choice, sched_chart_image, sched_chart_meaning],
        )
        run_if.click(
            run_anomaly_web,
            inputs=[mes_file],
            outputs=[anomaly_status, anomaly_kpis, anomaly_table, anomaly_charts,
                     anomaly_chart_choice, anomaly_chart_image, anomaly_chart_meaning],
        )
        sched_chart_choice.change(
            select_chart, inputs=[sched_charts, sched_chart_choice],
            outputs=[sched_chart_image, sched_chart_meaning],
        )
        anomaly_chart_choice.change(
            select_chart, inputs=[anomaly_charts, anomaly_chart_choice],
            outputs=[anomaly_chart_image, anomaly_chart_meaning],
        )
        mes_file.change(
            lambda file: f"✅ 已選擇 MES 檔案：{escape(os.path.basename(normalize_file_path(file)))}。開始分析。" if normalize_file_path(file) else "請上傳 YUNFA_MES.xlsx。",
            inputs=[mes_file], outputs=[upload_status],
        ).then(
            run_scheduling_web, inputs=[mes_file],
            outputs=[sched_status, sched_kpis, sched_table, sched_charts,
                     sched_chart_choice, sched_chart_image, sched_chart_meaning],
        ).then(
            run_anomaly_web, inputs=[mes_file],
            outputs=[anomaly_status, anomaly_kpis, anomaly_table, anomaly_charts,
                     anomaly_chart_choice, anomaly_chart_image, anomaly_chart_meaning],
        )
        return demo

demo = create_mes_app()

if __name__ == "__main__":
    print("🚀 啟動 YUNFA MES Analytics Lab")
    demo.queue().launch(share=True, show_error=True, debug=False)
