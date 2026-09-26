# ===========================================================
#  主題：雲發科 AI Agent 智慧製造戰情室應用實作
#  目標：整合 ERP 與 MES 企業資料，透過 Web-Based 介面串接
#        K-Means、Regression、Isolation Forest 與 Rule-Based
#        Scheduling 等分析方法，建立可查詢、可視覺化並具備
#        管理策略建議功能的 AI Agent 智慧製造戰情室。
#
#  情境：以金屬機電／設備製造業為應用情境，分為業務端與
#        生產端兩類 AI Agent 應用。業務端聚焦客戶分群、
#        客戶經營與製造成本分析；生產端聚焦工單排程、
#        製程異常、材料狀態、品質風險與設備利用率分析。
#
#  應用：使用者可透過 Web 介面上傳 ERP／MES 資料，選擇
#        業務端或生產端問題，由小型語言模型搭配既有 AI
#        分析工具進行查詢，並呈現 KPI、Evidence、分析圖表、
#        圖表說明與管理策略建議。
#
#  教學重點：企業資料 → AI 演算法 → Web 戰情室 →
#            Small Language Model → AI Agent 智慧分析
#
#  作者：國立雲林科技大學 電機工程系 林家仁
# ===========================================================

import os
import re
import io
import gc
import sys
import json
import time
import warnings
import subprocess
import importlib

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Step 0｜Minimal package check
# IMPORTANT:
# - Do NOT upgrade/downgrade the full Colab stack.
# - Install only a package that is actually missing.
# - Transformers is normally preinstalled in Colab; if absent, the data
#   tools still work and the UI will explain that the LLM is unavailable.
# ----------------------------------------------------------------------
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

gradio_mod = _ensure_import("gradio", "gradio==5.47.2")
openpyxl_mod = _ensure_import("openpyxl", "openpyxl")
if gradio_mod is None:
    raise RuntimeError("Gradio 無法載入，請重新建立乾淨 Colab Runtime 後再執行。")
if openpyxl_mod is None:
    raise RuntimeError("openpyxl 無法載入。")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Classroom / projector-friendly chart typography
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.style": "normal",
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})
import torch
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

TRANSFORMERS_AVAILABLE = True
TRANSFORMERS_ERROR = ""
try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_ERROR = str(e)

print(f"✅ Gradio {gr.__version__}")
if TRANSFORMERS_AVAILABLE:
    print(f"✅ Transformers {transformers.__version__}")
else:
    print(f"⚠️ Transformers unavailable: {TRANSFORMERS_ERROR}")
print("✅ YUNFA Agent base environment ready")

# ----------------------------------------------------------------------
# Application state
# ----------------------------------------------------------------------
MODEL_OPTIONS = {
    "Granite 4.0 350M（CPU 可用／最快）": "ibm-granite/granite-4.0-350m",
    "Granite 4.0 1B（CPU 可用但較慢／課堂比較）": "ibm-granite/granite-4.0-1b",
    "Granite 4.0 Micro 3B（建議 T4 GPU）": "ibm-granite/granite-4.0-micro",
}

BENCHMARK_QUESTIONS = [
    "Q1｜找出可立即排產、優先度最高的 5 張工單。",
    "Q2｜比較 Rule-Based 與 FIFO 的急單平均順位、Makespan 與平均 CNC 利用率。",
    "Q3｜列出 Anomaly Score 最高的 5 個製程異常事件與主要偏差因素，建立設備檢查清單。",
    "Q4｜找出交期最近但材料未齊套的 5 張工單，建立採購催料清單。",
    "Q5｜從沉睡／流失風險型客戶中，依年化採購金額列出前 5 名，建立業務追蹤名單。",
    "Q6｜分析汽車精密零組件產品線的成本驅動因素，列出預測誤差最高的 5 張工單，建立成本覆核清單。",
    "Q7｜同時列出最高異常的 3 個製程事件，以及最優先的 3 張可立即排產工單，分成兩類呈現。",
    "Q8｜同時呈現 3 位沉睡／流失風險客戶，以及汽車精密零組件成本誤差最高的 3 張工單。",
    "Q9｜建立廠長風險摘要：列出 3 個最高製程異常事件、3 張最優先排產工單，以及 3 張成本高誤差工單。",
    "Q10｜建立今日 AI 戰情室總覽：排程、製程異常、客戶、成本四個面向各抓最重要的 3 項，並建立 3 項管理 Action。",
]

# ----------------------------------------------------------------------
# AI Agent 展示分類：先區分「業務端」與「生產端」
# 第一版只調整操作分類，不改動既有 Tool / Planner / Agent 主流程。
# Q10 為跨部門戰情室總覽，因此兩端皆保留。
# ----------------------------------------------------------------------
BUSINESS_QUESTIONS = [
    "B1｜列出高價值活躍型客戶前 5 名，並呈現年化採購金額與建議策略。",
    "B2｜列出沉睡／流失風險型客戶前 5 名，建立業務優先追蹤名單。",
    "B3｜列出價格敏感型客戶前 5 名，檢視折扣率與預估毛利率。",
    "B4｜列出成長潛力型客戶前 5 名，觀察採購金額、訂單次數與最近交易天數。",
    "B5｜分析汽車精密零組件產品線的製造成本預測與主要成本驅動因素。",
    "B6｜列出製造成本預測誤差最高的 5 張工單，建立成本覆核名單。",
    "B7｜同時列出高價值活躍型客戶前 3 名，以及成本誤差最高的 3 張工單。",
    "B8｜同時列出沉睡／流失風險型客戶前 3 名，以及成本誤差最高的 3 張工單。",
    "B9｜列出年化採購金額最高的 5 位客戶，並呈現其 K-Means 客戶分群。",
    "B10｜建立業務端 AI 戰情室摘要：列出流失風險客戶前 3 名與成本高誤差工單前 3 名。",
]

PRODUCTION_QUESTIONS = [
    "P1｜找出可立即排產、優先度最高的 5 張工單。",
    "P2｜比較 Rule-Based 與 FIFO 的急單平均順位、Makespan 與平均 CNC 利用率。",
    "P3｜列出 Anomaly Score 最高的 5 個製程異常事件與主要偏差因素。",
    "P4｜找出交期最近但材料未齊套的 5 張工單，建立催料優先清單。",
    "P5｜列出 Rule-Based 排程後 CNC 機台利用率最高與最低的機台。",
    "P6｜列出高品質風險的 5 張未完工工單，作為品質優先確認清單。",
    "P7｜同時列出最高異常的 3 個製程事件，以及最優先的 3 張可立即排產工單。",
    "P8｜同時列出最高異常的 3 個製程事件，以及交期最近的 3 張待料工單。",
    "P9｜列出 FIFO 與 Rule-Based 排程順位差異最大的 5 張工單。",
    "P10｜建立生產端 AI 戰情室摘要：列出最優先排產工單前 3 名與最高異常事件前 3 名。",
]


def questions_for_domain(domain):
    """依 AI Agent 應用端切換展示問題，不改動後端 benchmark routing。"""
    choices = BUSINESS_QUESTIONS if domain == "業務端 AI Agent" else PRODUCTION_QUESTIONS
    return gr.update(choices=choices, value=choices[0])


# Q1-Q10 are classroom benchmark questions.
# The LLM is still asked to produce a plan for teaching/debugging purposes,
# but execution is guarded by these deterministic specifications so a small
# model cannot accidentally call unrelated tools.

# ----------------------------------------------------------------------
# AI Agent 展示問題：依使用情境重新編排，不沿用原本 Q1~Q10 順序
# ----------------------------------------------------------------------
BUSINESS_QUESTIONS = [
    "B1｜列出目前高價值活躍型客戶前 5 名，並說明建議維繫策略。",
    "B2｜找出沉睡／流失風險型客戶前 5 名，建立業務優先追蹤名單。",
    "B3｜比較四類客戶分群的年化採購金額、訂單次數與最近交易天數。",
    "B4｜找出價格敏感型客戶前 5 名，檢視折扣率與預估毛利率。",
    "B5｜分析汽車精密零組件產品線的成本驅動因素。",
    "B6｜列出製造成本預測誤差最高的 5 張工單，建立成本覆核名單。",
    "B7｜找出目前值得優先關注的客戶與成本風險，各列出前 3 項。",
    "B8｜建立業務主管摘要：高價值客戶、流失風險客戶與成本高誤差工單各列 3 項。",
    "B9｜哪些客戶最值得優先進行業務追蹤？請列出前 5 名並說明原因。",
    "B10｜建立業務端 AI 戰情室總覽：客戶分群、流失風險、成本預測與業務策略各列出重點。",
]

PRODUCTION_QUESTIONS = [
    "P1｜找出可立即排產、優先度最高的 5 張工單。",
    "P2｜比較 Rule-Based 與 FIFO 的急單平均順位、Makespan 與平均 CNC 利用率。",
    "P3｜列出 Anomaly Score 最高的 5 個製程異常事件與主要偏差因素。",
    "P4｜找出交期最近但材料未齊套的 5 張工單，建立催料優先清單。",
    "P5｜找出目前 CNC 負載最高與最低的機台，說明排程風險。",
    "P6｜列出品質風險最高、但仍值得優先關注的 5 張未完工工單。",
    "P7｜同時列出最高異常的 3 個製程事件，以及最優先的 3 張可立即排產工單。",
    "P8｜建立生產主管摘要：異常、缺料、排程與設備負載各列出最重要項目。",
    "P9｜若今天只能先處理 3 件事，請從製程異常、缺料與排程中各挑最重要的管理項目。",
    "P10｜建立生產端 AI 戰情室總覽：排程、製程異常、設備負載與材料風險各列出重點。",
]

DISPLAY_QUESTION_SPECS = {
    # Business
    BUSINESS_QUESTIONS[0]: {"tools":["customer"], "top_n":5, "requested_action":"display", "product_line":None},
    BUSINESS_QUESTIONS[1]: {"tools":["customer"], "top_n":5, "requested_action":"sales_follow_up", "product_line":None},
    BUSINESS_QUESTIONS[2]: {"tools":["customer"], "top_n":5, "requested_action":"display", "product_line":None},
    BUSINESS_QUESTIONS[3]: {"tools":["customer"], "top_n":5, "requested_action":"display", "product_line":None},
    BUSINESS_QUESTIONS[4]: {"tools":["cost"], "top_n":5, "requested_action":"display", "product_line":"汽車精密零組件"},
    BUSINESS_QUESTIONS[5]: {"tools":["cost"], "top_n":5, "requested_action":"cost_review", "product_line":None},
    BUSINESS_QUESTIONS[6]: {"tools":["customer","cost"], "top_n":3, "requested_action":"display", "product_line":None},
    BUSINESS_QUESTIONS[7]: {"tools":["customer","cost"], "top_n":3, "requested_action":"display", "product_line":None},
    BUSINESS_QUESTIONS[8]: {"tools":["customer"], "top_n":5, "requested_action":"display", "product_line":None},
    BUSINESS_QUESTIONS[9]: {"tools":["customer","cost"], "top_n":3, "requested_action":"display", "product_line":None},

    # Production
    PRODUCTION_QUESTIONS[0]: {"tools":["scheduling"], "top_n":5, "requested_action":"display", "product_line":None},
    PRODUCTION_QUESTIONS[1]: {"tools":["scheduling"], "top_n":5, "requested_action":"display", "product_line":None},
    PRODUCTION_QUESTIONS[2]: {"tools":["anomaly"], "top_n":5, "requested_action":"machine_inspection", "product_line":None},
    PRODUCTION_QUESTIONS[3]: {"tools":["scheduling"], "top_n":5, "requested_action":"material_expedite", "product_line":None},
    PRODUCTION_QUESTIONS[4]: {"tools":["scheduling"], "top_n":8, "requested_action":"display", "product_line":None},
    PRODUCTION_QUESTIONS[5]: {"tools":["scheduling"], "top_n":5, "requested_action":"display", "product_line":None},
    PRODUCTION_QUESTIONS[6]: {"tools":["anomaly","scheduling"], "top_n":3, "requested_action":"display", "product_line":None},
    PRODUCTION_QUESTIONS[7]: {"tools":["anomaly","scheduling"], "top_n":3, "requested_action":"display", "product_line":None},
    PRODUCTION_QUESTIONS[8]: {"tools":["scheduling"], "top_n":5, "requested_action":"display", "product_line":None},
    PRODUCTION_QUESTIONS[9]: {"tools":["scheduling","anomaly"], "top_n":3, "requested_action":"display", "product_line":None},
}

BENCHMARK_SPECS = {
    1:  {"tools": ["scheduling"],                    "top_n": 5, "requested_action": "display",            "product_line": None},
    2:  {"tools": ["scheduling"],                    "top_n": 5, "requested_action": "display",            "product_line": None},
    3:  {"tools": ["anomaly"],                       "top_n": 5, "requested_action": "machine_inspection", "product_line": None},
    4:  {"tools": ["scheduling"],                    "top_n": 5, "requested_action": "material_expedite",  "product_line": None},
    5:  {"tools": ["customer"],                      "top_n": 5, "requested_action": "sales_follow_up",    "product_line": None},
    6:  {"tools": ["cost"],                          "top_n": 5, "requested_action": "cost_review",        "product_line": "汽車精密零組件"},
    7:  {"tools": ["anomaly", "scheduling"],        "top_n": 3, "requested_action": "display",            "product_line": None},
    8:  {"tools": ["customer", "cost"],             "top_n": 3, "requested_action": "display",            "product_line": "汽車精密零組件"},
    9:  {"tools": ["anomaly", "scheduling", "cost"],"top_n": 3, "requested_action": "display",            "product_line": None},
    10: {"tools": ["scheduling", "anomaly", "customer", "cost"], "top_n": 3, "requested_action": "management_top3", "product_line": None},
}

def benchmark_number(question):
    m = re.match(r"\s*Q(10|[1-9])\s*[｜|:]?", str(question), re.I)
    return int(m.group(1)) if m else None

APP = {
    "erp": None,
    "mes": None,
    "analytics": {},
    "tool_status": {
        "scheduling": "Not Ready",
        "anomaly": "Not Ready",
        "customer": "Not Ready",
        "cost": "Not Ready",
    },
    "tool_error": {},
    "model_name": None,
    "model_label": None,
    "tokenizer": None,
    "model": None,
}

def normalize_file_path(file_value):
    if file_value is None:
        return None
    if isinstance(file_value, (str, os.PathLike)):
        return os.fspath(file_value)
    if isinstance(file_value, dict):
        return file_value.get("path") or file_value.get("name")
    return getattr(file_value, "name", None)

def df_to_dark_html(df, empty_text="尚無資料"):
    """Render table with explicit colors so light/dark system themes stay readable."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return f"<div class='yunfa-table-empty'>{empty_text}</div>"

    safe = df.copy()
    # UI display only: show missing values as an em dash instead of raw NaN.
    # This does not modify the underlying analytics DataFrame.
    safe = safe.astype(object).where(pd.notna(safe), "—")
    html = safe.to_html(
        index=False,
        escape=True,
        border=0,
        classes="yunfa-data-table",
    )
    return "<div class='yunfa-table-wrap'>" + html + "</div>"

def _read_excel(path):
    return pd.ExcelFile(path)


def _safe_num(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _empty_fig(title="尚無圖表"):
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white", height=420)
    return fig

# ------------------------------------------------------------
# Tool 1｜Rule-Based Scheduling（保留原始評分與 FIFO baseline 核心邏輯）
# ------------------------------------------------------------
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
        return pd.DataFrame(rows)
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

# ------------------------------------------------------------
# Tool 2｜K-Means Customer Segmentation
# ------------------------------------------------------------
def build_customer(erp_path):
    xls=pd.ExcelFile(erp_path)
    customer=pd.read_excel(xls,"客戶主檔"); quote=pd.read_excel(xls,"報價紀錄"); order=pd.read_excel(xls,"銷售訂單")
    quote["報價日期"]=pd.to_datetime(quote["報價日期"],errors="coerce"); order["下單日期"]=pd.to_datetime(order["下單日期"],errors="coerce")
    quote=_safe_num(quote,["折扣率","預估毛利率"]); order=_safe_num(order,["訂購數量","成交單價_TWD"])
    order["訂單金額_TWD"]=order["訂購數量"]*order["成交單價_TWD"]
    ref=max(quote["報價日期"].max(),order["下單日期"].max())
    q=quote.groupby("客戶ID").agg(報價次數=("報價ID","count"),成交報價次數=("結果",lambda x:(x=="成交").sum()),平均折扣率=("折扣率","mean"),平均預估毛利率=("預估毛利率","mean"))
    q["報價成交率"]=q["成交報價次數"]/q["報價次數"].replace(0,np.nan)
    o=order.groupby("客戶ID").agg(總採購金額_TWD=("訂單金額_TWD","sum"),訂單次數=("訂單ID","count"),平均訂單金額_TWD=("訂單金額_TWD","mean"),最近交易日=("下單日期","max"))
    o["最近交易天數"]=(ref-o["最近交易日"]).dt.days; o["年化採購金額_TWD"]=o["總採購金額_TWD"]/3
    c=customer.rename(columns={"Customer_ID":"客戶ID"}).merge(q.reset_index(),on="客戶ID",how="left").merge(o.reset_index(),on="客戶ID",how="left")
    if "客戶等級" in c.columns: c=c.rename(columns={"客戶等級":"既有客戶等級（ERP）"})
    feats=["年化採購金額_TWD","訂單次數","平均訂單金額_TWD","報價次數","報價成交率","平均折扣率","平均預估毛利率","最近交易天數"]
    c[feats]=c[feats].apply(pd.to_numeric,errors="coerce").fillna(0)
    scaler=StandardScaler(); X=scaler.fit_transform(c[feats]); km=KMeans(n_clusters=4,random_state=42,n_init=20); c["Cluster"]=km.fit_predict(X)
    centers=pd.DataFrame(km.cluster_centers_,columns=feats,index=range(4)); rem=set(range(4)); names={}
    risk=(centers["最近交易天數"]-.5*centers["年化採購金額_TWD"]-.5*centers["訂單次數"]).idxmax(); names[risk]="沉睡／流失風險型"; rem.remove(risk)
    vip=(centers["年化採購金額_TWD"]+centers["訂單次數"]+centers["報價次數"]-centers["最近交易天數"]).loc[list(rem)].idxmax(); names[vip]="高價值活躍型"; rem.remove(vip)
    price=(centers["平均折扣率"]-centers["平均預估毛利率"]).loc[list(rem)].idxmax(); names[price]="價格敏感型"; rem.remove(price)
    names[list(rem)[0]]="成長潛力型"; c["K-Means行為分群"]=c["Cluster"].map(names)
    strategy={"高價值活躍型":"維持高互動、優先服務、交叉銷售與長期合作","成長潛力型":"提高成交率、增加產品組合與業務接觸","價格敏感型":"管理折扣、檢查毛利、採差異化報價策略","沉睡／流失風險型":"優先喚回、追蹤未成交原因與近期需求"}
    c["建議策略"]=c["K-Means行為分群"].map(strategy)
    pca=PCA(2); Z=pca.fit_transform(X); c["PC1"]=Z[:,0]; c["PC2"]=Z[:,1]
    try: sil=float(silhouette_score(X,c["Cluster"]))
    except: sil=np.nan
    return {"data":c,"features":feats,"kpis":{"客戶數":len(c),"分群數":4,"Silhouette Score":None if pd.isna(sil) else round(sil,3)}}

# ------------------------------------------------------------
# Tool 3｜Multiple Linear Regression
# ------------------------------------------------------------
def build_cost(erp_path):
    xls=pd.ExcelFile(erp_path); product=pd.read_excel(xls,"產品主檔"); cost=pd.read_excel(xls,"成本結算")
    pk=["SKU","產品線","標準材料成本_TWD","標準人工成本_TWD","標準委外成本_TWD","標準製造費_TWD"]
    ck=["工單ID","SKU","完工數量","CNC加工工時_hr","組裝工時_hr","測試工時_hr","換線工時_hr","實際製造成本_TWD"]
    d=cost[ck].merge(product[pk],on="SKU",how="inner")
    for col in ck[2:]+pk[2:]:
        if col in d.columns: d[col]=pd.to_numeric(d[col],errors="coerce")
    d["標準材料總成本"]=d["完工數量"]*d["標準材料成本_TWD"]; d["標準人工總成本"]=d["完工數量"]*d["標準人工成本_TWD"]
    d["標準委外總成本"]=d["完工數量"]*d["標準委外成本_TWD"]; d["標準製造費總額"]=d["完工數量"]*d["標準製造費_TWD"]
    feats=["完工數量","標準材料總成本","標準人工總成本","標準委外總成本","標準製造費總額","CNC加工工時_hr","組裝工時_hr","測試工時_hr","換線工時_hr"]
    return {"data":d,"features":feats,"target":"實際製造成本_TWD","product_lines":sorted(d["產品線"].dropna().unique().tolist())}

def run_cost_model(cost_obj, product_line=None):
    d=cost_obj["data"]; feats=cost_obj["features"]; target=cost_obj["target"]
    if product_line not in set(d["產品線"].dropna()): product_line=cost_obj["product_lines"][0]
    s=d[d["產品線"]==product_line].dropna(subset=feats+[target]).copy()
    if len(s)<8: raise ValueError(f"{product_line} 建模資料不足。")
    X=s[feats]; y=s[target]; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42)
    scaler=StandardScaler(); Xtrz=scaler.fit_transform(Xtr); Xtez=scaler.transform(Xte); model=LinearRegression().fit(Xtrz,ytr); pred=model.predict(Xtez)
    r2=r2_score(yte,pred); mae=mean_absolute_error(yte,pred); rmse=np.sqrt(mean_squared_error(yte,pred)); mape=float(np.mean(np.abs((yte.values-pred)/np.maximum(np.abs(yte.values),1)))*100)
    res=s.loc[Xte.index,["工單ID","SKU","產品線"]].copy(); res["實際製造成本"]=yte.values; res["預測製造成本"]=pred; res["誤差率(%)"]=np.abs(res["實際製造成本"]-res["預測製造成本"])/np.maximum(res["實際製造成本"],1)*100
    res["是否超出±3%"] = np.where(res["誤差率(%)"]>3,"是","否")
    coef=pd.DataFrame({"變數":feats,"標準化迴歸係數":model.coef_}); coef["影響程度"]=coef["標準化迴歸係數"].abs(); coef=coef.sort_values("影響程度",ascending=False)
    return {"product_line":product_line,"result":res.sort_values("誤差率(%)",ascending=False),"coef":coef,"kpis":{"R²":round(float(r2),3),"MAE":round(float(mae),0),"RMSE":round(float(rmse),0),"MAPE(%)":round(mape,2),"±3%內比例(%)":round(float((res["誤差率(%)"]<=3).mean()*100),1)}}

# ------------------------------------------------------------
# Tool 4｜Isolation Forest
# ------------------------------------------------------------
def _safe_div(a,b):
    a=pd.to_numeric(a,errors="coerce"); b=pd.to_numeric(b,errors="coerce"); return a/b.replace(0,np.nan)

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



# ----------------------------------------------------------------------
# Dataset initialization
# ----------------------------------------------------------------------
def reset_tool_state():
    APP["analytics"] = {}
    for k in APP["tool_status"]:
        APP["tool_status"][k] = "Not Ready"
    APP["tool_error"] = {}

def initialize_yunfa_tools(erp_file, mes_file):
    erp_path = normalize_file_path(erp_file)
    mes_path = normalize_file_path(mes_file)
    reset_tool_state()

    if not erp_path or not mes_path:
        return "❌ 請同時上傳 YUNFA_ERP.xlsx 與 YUNFA_MES.xlsx。", df_to_dark_html(pd.DataFrame(), "尚未建立 Tool Status")

    APP["erp"], APP["mes"] = erp_path, mes_path

    jobs = [
        ("scheduling", lambda: build_scheduling(mes_path)),
        ("anomaly", lambda: build_anomaly(mes_path)),
        ("customer", lambda: build_customer(erp_path)),
        ("cost", lambda: build_cost(erp_path)),
    ]

    for name, fn in jobs:
        try:
            APP["analytics"][name] = fn()
            APP["tool_status"][name] = "Ready"
            APP["tool_error"][name] = "Ready"
        except Exception as e:
            APP["analytics"][name] = None
            APP["tool_status"][name] = "Not Ready"
            APP["tool_error"][name] = f"{type(e).__name__}: {e}"

    rows = [
        {
            "Tool": k,
            "Status": APP["tool_status"][k],
            "Reason": APP["tool_error"].get(k, ""),
        }
        for k in ["scheduling", "anomaly", "customer", "cost"]
    ]
    table = pd.DataFrame(rows)

    ready = [k for k, v in APP["tool_status"].items() if v == "Ready"]
    if len(ready) == 4:
        s = APP["analytics"]["scheduling"]["kpis"]
        a = APP["analytics"]["anomaly"]["kpis"]
        c = APP["analytics"]["customer"]["kpis"]
        msg = (
            "✅ YUNFA ERP / MES 初始化完成：4/4 Tools Ready\n\n"
            f"未完工工單：{s['未完工工單']}｜"
            f"可立即排產：{s['可立即排產']}｜"
            f"High Anomaly：{a['High Anomaly']}｜"
            f"客戶數：{c['客戶數']}\n\n"
            "下一步：選擇 Granite 模型後按「載入 / 切換模型」。"
        )
    elif ready:
        msg = (
            f"⚠️ 資料已讀取，但只有 {len(ready)}/4 Tools Ready。\n"
            "請直接查看 Tool Status Table 的 Reason；不需要猜是哪一步失敗。"
        )
    else:
        msg = "❌ ERP / MES 可讀取，但 4 個 Tool 全部建立失敗。請查看 Reason。"

    return msg, df_to_dark_html(table)

# ----------------------------------------------------------------------
# Granite model loading
# ----------------------------------------------------------------------
def unload_model():
    APP["model"] = None
    APP["tokenizer"] = None
    APP["model_name"] = None
    APP["model_label"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _model_dtype(model_id):
    if not torch.cuda.is_available():
        return torch.float32
    # This 1B checkpoint produced invalid repetitive punctuation in FP16
    # during the classroom benchmark, so use FP32 on T4 for this model.
    if model_id.endswith("granite-4.0-1b"):
        return torch.float32
    return torch.float16

def load_granite_model(model_label):
    if not TRANSFORMERS_AVAILABLE:
        yield (
            "❌ Transformers 無法使用。\n\n"
            f"{TRANSFORMERS_ERROR}\n\n"
            "ERP / MES 四個 Tool 仍可初始化，但語言模型無法載入。"
        )
        return

    model_id = MODEL_OPTIONS[model_label]
    if APP["model_name"] == model_id and APP["model"] is not None:
        yield f"✅ {model_label} 已在記憶體中，不需要重新載入。"
        return

    t0 = time.time()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    cpu_note = ""
    if not torch.cuda.is_available():
        if model_id.endswith("granite-4.0-micro"):
            cpu_note = (
                "\n⚠️ 目前是 CPU。3B 可以嘗試載入，但速度會很慢，"
                "而且 Colab 一般 RAM 可能不足；課堂建議改用 350M 或 1B。"
            )
        elif model_id.endswith("granite-4.0-1b"):
            cpu_note = (
                "\nℹ️ 目前是 CPU。1B 可以執行，但模型載入與 Planner 推論會明顯較久。"
            )
        else:
            cpu_note = (
                "\nℹ️ 目前是 CPU。350M 可以執行，速度會比 T4 慢；"
                "這是真實模型推論時間，不是人工延遲。"
            )

    yield f"⏳ 1/4 準備模型\n目標：{model_label}\n裝置：{device_name}{cpu_note}"
    unload_model()

    yield "⏳ 2/4 載入 Tokenizer\n第一次使用時可能需要下載。"
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        yield f"❌ Tokenizer 載入失敗\n{type(e).__name__}: {e}"
        return

    dtype = _model_dtype(model_id)
    dtype_name = str(dtype).replace("torch.", "")
    yield f"⏳ 3/4 載入模型權重\n精度：{dtype_name}\n第一次下載是最久的一步。"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
    except Exception as e:
        yield f"❌ 模型權重載入失敗\n{type(e).__name__}: {e}"
        return

    APP.update({
        "model_name": model_id,
        "model_label": model_label,
        "tokenizer": tok,
        "model": model,
    })

    yield "⏳ 4/4 Warm-up\n執行一次短推論。"
    try:
        messages = [{"role": "user", "content": "請只輸出 OK"}]
        try:
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            x = tok(prompt, return_tensors="pt").to(device)
        except Exception:
            x = tok("OK", return_tensors="pt").to(device)
        with torch.inference_mode():
            model.generate(
                **x,
                max_new_tokens=2,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
    except Exception:
        pass

    yield (
        f"✅ 模型 Ready：{model_label}\n"
        f"裝置：{device_name}\n"
        f"精度：{dtype_name}\n"
        f"載入時間：{time.time()-t0:.1f}s"
    )

# ----------------------------------------------------------------------
# Planner
# ----------------------------------------------------------------------
PLANNER_SYSTEM_PROMPT = """
你是製造業 AI Agent Planner。
只輸出一個 JSON object，不要 Markdown，不要解釋。

可用工具：
scheduling = 工單、排程、缺料、交期、FIFO、CNC
anomaly = 製程、設備、異常事件
customer = 客戶、分群、流失、業務
cost = 成本、迴歸、預測誤差

固定格式：
{"intent":"短名稱","tools":["scheduling"],"top_n":5,"product_line":null,"requested_action":"display"}

requested_action 只能使用：
display
material_expedite
sales_follow_up
machine_inspection
cost_review
schedule_review
management_top3

若同一問題需要多種資料，tools 必須列出所有需要的工具。
"""

ALLOWED_TOOLS = {"scheduling", "anomaly", "customer", "cost"}
ALLOWED_ACTIONS = {
    "display",
    "material_expedite",
    "sales_follow_up",
    "machine_inspection",
    "cost_review",
    "schedule_review",
    "management_top3",
}

def llm_generate(messages, max_new_tokens=96):
    tok = APP["tokenizer"]
    model = APP["model"]
    device = next(model.parameters()).device

    try:
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt").to(device)
    except Exception:
        joined = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        inputs = tok(joined, return_tensors="pt").to(device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

def parse_first_json(text):
    if not text:
        return None
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def infer_action(question):
    q = str(question)
    # Actions require explicit action intent, not merely mentioning the domain.
    if "廠長風險摘要" in q:
        return "display"
    if any(k in q for k in ["三項管理 Action", "管理 Action", "三項管理", "只能處理"]):
        return "management_top3"
    if any(k in q for k in ["建立採購催料", "建立催料", "催料清單", "催料任務"]):
        return "material_expedite"
    if any(k in q for k in ["建立業務追蹤", "業務追蹤名單", "業務追蹤任務", "建立喚回"]):
        return "sales_follow_up"
    if any(k in q for k in ["建立設備檢查", "設備檢查清單", "建立製程檢查", "優先調查清單"]):
        return "machine_inspection"
    if any(k in q for k in ["建立成本覆核", "成本覆核清單", "成本覆核任務"]):
        return "cost_review"
    if any(k in q for k in ["重新排程", "建立排程覆核", "排程覆核清單"]):
        return "schedule_review"
    return "display"

def infer_tools(question, default_to_scheduling=True):
    q = str(question)
    tools = []
    if any(k in q for k in ["排產", "排程", "缺料", "材料未齊", "交期", "FIFO", "急單"]):
        tools.append("scheduling")
    if any(k in q for k in ["Anomaly", "異常", "製程事件", "設備檢查"]):
        tools.append("anomaly")
    if any(k in q for k in ["客戶", "流失", "沉睡", "業務追蹤"]):
        tools.append("customer")
    if any(k in q for k in ["成本", "迴歸", "預測誤差"]):
        tools.append("cost")
    if not tools and default_to_scheduling:
        tools = ["scheduling"]
    return list(dict.fromkeys(tools))

def parse_top_n_from_question(question, default=5):
    q = str(question)
    patterns = [
        r"(?:前|Top)\s*(\d+)",
        r"(?:最高的?|最優先的?|最重要的?)\s*(\d+)",
        r"(?:列出|呈現|各抓)\s*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, q, re.I)
        if m:
            return max(1, min(10, int(m.group(1))))
    m = re.search(r"(?<!\d)([1-9]|10)\s*(?:個|位|張|項)", q)
    if m:
        return max(1, min(10, int(m.group(1))))
    return default

def normalize_plan(question, raw_plan):
    p = dict(raw_plan) if isinstance(raw_plan, dict) else {}

    # 0) 業務端 / 生產端展示問題：使用明確 guardrail，確保課堂展示穩定。
    if str(question) in DISPLAY_QUESTION_SPECS:
        spec = DISPLAY_QUESTION_SPECS[str(question)].copy()
        p["intent"] = str(p.get("intent") or "guided_demo")
        p.update(spec)
        p["routing_source"] = "guided_demo_guardrail"
        return p

    # 1) Q1-Q10: exact deterministic execution contract.
    qno = benchmark_number(question)
    if qno in BENCHMARK_SPECS:
        spec = BENCHMARK_SPECS[qno].copy()
        p["intent"] = str(p.get("intent") or f"benchmark_q{qno}")
        p.update(spec)
        p["routing_source"] = f"benchmark_guardrail_Q{qno}"
        return p

    # 2) Custom questions: deterministic keyword routing has priority.
    #    Only if no known domain is detected do we accept valid LLM tools.
    deterministic_tools = infer_tools(question, default_to_scheduling=False)
    raw_tools = p.get("tools")
    llm_tools = [t for t in raw_tools if t in ALLOWED_TOOLS] if isinstance(raw_tools, list) else []

    if deterministic_tools:
        tools = deterministic_tools
        routing_source = "deterministic_keywords"
    elif llm_tools:
        tools = llm_tools
        routing_source = "llm_fallback"
    else:
        tools = ["scheduling"]
        routing_source = "default_scheduling"

    p["tools"] = list(dict.fromkeys(tools))
    p["intent"] = str(p.get("intent") or "manufacturing_analysis")
    p["top_n"] = parse_top_n_from_question(question, default=int(p.get("top_n") or 5))

    product_line = p.get("product_line")
    if "汽車精密零組件" in str(question):
        product_line = "汽車精密零組件"
    p["product_line"] = product_line

    p["requested_action"] = infer_action(question)
    p["routing_source"] = routing_source
    return p

def plan_question(question):
    if APP["model"] is None:
        # Allows complete Tool/UI testing without an LLM.
        raw = "MODEL_NOT_LOADED → deterministic fallback"
        return normalize_plan(question, None), raw

    raw = llm_generate(
        [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": str(question)},
        ],
        max_new_tokens=90,
    )
    return normalize_plan(question, parse_first_json(raw)), raw


def _fig_to_gallery(fig, caption):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=135, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img = Image.open(buf).convert("RGB").copy()
    plt.close(fig)
    return (img, caption)

def _error_chart(message):
    fig, ax = plt.subplots(figsize=(8,3.5), facecolor="white")
    ax.axis("off")
    ax.text(0.5,0.5,str(message),ha="center",va="center",wrap=True,fontsize=12)
    return _fig_to_gallery(fig, "Chart Error")

def scheduling_original_charts(x):
    charts = []
    priority_df = x["priority"]
    schedule = x["schedule"]
    fifo = x["fifo"]
    util = x["util"]
    k = x["kpis"]
    machines = util["CNC機台"].astype(str).tolist()

    # CHART 1｜Horizontal bars improve readability for 15 work-order IDs
    top15 = priority_df.head(15).copy()
    fig, ax = plt.subplots(figsize=(10,7.2), facecolor="white")
    bars = ax.barh(
        top15["工單ID"].astype(str),
        top15["排程分數"],
        color="#91B8A4",
        height=0.62,
    )
    ax.invert_yaxis()
    for bar in bars:
        bar_width = bar.get_width()
        ax.text(
            bar_width + 1,
            bar.get_y() + bar.get_height()/2,
            f"{int(bar_width)}",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_title("Top 15 Work Orders by Priority Score", fontweight="bold", pad=14)
    ax.set_xlabel("Priority Score")
    ax.set_ylabel("Work Order")
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="y", labelsize=11)
    fig.tight_layout()
    charts.append(_fig_to_gallery(fig, "Scheduling 1｜Top 15 Work Orders by Priority Score"))

    # Original CHART 2 - Gantt
    fig, ax = plt.subplots(figsize=(12,6), facecolor="white")
    colors = ["#82B89A","#7FAF9B","#B7A7CC","#D8B58B","#8FBFB1","#D8A0A0","#78909C","#D8C989"]
    for idx, row in schedule.iterrows():
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

    # Original CHART 3
    fig, ax = plt.subplots(figsize=(10,5), facecolor="white")
    bars = ax.bar(util["CNC機台"], util["Rule-Based利用率(%)"], color="#82B89A", width=0.55)
    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, y+1, f"{y:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(float(util["Rule-Based利用率(%)"].max())+15,100))
    ax.set_title("CNC Utilization after Rule-Based Scheduling", fontweight="bold")
    ax.set_xlabel("CNC Machine"); ax.set_ylabel("Utilization (%)")
    fig.tight_layout()
    charts.append(_fig_to_gallery(fig, "Scheduling 3｜CNC Utilization after Rule-Based Scheduling"))

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
    b2=ax.bar(xx+width/2,[urgent_rb,vip_rb],width,label="Rule-Based",color="#7FAF9B")
    for bar in list(b1)+list(b2):
        y=bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2,y+.2,f"{y:.1f}",ha="center",va="bottom")
    ax.set_xticks(xx); ax.set_xticklabels(["Urgent Avg Rank","VIP Avg Rank"])
    ax.set_ylabel("Average Dispatch Rank")
    ax.set_title("FIFO vs Rule-Based Dispatch Priority", fontweight="bold")
    ax.legend(frameon=False); fig.tight_layout()
    charts.append(_fig_to_gallery(fig, "Scheduling 4｜FIFO vs Rule-Based Dispatch Priority"))

    # Original CHART 5
    fig, ax = plt.subplots(figsize=(8,5), facecolor="white")
    xx=np.arange(2); width=.35
    fifo_eff=[float(k["FIFO Makespan(hr)"]),float(k["FIFO平均利用率(%)"])]
    rb_eff=[float(k["Rule-Based Makespan(hr)"]),float(k["Rule-Based平均利用率(%)"])]
    b1=ax.bar(xx-width/2,fifo_eff,width,label="FIFO",color="#95a5a6")
    b2=ax.bar(xx+width/2,rb_eff,width,label="Rule-Based",color="#82B89A")
    for bar in list(b1)+list(b2):
        y=bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2,y+1,f"{y:.1f}",ha="center",va="bottom")
    ax.set_xticks(xx); ax.set_xticklabels(["Makespan (hr)","Avg CNC Utilization (%)"])
    ax.set_ylabel("Metric Value"); ax.set_title("FIFO vs Rule-Based Scheduling Efficiency",fontweight="bold")
    ax.legend(frameon=False); fig.tight_layout()
    charts.append(_fig_to_gallery(fig, "Scheduling 5｜FIFO vs Rule-Based Scheduling Efficiency"))

    # Original CHART 6
    fig, ax = plt.subplots(figsize=(9,5), facecolor="white")
    categories=["Unfinished","Ready","Material Hold","Quality Risk"]
    counts=[len(priority_df), int(k["可立即排產"]), len(x["material_hold"]), len(x["quality_hold"])]
    cols=["#7FAF9B","#82B89A","#D8B58B","#D8A0A0"]
    bars=ax.bar(categories,counts,color=cols,width=.55)
    for bar in bars:
        y=bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2,y+.5,f"{int(y)}",ha="center",va="bottom")
    ax.set_ylabel("Number of Work Orders")
    ax.set_title("MES Work Order Screening for Production Scheduling",fontweight="bold")
    fig.tight_layout()
    charts.append(_fig_to_gallery(fig, "Scheduling 6｜MES Work Order Screening"))
    return charts

def anomaly_original_charts(x):
    charts=[]
    d=x["data"]
    top=d.head(15).sort_values("Anomaly Score",ascending=True).copy()
    top["EventLabel"]="WO "+top["工單ID"].astype(str)+" | OP "+top["工序序號"].astype(str)+" | MC "+top["機台ID"].astype(str)

    # Original CHART 1
    fig,ax=plt.subplots(figsize=(12,7),facecolor="white")
    colors=["#C98282" if z=="High Anomaly" else "#E8D58A" for z in top["異常層級"]]
    bars=ax.barh(top["EventLabel"],top["Anomaly Score"],color=colors,edgecolor="white")
    for bar,val in zip(bars,top["Anomaly Score"]):
        ax.text(val+.5,bar.get_y()+bar.get_height()/2,f"{val:.1f}",va="center",fontsize=9)
    ax.set_xlabel("Anomaly Score (Historical Percentile)")
    ax.set_ylabel("Work Order | Operation | Machine")
    ax.set_title(f"Top 15 MES Anomaly Events - {x['detect_year']}",fontweight="bold")
    ax.set_xlim(0,105); ax.grid(axis="x",alpha=.35); fig.tight_layout()
    charts.append(_fig_to_gallery(fig,"Anomaly 1｜Top 15 MES Anomaly Events"))

    # Original CHART 2
    fig,ax=plt.subplots(figsize=(11,7),facecolor="white")
    tr=x["train_pca"]
    ax.scatter(tr["PC1"],tr["PC2"],s=28,alpha=.22,color="#D9D9D9",label="Historical 2023-2025")
    normal=d["IF判定"].eq("正常"); abn=d["IF判定"].eq("異常")
    ax.scatter(d.loc[normal,"PC1"],d.loc[normal,"PC2"],s=50,alpha=.72,color="#A8C69F",
               edgecolors="white",linewidths=.6,label="2026 Normal")
    ax.scatter(d.loc[abn,"PC1"],d.loc[abn,"PC2"],s=95,alpha=.95,color="#C98282",
               edgecolors="white",linewidths=.8,label="2026 Anomaly")
    ax.set_xlabel("Principal Component 1"); ax.set_ylabel("Principal Component 2")
    ax.set_title("MES Multi-Variable Anomaly Map - PCA",fontweight="bold")
    ax.legend(frameon=False); ax.grid(True,alpha=.35); fig.tight_layout()
    charts.append(_fig_to_gallery(fig,"Anomaly 2｜PCA 2D MES Anomaly Map"))

    # Original CHART 3
    prof=x["profile_df"].sort_values("相對歷史標準化偏差",ascending=True)
    fig,ax=plt.subplots(figsize=(12,6.5),facecolor="white")
    cols=["#C98282" if abs(v)>=2 else "#E8D5BA" for v in prof["相對歷史標準化偏差"]]
    bars=ax.barh(prof["英文名稱"],prof["相對歷史標準化偏差"],color=cols,edgecolor="white")
    ax.axvline(0,color="#888888",linestyle="--",linewidth=1)
    for bar,val in zip(bars,prof["相對歷史標準化偏差"]):
        ax.text(val+(.08 if val>=0 else -.08),bar.get_y()+bar.get_height()/2,f"{val:.2f}",
                va="center",ha="left" if val>=0 else "right",fontsize=9)
    ax.set_xlabel("Standardized Deviation from Historical Baseline")
    ax.set_ylabel("MES Feature"); ax.set_title("Top Anomaly - Multi-Factor Deviation Profile",fontweight="bold")
    ax.grid(axis="x",alpha=.35); fig.tight_layout()
    charts.append(_fig_to_gallery(fig,"Anomaly 3｜Top Anomaly Multi-Factor Deviation Profile"))

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
    return charts

def customer_original_charts(x):
    charts=[]
    d=x["data"]
    feats=x["features"]
    cluster_order=["高價值活躍型","成長潛力型","價格敏感型","沉睡／流失風險型"]
    name_en={
        "高價值活躍型":"High-Value Active",
        "成長潛力型":"Growth Potential",
        "價格敏感型":"Price Sensitive",
        "沉睡／流失風險型":"Dormant / Churn Risk",
    }
    colors={
        "高價值活躍型":"#E8D5BA",
        "成長潛力型":"#A8C69F",
        "價格敏感型":"#B9CED9",
        "沉睡／流失風險型":"#CEC3D9",
    }

    # Original PCA
    fig,ax=plt.subplots(figsize=(11,7),facecolor="white")
    for group in cluster_order:
        part=d[d["K-Means行為分群"]==group]
        ax.scatter(part["PC1"],part["PC2"],s=85,alpha=.82,label=name_en[group],
                   color=colors[group],edgecolors="white",linewidths=.8)
    ax.set_xlabel("Principal Component 1"); ax.set_ylabel("Principal Component 2")
    ax.set_title("ERP Customer Segmentation - K-Means",fontsize=15,fontweight="bold")
    ax.legend(frameon=False); ax.grid(True,alpha=.35); fig.tight_layout()
    charts.append(_fig_to_gallery(fig,"Customer 1｜ERP Customer Segmentation - K-Means"))

    # Original cluster size
    size=d["K-Means行為分群"].value_counts().reindex(cluster_order).fillna(0)
    fig,ax=plt.subplots(figsize=(10,6),facecolor="white")
    bars=ax.bar([name_en[z] for z in size.index],size.values,
                color=[colors[z] for z in size.index],edgecolor="white",linewidth=1)
    for bar,val in zip(bars,size.values):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.25,f"{int(val)}",ha="center",va="bottom",fontsize=10)
    ax.set_xlabel("Customer Segment"); ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Segment Distribution",fontsize=14,fontweight="bold")
    ax.grid(axis="y",alpha=.35); fig.tight_layout()
    charts.append(_fig_to_gallery(fig,"Customer 2｜Customer Segment Distribution"))

    # Original normalized profile
    scaler=StandardScaler()
    z=scaler.fit_transform(d[feats])
    zdf=pd.DataFrame(z,columns=feats,index=d.index)
    zdf["K-Means行為分群"]=d["K-Means行為分群"].values
    profile=zdf.groupby("K-Means行為分群")[feats].mean().reindex(cluster_order)
    profile_en={
        "年化採購金額_TWD":"Annual Purchase","訂單次數":"Order Frequency",
        "平均訂單金額_TWD":"Avg. Order","報價次數":"Quote Count",
        "報價成交率":"Conversion","平均折扣率":"Discount",
        "平均預估毛利率":"Gross Margin","最近交易天數":"Recency",
    }
    fig,ax=plt.subplots(figsize=(14,6.5),facecolor="white")
    xx=np.arange(len(feats)); width=.18
    for i,group in enumerate(cluster_order):
        ax.bar(xx+(i-1.5)*width,profile.loc[group].values,width=width,label=name_en[group],
               color=colors[group],edgecolor="white")
    ax.axhline(0,color="#888888",linewidth=1,linestyle="--")
    ax.set_xticks(xx); ax.set_xticklabels([profile_en[f] for f in feats],fontsize=9)
    ax.set_ylabel("Standardized Cluster Mean"); ax.set_title("Customer Segment Profile",fontsize=14,fontweight="bold")
    ax.legend(frameon=False,ncol=2); ax.grid(axis="y",alpha=.35); fig.tight_layout()
    charts.append(_fig_to_gallery(fig,"Customer 3｜Customer Segment Profile"))
    return charts

def cost_original_charts(r):
    charts=[]
    result=r["result"].copy()
    coef=r["coef"].copy()
    product_line=r["product_line"]
    product_line_en={
        "汽車精密零組件":"Automotive Precision Parts",
        "精密金屬加工件":"Precision Metal Components",
        "機電／自動化模組":"Mechatronics & Automation Modules",
        "半導體設備零組件":"Semiconductor Equipment Parts",
    }.get(product_line,str(product_line))
    feature_en={
        "完工數量":"Completed Quantity",
        "標準材料總成本":"Standard Material Cost",
        "標準人工總成本":"Standard Labor Cost",
        "標準委外總成本":"Standard Outsourcing Cost",
        "標準製造費總額":"Standard Manufacturing Overhead",
        "CNC加工工時_hr":"CNC Machining Hours",
        "組裝工時_hr":"Assembly Hours",
        "測試工時_hr":"Test Hours",
        "換線工時_hr":"Setup Hours",
    }

    # Original CHART 1 Actual vs Predicted with ±3%
    actual=result["實際製造成本"]/1000
    pred=result["預測製造成本"]/1000
    within=result["誤差率(%)"]<=3
    fig,ax=plt.subplots(figsize=(10,6),facecolor="white")
    ax.scatter(actual[within],pred[within],alpha=.82,s=62,color="#F6B26B",edgecolors="white",linewidths=.7,label="Within ±3%")
    ax.scatter(actual[~within],pred[~within],alpha=.95,s=78,color="#C98282",edgecolors="white",linewidths=.9,label="Outside ±3%")
    min_v=min(actual.min(),pred.min()); max_v=max(actual.max(),pred.max())
    ax.plot([min_v,max_v],[min_v,max_v],linestyle="--",linewidth=2,color="#6FA8DC",label="Perfect Prediction")
    xb=np.linspace(min_v,max_v,200); lo=xb*.97; hi=xb*1.03
    ax.fill_between(xb,lo,hi,alpha=.18,color="#93C47D",label="±3% Prediction Range")
    ax.plot(xb,lo,linestyle=":",linewidth=1.2,color="#93C47D")
    ax.plot(xb,hi,linestyle=":",linewidth=1.2,color="#93C47D")
    ax.set_xlabel("Actual Manufacturing Cost (K NTD)")
    ax.set_ylabel("Predicted Manufacturing Cost (K NTD)")
    ax.set_title(f"Actual vs Predicted Cost with ±3% Range - {product_line_en}",fontweight="bold")
    ax.grid(True,alpha=.35); ax.legend(frameon=False); fig.tight_layout()
    charts.append(_fig_to_gallery(fig,"Cost 1｜Actual vs Predicted Cost with ±3% Range"))

    # Original CHART 2 Cost Drivers
    pc=coef.sort_values("影響程度",ascending=True).copy()
    pc["English"]=pc["變數"].map(feature_en).fillna(pc["變數"])
    pc["CoefK"]=pc["標準化迴歸係數"]/1000
    palette=["#A8C69F","#E8D5BA","#B9CED9","#CEC3D9","#E8D58A","#C9D8B6","#E6B8AF","#B7D7D0","#F4D7B9"]
    fig,ax=plt.subplots(figsize=(12,6.5),facecolor="white")
    bars=ax.barh(pc["English"],pc["CoefK"],color=[palette[i%len(palette)] for i in range(len(pc))],
                 edgecolor="#F7F2EC",linewidth=1.2,height=.62)
    coefmax=max(pc["CoefK"].abs().max(),1); off=.02*coefmax
    for bar,val in zip(bars,pc["CoefK"]):
        ax.text(val+(off if val>=0 else -off),bar.get_y()+bar.get_height()/2,f"{val:,.0f}",
                va="center",ha="left" if val>=0 else "right",fontsize=9,color="#3F3F3F")
    ax.axvline(0,linestyle="--",linewidth=1.5,color="#888888")
    xmin=min(pc["CoefK"].min(),0); xmax=max(pc["CoefK"].max(),0); pad=.12*max(abs(xmin),abs(xmax),1)
    ax.set_xlim(xmin-pad,xmax+pad)
    ax.set_xlabel("Standardized Coefficient (K NTD)"); ax.set_ylabel("Feature")
    ax.set_title(f"Manufacturing Cost Drivers - {product_line_en}",fontweight="bold")
    ax.grid(axis="x",alpha=.35); fig.tight_layout()
    charts.append(_fig_to_gallery(fig,"Cost 2｜Manufacturing Cost Drivers"))

    # Original CHART 3 Error Distribution
    fig,ax=plt.subplots(figsize=(10,6),facecolor="white")
    ax.hist(result["誤差率(%)"],bins=15,color="#B7D7C4",edgecolor="white",linewidth=1)
    ax.set_xlabel("Absolute Error Rate (%)"); ax.set_ylabel("Number of Work Orders")
    ax.set_title(f"Prediction Error Distribution - {product_line_en}",fontweight="bold")
    ax.grid(axis="y",alpha=.35); fig.tight_layout()
    charts.append(_fig_to_gallery(fig,"Cost 3｜Prediction Error Distribution"))
    return charts


def tool_scheduling(question,p):
    x=APP["analytics"]["scheduling"]; top=p["top_n"]
    q = str(question)

    if any(k in q for k in ["缺料","材料未齊","待料","催料"]):
        d=x["material_hold"].sort_values(["距離交期天數","排程分數"],ascending=[True,False]).head(top)
        cols=[c for c in [
            "排程順位","工單ID","SKU","客戶等級","急單","材料齊套","品質風險",
            "距離交期天數","排程分數"
        ] if c in d.columns]
        table=d[cols].copy()

    elif any(k in q for k in ["品質風險","品質優先"]):
        d=x["quality_hold"].sort_values(["排程分數","距離交期天數"],ascending=[False,True]).head(top)
        cols=[c for c in [
            "排程順位","工單ID","SKU","客戶等級","急單","材料齊套","品質風險",
            "距離交期天數","排程分數","狀態"
        ] if c in d.columns]
        table=d[cols].copy()

    elif any(k in q for k in ["機台利用率","CNC 機台利用率","負載最高","負載最低"]):
        d=x["util"].copy().sort_values("Rule-Based利用率(%)", ascending=False)
        table=d[["CNC機台","FIFO利用率(%)","Rule-Based利用率(%)"]].copy()

    elif any(k in q for k in ["順位差異","順位提前","排程順位差"]):
        d=x["rank_compare"].copy()
        d["順位差異絕對值"] = d["順位提前"].abs()
        d=d.sort_values(["順位差異絕對值","順位提前"],ascending=[False,False]).head(top)
        table=d[["工單ID","RuleBased順位","FIFO順位","順位提前","順位差異絕對值"]].copy()

    elif "FIFO" in q or "Rule-Based" in q or "比較" in q:
        k=x["kpis"]
        table=pd.DataFrame([
            {"比較指標":"急單平均順位","Rule-Based":k.get("急單Rule-Based平均順位"),"FIFO":k.get("急單FIFO平均順位"),"較佳方向":"越小越前"},
            {"比較指標":"Makespan (hr)","Rule-Based":k.get("Rule-Based Makespan(hr)"),"FIFO":k.get("FIFO Makespan(hr)"),"較佳方向":"越低越好"},
            {"比較指標":"平均 CNC 利用率 (%)","Rule-Based":k.get("Rule-Based平均利用率(%)"),"FIFO":k.get("FIFO平均利用率(%)"),"較佳方向":"越高越好"},
        ])

    else:
        d=x["schedule"].head(top) if len(x["schedule"]) else x["priority"].head(top)
        cols=[c for c in [
            "排程順位","RuleBased順位","工單ID","SKU","客戶等級","急單","材料齊套","品質風險",
            "距離交期天數","排程分數","機台","開始工時","完成工時"
        ] if c in d.columns]
        table=d[cols].copy()

    return {"name":"scheduling","kpis":x["kpis"],"table":table,"charts":scheduling_original_charts(x)}
def tool_anomaly(question,p):
    x=APP["analytics"]["anomaly"]; d=x["data"].head(p["top_n"])
    cols=[c for c in [
        "Anomaly Rank","工單ID","工序序號","工序名稱","機台ID","Anomaly Score",
        "異常層級","Rule判定","IF判定","方法對照","主要偏差特徵","OEE","品質良率","有品質異常紀錄"
    ] if c in d.columns]
    return {"name":"anomaly","kpis":x["kpis"],"table":d[cols].copy(),"charts":anomaly_original_charts(x)}

def tool_customer(question,p):
    x=APP["analytics"]["customer"]; d=x["data"]
    seg=None
    if any(k in question for k in ["流失","沉睡","喚回"]):
        seg="沉睡／流失風險型"
    else:
        for ss in ["高價值活躍型","成長潛力型","價格敏感型","沉睡／流失風險型"]:
            if ss in question:
                seg=ss
    if seg:
        d=d[d["K-Means行為分群"]==seg]
    d=d.sort_values("年化採購金額_TWD",ascending=False).head(p["top_n"])
    cols=[c for c in [
        "客戶ID","客戶名稱","既有客戶等級（ERP）","年化採購金額_TWD","訂單次數","報價成交率",
        "平均折扣率","平均預估毛利率","最近交易天數","K-Means行為分群","建議策略"
    ] if c in d.columns]
    return {"name":"customer","kpis":x["kpis"],"table":d[cols].copy(),"charts":customer_original_charts(x)}

def tool_cost(question,p):
    obj=APP["analytics"]["cost"]; pl=p.get("product_line")
    if pl:
        pl=str(pl).replace("產品線","").strip()
    if not pl or pl not in obj["product_lines"]:
        detected=None
        for candidate in obj["product_lines"]:
            if str(candidate) in str(question):
                detected=candidate; break
        pl=detected

    # Explicit product line (Q6/Q8): preserve the original per-line regression behavior.
    if pl:
        r=run_cost_model(obj,pl)
        d=r["result"].head(p["top_n"])
        return {"name":"cost","kpis":r["kpis"],"table":d.copy(),"charts":cost_original_charts(r)}

    # No product line specified (Q9/Q10): run the existing regression independently
    # for every product line, then compare their test-set error rows by error rate.
    runs=[]
    rows=[]
    for line in obj["product_lines"]:
        try:
            rr=run_cost_model(obj,line)
            runs.append(rr)
            temp=rr["result"].copy()
            temp["模型產品線"] = line
            rows.append(temp)
        except Exception:
            continue
    if not rows:
        raise ValueError("所有產品線皆無足夠資料建立成本模型。")
    combined=pd.concat(rows,ignore_index=True,sort=False).sort_values("誤差率(%)",ascending=False)
    d=combined.head(p["top_n"]).copy()
    kpis={
        "已評估產品線":len(runs),
        "最高誤差率(%)":round(float(combined["誤差率(%)"].max()),2),
        "各產品線平均MAPE(%)":round(float(np.mean([r["kpis"]["MAPE(%)"] for r in runs])),2),
    }
    # Use the charts of the product line containing the highest-error row so the
    # visualization remains based on one valid regression model.
    top_line=str(d.iloc[0]["產品線"]) if len(d) and "產品線" in d.columns else runs[0]["product_line"]
    chart_run=next((r for r in runs if r["product_line"]==top_line), runs[0])
    return {"name":"cost","kpis":kpis,"table":d,"charts":cost_original_charts(chart_run)}


RUNNERS={"scheduling":tool_scheduling,"anomaly":tool_anomaly,"customer":tool_customer,"cost":tool_cost}

TOOL_LABELS = {
    "scheduling": "排程 / Scheduling",
    "anomaly": "製程異常 / Anomaly",
    "customer": "客戶 / Customer",
    "cost": "成本 / Cost",
}

def make_actions(plan_obj, results):
    action = plan_obj.get("requested_action", "display")
    rows = []

    # Q10 / management_top3: create three domain-diverse actions rather than
    # taking the first three rows from the first tool.
    if action == "management_top3":
        by_name = {r["name"]: r for r in results if r.get("table") is not None and len(r.get("table"))}
        if "anomaly" in by_name:
            z = by_name["anomaly"]["table"].iloc[0]
            rows.append({"Action":"建立設備／製程檢查","Target":z.get("機台ID", z.get("工單ID","—")),"Status":"Created (Simulation)"})
        if "scheduling" in by_name:
            z = by_name["scheduling"]["table"].iloc[0]
            rows.append({"Action":"建立排程覆核任務","Target":z.get("工單ID","—"),"Status":"Created (Simulation)"})
        if "customer" in by_name:
            z = by_name["customer"]["table"].iloc[0]
            rows.append({"Action":"建立業務追蹤任務","Target":z.get("客戶ID","—"),"Status":"Created (Simulation)"})
        elif "cost" in by_name:
            z = by_name["cost"]["table"].iloc[0]
            rows.append({"Action":"建立成本覆核任務","Target":z.get("工單ID","—"),"Status":"Created (Simulation)"})
        return pd.DataFrame(rows[:3], columns=["Action","Target","Status"])

    for r in results:
        d = r["table"]
        if d is None or len(d) == 0:
            continue
        if r["name"] == "scheduling":
            if action == "material_expedite":
                for _,z in d.head(3).iterrows():
                    rows.append({"Action":"建立採購催料任務","Target":z.get("工單ID","—"),"Status":"Created (Simulation)"})
            elif action == "schedule_review":
                for _,z in d.head(3).iterrows():
                    rows.append({"Action":"建立排程覆核任務","Target":z.get("工單ID","—"),"Status":"Created (Simulation)"})
            elif action == "display":
                rows.append({"Action":"更新排產優先清單","Target":f"Top {min(len(d), plan_obj.get('top_n',5))} 工單","Status":"Completed"})
        elif r["name"] == "anomaly":
            if action == "machine_inspection":
                seen=set()
                for _,z in d.iterrows():
                    target=z.get("機台ID",z.get("工單ID","—"))
                    if target in seen:
                        continue
                    seen.add(target)
                    rows.append({"Action":"建立設備／製程檢查","Target":target,"Status":"Created (Simulation)"})
                    if len(rows) >= 3:
                        break
            elif action == "display":
                rows.append({"Action":"更新異常觀察清單","Target":f"Top {min(len(d), plan_obj.get('top_n',5))} 事件","Status":"Completed"})
        elif r["name"] == "customer":
            if action == "sales_follow_up":
                for _,z in d.head(3).iterrows():
                    rows.append({"Action":"建立業務追蹤任務","Target":z.get("客戶ID","—"),"Status":"Created (Simulation)"})
            elif action == "display":
                rows.append({"Action":"更新客戶分群清單","Target":f"Top {min(len(d), plan_obj.get('top_n',5))} 客戶","Status":"Completed"})
        elif r["name"] == "cost":
            if action == "cost_review":
                for _,z in d.head(3).iterrows():
                    rows.append({"Action":"建立成本覆核任務","Target":z.get("工單ID","—"),"Status":"Created (Simulation)"})
            elif action == "display":
                rows.append({"Action":"更新成本風險清單","Target":f"Top {min(len(d), plan_obj.get('top_n',5))} 工單","Status":"Completed"})

    if not rows:
        rows=[{"Action":"完成分析","Target":"本次查詢","Status":"No additional action required"}]
    return pd.DataFrame(rows, columns=["Action","Target","Status"])

def build_decision_summary(question, plan_obj, results, actions):
    lines=[]
    top_n = int(plan_obj.get("top_n", 5) or 5)

    failed = [
        (r.get("name"), (r.get("kpis") or {}).get("error"))
        for r in results
        if "error" in (r.get("kpis") or {})
    ]
    if failed:
        for name, err in failed:
            label = TOOL_LABELS.get(name, name)
            lines.append(f"{label} 分析未完成：{err}")
        return "\n\n".join(lines)

    for r in results:
        d=r["table"]
        if r["name"]=="scheduling" and len(d):
            k=r.get("kpis",{})
            if any(v in str(question) for v in ["FIFO","Rule-Based","比較"]):
                lines.append(
                    "排程比較："
                    f"急單平均順位 Rule-Based={k.get('急單Rule-Based平均順位','—')}、FIFO={k.get('急單FIFO平均順位','—')}；"
                    f"Makespan Rule-Based={k.get('Rule-Based Makespan(hr)','—')}hr、FIFO={k.get('FIFO Makespan(hr)','—')}hr；"
                    f"平均利用率 Rule-Based={k.get('Rule-Based平均利用率(%)','—')}%、FIFO={k.get('FIFO平均利用率(%)','—')}%。"
                )
            elif any(v in str(question) for v in ["缺料","材料未齊","催料"]):
                ids=[str(v) for v in d.get("工單ID",pd.Series(dtype=str)).head(top_n).tolist()]
                lines.append("待料：優先催料 " + ("、".join(ids) if ids else "目前沒有符合條件的工單") + "。")
            else:
                ids=[str(v) for v in d.get("工單ID",pd.Series(dtype=str)).head(top_n).tolist()]
                lines.append("排程：優先工單為 " + ("、".join(ids) if ids else "目前沒有符合條件的工單") + "。")
        elif r["name"]=="anomaly" and len(d):
            z=d.iloc[0]
            lines.append(
                f"異常：優先檢視 {z.get('工單ID','—')} / {z.get('機台ID','—')}，"
                f"Anomaly Score {z.get('Anomaly Score','—')}；此分數代表相對異常程度，不是故障機率。"
            )
        elif r["name"]=="customer" and len(d):
            ids=[str(v) for v in d.get("客戶ID",pd.Series(dtype=str)).head(top_n).tolist()]
            lines.append("客戶：優先名單 " + "、".join(ids) + "。")
        elif r["name"]=="cost" and len(d):
            ids=[str(v) for v in d.get("工單ID",pd.Series(dtype=str)).head(top_n).tolist()]
            lines.append("成本：優先覆核 " + "、".join(ids) + "；迴歸係數用於判讀關聯方向與相對影響。")

    if not lines:
        lines=["本次查詢已完成，目前沒有符合條件的資料。"]

    if len(actions) and plan_obj.get("requested_action") != "display":
        act_txt="；".join([f"{z['Action']}→{z['Target']}" for _,z in actions.head(3).iterrows()])
        lines.append("建議後續處理：" + act_txt + "。")

    return "\n\n".join(lines)


def compact_evidence(results):
    out=[]
    for r in results:
        df = r["table"].head(5).copy()
        df = df.astype(object).where(pd.notna(df), None)
        out.append({"tool":r["name"],"kpis":r["kpis"],"top_rows":df.to_dict("records")})
    return out

def render_evidence_sections(results):
    """Render one independent table per tool so unrelated schemas are never concatenated."""
    sections=[]
    for r in results:
        label = TOOL_LABELS.get(r["name"], r["name"])
        df = r.get("table")
        if df is None or df.empty:
            body = df_to_dark_html(pd.DataFrame(), f"{label} 沒有可顯示資料")
        else:
            df = df.dropna(axis=1, how="all")
            body = df_to_dark_html(df)
        sections.append(
            f"<div class='evidence-section'><div class='evidence-title'>{label}</div>{body}</div>"
        )
    return "".join(sections) if sections else df_to_dark_html(pd.DataFrame(), "尚無 Evidence")

def render_kpi_sections(results):
    sections=[]
    for r in results:
        label = TOOL_LABELS.get(r["name"], r["name"])
        cards = "<div class='kpi-wrap'>"
        for k,v in r.get("kpis",{}).items():
            cards += (
                "<div class='kpi-card'><div style='font-size:12px;font-weight:800;opacity:.72'>"
                f"{label}</div><b>{k}</b><br><span style='font-size:18px;font-weight:700'>{v}</span></div>"
            )
        cards += "</div>"
        sections.append(cards)
    return "".join(sections)


# ----------------------------------------------------------------------
# Agent execution
# ----------------------------------------------------------------------
def _tool_ready(name):
    return APP["tool_status"].get(name) == "Ready" and APP["analytics"].get(name) is not None



def build_chart_notes(results):
    chart_notes = {
        "Scheduling 1｜Top 15 Work Orders by Priority Score":
            "依工單優先分數排序，可快速辨識目前最值得優先安排的工單，協助現場將急單、客戶等級、材料與品質條件轉換成可執行的排產順序。",
        "Scheduling 2｜Rule-Based CNC Production Schedule":
            "以甘特圖呈現各 CNC 機台的工單配置與時間區段，可檢視工作是否集中於少數設備，並掌握整體生產節奏。",
        "Scheduling 3｜CNC Utilization after Rule-Based Scheduling":
            "比較各 CNC 機台利用率，可辨識高負載與仍有可用產能的設備，作為負載平衡與派工調整的依據。",
        "Scheduling 4｜FIFO vs Rule-Based Dispatch Priority":
            "比較 FIFO 與 Rule-Based 的工單順位變化，可看出急單、重要客戶與材料齊套等條件如何影響生產優先順序。",
        "Scheduling 5｜FIFO vs Rule-Based Scheduling Efficiency":
            "從急單平均順位、Makespan 與設備利用率比較兩種排程方式，協助評估規則式排程對生產效率的改善。",
        "Scheduling 6｜MES Work Order Screening":
            "將未完工工單依可立即排產、待料與品質風險進行篩選，方便管理者先掌握真正可執行與需先處理的工單。",
        "Anomaly 1｜Top 15 MES Anomaly Events":
            "依 Anomaly Score 排序製程事件，可優先檢視偏離歷史基準最明顯的事件，集中設備與製程改善資源。",
        "Anomaly 2｜PCA 2D MES Anomaly Map":
            "透過 PCA 將多維製程特徵投影至二維空間，協助觀察正常與異常事件在整體資料分布中的相對位置。",
        "Anomaly 3｜Top Anomaly Multi-Factor Deviation Profile":
            "呈現最高異常事件的主要偏差來源，可進一步判斷問題較偏向加工、換線、停機、報廢、重工或產出效率。",
        "Anomaly 4｜Rule-Based vs Isolation Forest Matrix":
            "比較規則警示與 Isolation Forest 判定，可辨識共同警示、單變數警示，以及規則較難發現的多變數隱性異常。",
        "Customer 1｜ERP Customer Segmentation - K-Means":
            "以 PCA 視覺化 K-Means 客戶分群，可觀察不同客群在採購金額、訂單頻率、報價互動與最近交易等特徵上的差異。",
        "Customer 2｜Customer Segment Distribution":
            "顯示各類客戶數量分布，協助掌握目前客群結構與不同經營策略所涵蓋的客戶規模。",
        "Customer 3｜Customer Segment Profile":
            "比較不同客群的主要特徵輪廓，可作為客戶維繫、成長、價格管理與喚回策略的依據。",
        "Cost 1｜Actual vs Predicted Cost with ±3% Range":
            "比較實際製造成本與模型預測值，並以 ±3% 範圍觀察預測穩定度，可快速找出需要進一步覆核的工單。",
        "Cost 2｜Manufacturing Cost Drivers":
            "呈現各成本變數的標準化迴歸係數，可協助理解哪些材料、人工、加工、換線或製造費因素與成本變動關聯較高。",
        "Cost 3｜Prediction Error Distribution":
            "觀察成本預測誤差分布，可判斷多數工單的模型誤差是否集中，並辨識少數需要進一步確認的高誤差案例。",
    }

    items = []
    for r in results:
        for item in r.get("charts", []):
            caption = item[1] if isinstance(item, tuple) and len(item) > 1 else "分析圖表"
            note = chart_notes.get(caption, "此圖表提供本次分析的重要視覺化結果，可搭配 Evidence 與策略建議進行管理判讀。")
            items.append((caption, note))

    if not items:
        return "<div class='chart-note-panel'><div class='chart-note-main-title'>圖表重點說明</div><div class='chart-note-card'>執行分析後，將在此顯示每張圖表的判讀重點。</div></div>"

    html = "<div class='chart-note-panel'><div class='chart-note-main-title'>圖表重點說明</div>"
    for i, (caption, note) in enumerate(items, 1):
        html += (
            "<div class='chart-note-card'>"
            f"<div class='chart-note-title'>{i}. {caption}</div>"
            f"<div class='chart-note-text'>{note}</div>"
            "</div>"
        )
    html += "</div>"
    return html


def build_strategy_panel(question, plan_obj, results):
    """
    Generate management strategy only when the corresponding analysis succeeded.
    """
    cards = []

    for r in results:
        name = r.get("name")
        d = r.get("table", pd.DataFrame())
        k = r.get("kpis", {}) or {}

        if "error" in k:
            cards.append(
                ("分析狀態",
                 "本次分析尚未完成，因此暫不產生策略建議。"
                 "請先確認資料與分析功能正常後，再依實際結果進行管理判讀。")
            )
            continue

        if name == "customer":
            top_ids = []
            if isinstance(d, pd.DataFrame) and len(d) and "客戶ID" in d.columns:
                top_ids = [str(v) for v in d["客戶ID"].head(3).tolist()]
            target_text = "、".join(top_ids) if top_ids else "重點客戶"
            cards.append(
                ("客戶經營策略",
                 f"優先聚焦 {target_text}，依客戶分群設計差異化互動。"
                 "高價值客戶可強化長期合作與交叉銷售；成長型客戶可增加接觸與產品組合；"
                 "價格敏感客戶可透過價值溝通與毛利管理提升成交品質。")
            )

        elif name == "cost":
            mape = k.get("MAPE(%)", k.get("各產品線平均MAPE(%)", "—"))
            cards.append(
                ("成本管理策略",
                 f"目前成本模型可作為報價與成本覆核的輔助依據（MAPE：{mape}）。"
                 "建議優先檢視預測誤差較高的工單，並回查材料、加工、換線與製造費用，"
                 "讓業務報價與製造成本逐步形成一致的管理基準。")
            )

        elif name == "scheduling":
            ready = k.get("可立即排產", "—")
            rb_ms = k.get("Rule-Based Makespan(hr)", "—")
            rb_util = k.get("Rule-Based平均利用率(%)", "—")
            cards.append(
                ("生產排程策略",
                 f"目前可立即排產工單為 {ready} 張，Rule-Based Makespan 為 {rb_ms} hr，"
                 f"平均 CNC 利用率為 {rb_util}%。"
                 "建議優先安排高優先度且材料齊套的工單，"
                 "並搭配急單、交期與機台負載進行派工調整，"
                 "使重要工單更快進入生產並維持設備使用效率。")
            )

        elif name == "anomaly":
            high = k.get("High Anomaly", "—")
            hidden = k.get("多變數隱性異常", "—")
            cards.append(
                ("製程改善策略",
                 f"目前 High Anomaly 事件為 {high} 筆，其中多變數隱性異常為 {hidden} 筆。"
                 "建議優先檢視異常分數較高的事件與主要偏差特徵，"
                 "再對應機台、加工工時、停機、報廢與重工資訊安排確認，"
                 "可更有效率地集中改善資源。")
            )

    if not cards:
        cards = [("策略建議", "目前已完成資料分析，可依 Evidence 與圖表進一步安排管理追蹤。")]

    html = "<div class='strategy-panel'><div class='strategy-main-title'>策略建議</div>"
    for title, body in cards:
        html += (
            "<div class='strategy-card'>"
            f"<div class='strategy-title'>{title}</div>"
            f"<div class='strategy-text'>{body}</div>"
            "</div>"
        )
    html += "</div>"
    return html

def run_ai_agent(model_label, question, progress=gr.Progress()):
    if not question or not str(question).strip():
        return (
            "請輸入問題。",
            "",
            df_to_dark_html(pd.DataFrame(), "尚無 Evidence"),
            [],
            "<div class='strategy-panel'><div class='strategy-main-title'>策略建議</div><div class='strategy-card'><div class='strategy-text'>完成資料初始化後，即可顯示對應策略建議。</div></div></div>",
            "<div class='chart-note-panel'><div class='chart-note-main-title'>圖表重點說明</div><div class='chart-note-card'>執行分析後，將在此顯示每張圖表的判讀重點。</div></div>",
            df_to_dark_html(pd.DataFrame(columns=["Action","Target","Status"]), "尚無 Action"),
            "{}",
            "No query",
            "",
        )

    ready_count = sum(v == "Ready" for v in APP["tool_status"].values())
    if ready_count == 0:
        return (
            "⚠️ 請先到 System Setup 上傳 YUNFA_ERP.xlsx / YUNFA_MES.xlsx 並初始化。",
            "",
            df_to_dark_html(pd.DataFrame(), "尚無 Evidence"),
            [],
            "<div class='strategy-panel'><div class='strategy-main-title'>策略建議</div><div class='strategy-card'><div class='strategy-text'>完成資料初始化後，即可顯示對應策略建議。</div></div></div>",
            "<div class='chart-note-panel'><div class='chart-note-main-title'>圖表重點說明</div><div class='chart-note-card'>執行分析後，將在此顯示每張圖表的判讀重點。</div></div>",
            df_to_dark_html(pd.DataFrame(columns=["Action","Target","Status"]), "尚無 Action"),
            "{}",
            "Dataset not initialized",
            "",
        )

    t0 = time.perf_counter()
    progress(0.1, desc="Planner")

    # If a model is loaded, require the dropdown to match the loaded model.
    expected_id = MODEL_OPTIONS.get(model_label)
    if APP["model"] is not None and APP["model_name"] != expected_id:
        return (
            "⚠️ 目前載入的模型與下拉選單不同。請先按「載入 / 切換模型」。",
            "",
            df_to_dark_html(pd.DataFrame(), "尚無 Evidence"),
            [],
            "<div class='strategy-panel'><div class='strategy-main-title'>策略建議</div><div class='strategy-card'><div class='strategy-text'>完成資料初始化後，即可顯示對應策略建議。</div></div></div>",
            "<div class='chart-note-panel'><div class='chart-note-main-title'>圖表重點說明</div><div class='chart-note-card'>執行分析後，將在此顯示每張圖表的判讀重點。</div></div>",
            df_to_dark_html(pd.DataFrame(columns=["Action","Target","Status"]), "尚無 Action"),
            "{}",
            f"Loaded={APP.get('model_label')} / Selected={model_label}",
            "",
        )

    plan_obj, raw = plan_question(question)
    planner_sec = time.perf_counter() - t0

    progress(0.4, desc="ERP / MES Tools")
    results = []
    for name in plan_obj["tools"]:
        if name not in RUNNERS:
            continue
        if not _tool_ready(name):
            results.append({
                "name": name,
                "kpis": {"error": APP["tool_error"].get(name, "Tool not ready")},
                "table": pd.DataFrame(),
                "charts": [_error_chart(APP["tool_error"].get(name, "Tool not ready"))],
            })
            continue
        try:
            results.append(RUNNERS[name](question, plan_obj))
        except Exception as e:
            results.append({
                "name": name,
                "kpis": {"error": f"{type(e).__name__}: {e}"},
                "table": pd.DataFrame(),
                "charts": [_error_chart(f"{type(e).__name__}: {e}")],
            })

    progress(0.72, desc="Decision / Action / Charts")
    actions = make_actions(plan_obj, results)
    final_summary = build_decision_summary(question, plan_obj, results, actions)
    evidence = compact_evidence(results)

    gallery = []
    for r in results:
        gallery.extend(r.get("charts", []))

    evidence_html = render_evidence_sections(results)
    kpi_html = render_kpi_sections(results)
    strategy_html = build_strategy_panel(question, plan_obj, results)
    chart_notes_html = build_chart_notes(results)

    total_sec = time.perf_counter() - t0
    model_text = APP.get("model_label") or "未載入模型（使用 deterministic fallback）"

    decision = (
        "### AI Agent 分析摘要\n\n"
        f"{final_summary}"
    )

    meta_html = (
        "<div class='system-meta'>"
        f"目前模型：{model_text}　｜　Planner：{planner_sec:.2f}s　｜　總耗時：{total_sec:.2f}s"
        "</div>"
    )

    planner_json = json.dumps(plan_obj, ensure_ascii=False, indent=2)
    trace = (
        f"Question: {question}\n\n"
        f"Loaded Model: {model_text}\n\n"
        f"Raw Planner Output:\n{raw}\n\n"
        f"Normalized Plan:\n{planner_json}\n\n"
        f"Routing Source: {plan_obj.get('routing_source','unknown')}\n"
        f"Executed Tools: {[r['name'] for r in results]}\n"
        f"Planner Time: {planner_sec:.3f}s\n"
        f"Total Time: {total_sec:.3f}s\n\n"
        "Notes:\n"
        "- Anomaly Score = relative anomaly priority, NOT failure probability.\n"
        "- Regression coefficients = direction/relative association, NOT causality.\n"
        "- Multi-tool questions do not invent cross-system joins without a valid key."
    )

    progress(1.0, desc="完成")
    return (
        decision,
        kpi_html,
        evidence_html,
        gallery,
        strategy_html,
        chart_notes_html,
        df_to_dark_html(actions, "本次沒有 Action"),
        planner_json,
        trace,
        meta_html,
    )

# ----------------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------------
CUSTOM_CSS = """

/* ============================================================
   Typography｜正黑體 / 專業無襯線
   ============================================================ */
.gradio-container,
.gradio-container *,
.yunfa-hero,
.yunfa-hero *,
button,
input,
textarea,
select,
table,
th,
td {
    font-family:
      "Microsoft JhengHei",
      "Microsoft JhengHei UI",
      "Noto Sans TC",
      "Noto Sans CJK TC",
      "PingFang TC",
      Arial,
      "Liberation Sans",
      sans-serif !important;
    font-style: normal !important;
}

/* ============================================================
   雲發科 × YunTech｜淡綠暖色柔性主視覺
   ============================================================ */
.gradio-container {
    background:
      linear-gradient(180deg, #FAFBF5 0%, #FBF7EC 48%, #F7FAF5 100%) !important;
    color: #38483F !important;
}
footer { display: none !important; }

.yunfa-hero {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:20px;
    padding:18px 22px;
    margin-bottom:14px;
    border:1px solid #C9DCCF;
    border-radius:20px;
    background:linear-gradient(120deg,#EAF5EC 0%,#F8F3E6 52%,#F1F6EE 100%);
    box-shadow:0 8px 24px rgba(61,103,77,.08);
}
.yunfa-hero-left {
    display:flex;
    align-items:center;
    gap:16px;
}
.yuntech-mark {
    display:flex;
    align-items:center;
    gap:11px;
    padding:10px 14px;
    border-radius:16px;
    background:rgba(255,255,255,.82);
    border:1px solid #C8DCCC;
    min-width:270px;
}
.yuntech-logo-img {
    width:64px;
    height:64px;
    object-fit:contain;
    display:block;
    flex:0 0 auto;
}
.yuntech-wordmark-main {
    font-size:20px;
    font-weight:900;
    color:#25765B;
    line-height:1.15;
}
.yuntech-wordmark-sub {
    font-size:12px;
    color:#67776D;
    margin-top:3px;
}
.yunfa-title {
    font-size:31px;
    line-height:1.2;
    font-weight:900;
    color:#2F624E;
}
.yunfa-subtitle {
    font-size:16px;
    color:#6F7B72;
    margin-top:7px;
}
.yunfa-badge {
    border-radius:999px;
    padding:9px 14px;
    font-weight:800;
    color:#52675A;
    background:#FFF8DE;
    border:1px solid #E1D7A9;
    white-space:nowrap;
}

/* Primary controls｜藍色按鈕、綠色線條 */
button.primary,
.primary,
.gr-button-primary {
    background:#6F91B2 !important;
    border:1px solid #5D7F9F !important;
    color:#FFFFFF !important;
}
button.primary:hover,
.primary:hover,
.gr-button-primary:hover {
    background:#5F82A4 !important;
    border-color:#527493 !important;
    color:#FFFFFF !important;
}
.tabs button.selected {
    color:#2F6A53 !important;
    border-color:#7FAE93 !important;
    background:#F5FAF4 !important;
}


/* Top-level navigation */
.gradio-container .tab-nav {
    border-bottom:1px solid #BFD4C4 !important;
}

/* Tabs｜綠色線條，白/暖色背景 */
.gradio-container [role="tab"][aria-selected="true"] {
    color:#2F6A53 !important;
    border-color:#7FAE93 !important;
}
.gradio-container [role="tab"] {
    font-weight:800 !important;
}

/* Upload instructions: intentionally large for classroom projection */
.upload-instruction {
    border: 1.5px solid #A9C9AE;
    border-radius: 14px;
    padding: 15px 17px;
    margin-bottom: 8px;
    background: linear-gradient(135deg,#F3F8DE 0%,#FFF7DE 100%);
    box-shadow: 0 3px 10px rgba(95,125,86,.06);
}
.upload-instruction .file-kind {
    font-size: 21px;
    font-weight: 850;
    line-height: 1.35;
    color:#315D48 !important;
}
.upload-instruction .file-name {
    font-size: 19px;
    font-weight: 800;
    line-height: 1.4;
    margin-top: 4px;
    color:#37493F !important;
}
.upload-instruction .file-note {
    font-size: 15px;
    color:#69766E !important;
    margin-top: 4px;
}

/* Make Gradio's own file labels easier to see too */
#erp-upload,
#mes-upload {
    min-height: 150px;
    border:1px solid #B8CFBE !important;
    border-radius:14px !important;
    background:#FCFDF8 !important;
}
#erp-upload .block-label,
#mes-upload .block-label {
    color:#426453 !important;
}


.setup-section-title {
    border-left:4px solid #76A98B;
    background:linear-gradient(90deg,#F1F7E5 0%,rgba(255,255,255,0) 85%);
    padding:10px 14px;
    border-radius:8px;
    margin-bottom:12px;
}
.setup-section-title h3 {
    margin:0 !important;
    color:#315E49 !important;
    font-size:20px !important;
}

/* Slightly larger setup text for projector / classroom use */
.setup-note {
    font-size: 16px;
    line-height: 1.55;
}



/* ---------- AI Agent result layout ---------- */
.result-summary-card {
    background:#FBFCF7 !important;
    border:1px solid #D8E4D8;
    border-radius: 14px;
    padding: 16px 18px;
    min-height: 120px;
}
.result-summary-card h3 {
    margin-top: 0;
    font-size: 21px;
}
.result-summary-card p,
.result-summary-card div {
    font-size: 17px;
    line-height: 1.7;
}
.chart-note-panel {
    border:1px solid #D6E2D7;
    border-radius: 16px;
    padding: 18px;
    background:#F7FAF5 !important;
}
.chart-note-main-title {
    font-size: 23px;
    font-weight: 900;
    margin-bottom: 14px;
    color: #5E7380 !important;
}
.chart-note-card {
    background: #ffffff !important;
    border:1px solid #E1E8DD;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.chart-note-title {
    font-size: 18px;
    font-weight: 850;
    color: #5E7380 !important;
    margin-bottom: 6px;
}
.chart-note-text {
    font-size: 17px;
    line-height: 1.7;
    color: #5B6D78 !important;
}
.system-meta {
    margin-top: 18px;
    padding: 10px 14px;
    border-top: 1px solid #2E493D;
    font-size: 13px;
    opacity: .68;
}
/* ---------- AI Agent user-facing typography ---------- */
.strategy-panel {
    border:1px solid #E0D7B9;
    border-radius: 16px;
    padding: 18px;
    background: #FBF8F2 !important;
    min-height: 260px;
}
.strategy-main-title {
    font-size: 24px;
    font-weight: 900;
    margin-bottom: 14px;
    color: #4D6575 !important;
}
.strategy-card {
    background: #ffffff !important;
    border: 1px solid #E5DED3;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 14px;
}
.strategy-title {
    font-size: 20px;
    font-weight: 850;
    color: #5B7180 !important;
    margin-bottom: 8px;
}
.strategy-text {
    font-size: 18px;
    line-height: 1.75;
    color: #546773 !important;
}
.agent-run-button button,
.agent-run-button .primary,
#business-run,
#business-run button,
#production-run,
#production-run button {
    font-size:18px !important;
    font-weight:850 !important;
    min-height:50px !important;
    background:#6E91B3 !important;
    border:1px solid #587B9B !important;
    color:#FFFFFF !important;
    box-shadow:0 3px 10px rgba(74,104,134,.14) !important;
}
.agent-run-button button:hover,
.agent-run-button .primary:hover,
#business-run:hover,
#business-run button:hover,
#production-run:hover,
#production-run button:hover {
    background:#5F83A6 !important;
    border-color:#4F7394 !important;
    color:#FFFFFF !important;
}



.agent-section-title {
    border-left:4px solid #7FAE93;
    padding-left: 12px;
    margin: 6px 0 12px 0;
}
.agent-section-title h2,
.agent-section-title h3 {
    margin: 0 !important;
    color:#315E49 !important;
}
.strategy-main-title,
.chart-note-main-title {
    position: relative;
    padding-left: 12px;
}
.strategy-main-title::before,
.chart-note-main-title::before {
    content: "";
    position: absolute;
    left: 0;
    top: 15%;
    width: 4px;
    height: 70%;
    border-radius: 4px;
    background:#8DB29A;
}

/* Clean, professional headings and controls */
.yunfa-title,
.yuntech-wordmark-main,
.strategy-main-title,
.chart-note-main-title,
.evidence-title {
    letter-spacing: 0.01em;
}
.gradio-container label,
.gradio-container .block-label {
    color:#4E626F !important;
    font-weight:750 !important;
}

/* ---------- Theme-safe tables ---------- */
.yunfa-table-wrap {
    width: 100%;
    overflow-x: auto;
    border: 1px solid #D2DAE1;
    border-radius: 12px;
    background: #F7F8F9 !important;
}

.yunfa-data-table {
    width: 100%;
    border-collapse: collapse;
    background: #F7F8F9 !important;
    color: #2E493D !important;
    font-size: 14px;
}

.yunfa-data-table thead th {
    background: #EEF2F5 !important;
    color: #415662 !important;
    font-weight: 800 !important;
    border-right: 1px solid #D2DAE1 !important;
    border-bottom: 1px solid #C0CCD5 !important;
    padding: 10px 12px !important;
    white-space: nowrap;
    text-align: left;
}

.yunfa-data-table tbody td {
    background: #F7F8F9 !important;
    color: #2E493D !important;
    border-right: 1px solid #E3E8EC !important;
    border-bottom: 1px solid #E3E8EC !important;
    padding: 9px 12px !important;
    white-space: nowrap;
}

.yunfa-data-table tbody tr:nth-child(even) td {
    background: #F3F5F7 !important;
}

.yunfa-data-table tbody tr:hover td {
    background: #EEF2F5 !important;
    color: #40525E !important;
}

.yunfa-table-empty {
    border: 1px dashed #C0CCD5;
    border-radius: 10px;
    padding: 18px;
    background: #F7F8F9 !important;
    color: #697982 !important;
    font-size: 15px;
}

/* KPI cards are light by design, so force all text dark. */
.kpi-card,
.kpi-card * {
    color: #38483F !important;
}
.kpi-wrap {
    display:flex;
    gap:10px;
    flex-wrap:wrap;
    margin:8px 0 14px 0;
}
.kpi-card {
    border:1px solid #D9E0E6;
    border-radius:10px;
    padding:10px 14px;
    background:#FFFFFF !important;
    min-width:140px;
}

.kpi-card {
    border:1px solid #D4DCE3 !important;
    border-radius:12px !important;
    padding:12px 16px !important;
    background:#FFFFFF !important;
    min-width:150px;
    box-shadow:0 3px 10px rgba(67, 103, 82, 0.06);
}
.kpi-card b {
    color:#475D6D !important;
    font-size:15px !important;
    font-weight:800 !important;
}
.kpi-card span {
    color:#2E4454 !important;
    font-size:20px !important;
    font-weight:900 !important;
}

.evidence-section {
    margin: 0 0 22px 0;
}
.evidence-title {
    font-size: 18px;
    font-weight: 850;
    margin: 6px 0 8px 2px;
}
"""

def create_app():
    with gr.Blocks(
        title="雲發科 AI Agent 智慧製造戰情室",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="gray"),
        css=CUSTOM_CSS,
    ) as demo:
        gr.HTML(
            """
            <div class="yunfa-hero">
              <div class="yunfa-hero-left">
                <div class="yuntech-mark" aria-label="YunTech 國立雲林科技大學">
                  <img class="yuntech-logo-img" src="data:image/png;base64,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" alt="國立雲林科技大學 YunTech Logo">
                  <div>
                    <div class="yuntech-wordmark-main">YunTech</div>
                    <div class="yuntech-wordmark-sub">國立雲林科技大學</div>
                  </div>
                </div>
                <div>
                  <div class="yunfa-title">雲發科 AI Agent 智慧製造戰情室</div>
                  <div class="yunfa-subtitle">ERP／MES × AI Analytics × Small Language Model × Management Strategy</div>
                </div>
              </div>
              <div class="yunfa-badge">智慧製造・柔性決策</div>
            </div>
            """
        )

        with gr.Tab("1｜System Setup"):
            gr.HTML("<div class='setup-section-title'><h3>① 上傳 YUNFA ERP / MES 並建立 4 個 Tool</h3></div>")

            with gr.Row():
                with gr.Column():
                    gr.HTML(
                        """
                        <div class="upload-instruction">
                          <div class="file-kind">📘 ERP 資料</div>
                          <div class="file-name">請上傳：YUNFA_ERP.xlsx</div>
                          <div class="file-note">客戶、銷售、產品與成本分析資料</div>
                        </div>
                        """
                    )
                    erp_file = gr.File(
                        label="選擇 YUNFA_ERP.xlsx",
                        type="filepath",
                        file_types=[".xlsx"],
                        elem_id="erp-upload",
                    )

                with gr.Column():
                    gr.HTML(
                        """
                        <div class="upload-instruction">
                          <div class="file-kind">🏭 MES 資料</div>
                          <div class="file-name">請上傳：YUNFA_MES.xlsx</div>
                          <div class="file-note">工單、機台、報工與製程異常資料</div>
                        </div>
                        """
                    )
                    mes_file = gr.File(
                        label="選擇 YUNFA_MES.xlsx",
                        type="filepath",
                        file_types=[".xlsx"],
                        elem_id="mes-upload",
                    )

            btn_build = gr.Button("① 初始化 YUNFA ERP / MES", variant="primary")
            ds_status = gr.Textbox(label="Dataset Status", interactive=False, lines=6)
            gr.Markdown("#### Tool Status")
            tool_table = gr.HTML(
                value=df_to_dark_html(pd.DataFrame(), "尚未初始化 ERP / MES")
            )

            btn_build.click(
                fn=initialize_yunfa_tools,
                inputs=[erp_file, mes_file],
                outputs=[ds_status, tool_table],
            )

            gr.Markdown("---")
            gr.Markdown("### ② 載入 / 切換 Granite Planner")

            if torch.cuda.is_available():
                compute_note = (
                    f"<b>目前運算環境：GPU</b> — {torch.cuda.get_device_name(0)}<br>"
                    "350M / 1B / 3B 都可使用；3B 第一次下載與載入仍會較久。"
                )
            else:
                compute_note = (
                    "<b>目前運算環境：CPU</b><br>"
                    "四個 ERP / MES 分析 Tool 可正常使用。"
                    "Granite 350M 可用；1B 可用但推論較慢；"
                    "3B 可能非常慢且有 RAM 不足風險，課堂不建議用 CPU 跑 3B。"
                )

            gr.HTML(
                f"""
                <div class="upload-instruction setup-note">
                  {compute_note}<br>
                  <span style="opacity:.85">
                  CPU 模式的等待時間來自真實的模型載入與推論，不會額外加入人工延遲。
                  </span>
                </div>
                """
            )

            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=list(MODEL_OPTIONS.keys()),
                    value="Granite 4.0 350M（CPU 可用／最快）",
                    label="語言模型",
                )
                btn_model = gr.Button("② 載入 / 切換模型", variant="secondary")

            model_status = gr.Textbox(
                label="Model Loading Status",
                value="尚未載入模型。資料 Tool 可以先獨立初始化。",
                interactive=False,
                lines=6,
            )
            btn_model.click(
                fn=load_granite_model,
                inputs=[model_selector],
                outputs=[model_status],
            )

        with gr.Tab("2｜AI Agent"):
            gr.HTML("<div class='agent-section-title'><h2>AI Agent 智慧分析</h2></div>")
            gr.Markdown("先選擇應用情境，再從該情境的 10 個展示問題中進行詢問。")

            with gr.Tabs():
                with gr.Tab("業務端 AI Agent"):
                    gr.Markdown("### 業務端｜客戶、成本與營運分析")
                    with gr.Row():
                        business_question = gr.Dropdown(
                            choices=BUSINESS_QUESTIONS,
                            value=BUSINESS_QUESTIONS[0],
                            allow_custom_value=True,
                            label="業務端展示問題（B1～B10，可直接修改）",
                            scale=3,
                        )
                        business_model = gr.Dropdown(
                            choices=list(MODEL_OPTIONS.keys()),
                            value="Granite 4.0 350M（CPU 可用／最快）",
                            label="本次使用模型",
                            scale=1,
                        )

                    business_run = gr.Button(
                        "執行業務分析",
                        variant="secondary",
                        elem_id="business-run",
                        elem_classes=["agent-run-button"],
                    )

                    with gr.Row():
                        with gr.Column(scale=2):
                            business_decision = gr.Markdown(elem_classes=["result-summary-card"])
                        with gr.Column(scale=2):
                            business_kpis = gr.HTML()

                    business_strategy = gr.HTML(
                        value="<div class='strategy-panel'><div class='strategy-main-title'>策略建議</div><div class='strategy-card'><div class='strategy-text'>執行分析後，將在此顯示對應的管理策略。</div></div></div>"
                    )

                    with gr.Tab("分析圖表與說明 / Charts & Insights"):
                        with gr.Row():
                            with gr.Column(scale=3):
                                business_gallery = gr.Gallery(
                                    label="業務端分析圖表",
                                    columns=1,
                                    object_fit="contain",
                                    preview=True,
                                )
                            with gr.Column(scale=2):
                                business_chart_notes = gr.HTML(
                                    value="<div class='chart-note-panel'><div class='chart-note-main-title'>圖表重點說明</div><div class='chart-note-card'>執行分析後，將在此顯示每張圖表的判讀重點。</div></div>"
                                )

                    with gr.Tab("分析證據 / Evidence"):
                        business_evidence = gr.HTML(
                            value=df_to_dark_html(pd.DataFrame(), "尚無 Evidence")
                        )

                    business_meta = gr.HTML()

                    business_action = gr.HTML(visible=False)
                    business_planner = gr.Textbox(visible=False)
                    business_trace = gr.Textbox(visible=False)

                    business_run.click(
                        fn=run_ai_agent,
                        inputs=[business_model, business_question],
                        outputs=[
                            business_decision,
                            business_kpis,
                            business_evidence,
                            business_gallery,
                            business_strategy,
                            business_chart_notes,
                            business_action,
                            business_planner,
                            business_trace,
                            business_meta,
                        ],
                    )

                with gr.Tab("生產端 AI Agent"):
                    gr.Markdown("### 生產端｜排程、製程異常、設備與材料分析")
                    with gr.Row():
                        production_question = gr.Dropdown(
                            choices=PRODUCTION_QUESTIONS,
                            value=PRODUCTION_QUESTIONS[0],
                            allow_custom_value=True,
                            label="生產端展示問題（P1～P10，可直接修改）",
                            scale=3,
                        )
                        production_model = gr.Dropdown(
                            choices=list(MODEL_OPTIONS.keys()),
                            value="Granite 4.0 350M（CPU 可用／最快）",
                            label="本次使用模型",
                            scale=1,
                        )

                    production_run = gr.Button(
                        "執行生產分析",
                        variant="secondary",
                        elem_id="production-run",
                        elem_classes=["agent-run-button"],
                    )

                    with gr.Row():
                        with gr.Column(scale=2):
                            production_decision = gr.Markdown(elem_classes=["result-summary-card"])
                        with gr.Column(scale=2):
                            production_kpis = gr.HTML()

                    production_strategy = gr.HTML(
                        value="<div class='strategy-panel'><div class='strategy-main-title'>策略建議</div><div class='strategy-card'><div class='strategy-text'>執行分析後，將在此顯示對應的管理策略。</div></div></div>"
                    )

                    with gr.Tab("分析圖表與說明 / Charts & Insights"):
                        with gr.Row():
                            with gr.Column(scale=3):
                                production_gallery = gr.Gallery(
                                    label="生產端分析圖表",
                                    columns=1,
                                    object_fit="contain",
                                    preview=True,
                                )
                            with gr.Column(scale=2):
                                production_chart_notes = gr.HTML(
                                    value="<div class='chart-note-panel'><div class='chart-note-main-title'>圖表重點說明</div><div class='chart-note-card'>執行分析後，將在此顯示每張圖表的判讀重點。</div></div>"
                                )

                    with gr.Tab("分析證據 / Evidence"):
                        production_evidence = gr.HTML(
                            value=df_to_dark_html(pd.DataFrame(), "尚無 Evidence")
                        )

                    production_meta = gr.HTML()

                    production_action = gr.HTML(visible=False)
                    production_planner = gr.Textbox(visible=False)
                    production_trace = gr.Textbox(visible=False)

                    production_run.click(
                        fn=run_ai_agent,
                        inputs=[production_model, production_question],
                        outputs=[
                            production_decision,
                            production_kpis,
                            production_evidence,
                            production_gallery,
                            production_strategy,
                            production_chart_notes,
                            production_action,
                            production_planner,
                            production_trace,
                            production_meta,
                        ],
                    )

        return demo

demo = create_app()

if __name__ == "__main__":
    print("🚀 啟動 YUNFA AI Agent 戰情室")
    print("如果 Colab 下方直接顯示網頁，即可使用；也可開啟產生的 share link。")
    demo.queue().launch(
        share=True,
        debug=False,
        show_error=True,
    )
