
# ============================================================
# Rule-Based Scheduling
# AI 智慧工單排程與生產優先順序分析
# 排成分析表 + 排程圖
#
# 作者：國立雲林科技大學電機系 林家仁
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

np.random.seed(42)

# ============================================================
# 1. 配色
# ============================================================

COLORS = [
    "#4E79A7","#F28E2B","#59A14F","#E15759","#B07AA1",
    "#76B7B2","#EDC948","#FF9DA7","#9C755F","#BAB0AC",
    "#6B8EAD","#D18B47","#7FA36B","#C96A72","#8A76A8",
    "#5E9C9A","#D6B13F","#E28FA7","#A67855","#9DA3A6"
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
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.style"] = "normal"

def style_table(df_show, header_color=PASTEL_BLUE, formats=None):
    s = (
        df_show.style
        .set_properties(**{
            "color": TEXT_DARK,
            "text-align": "center",
            "font-style": "normal"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", header_color),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("padding", "8px"),
                    ("border", "2px solid white")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("padding", "8px"),
                    ("border", "2px solid white")
                ]
            },
            {
                "selector": "tbody tr:nth-child(odd) td",
                "props": [("background-color", "#FFFFFF")]
            },
            {
                "selector": "tbody tr:nth-child(even) td",
                "props": [("background-color", "#F8FAFC")]
            }
        ])
    )
    if formats:
        s = s.format(formats)
    return s


# ============================================================
# 2. 建立 100 筆待排程工單
# ============================================================

N = 100
N_MACHINES = 8
WORK_HOURS_PER_DAY = 8
AT_RISK_WINDOW = 16
MACHINE_NAMES = [f"Machine {i}" for i in range(1, N_MACHINES + 1)]

df = pd.DataFrame({
    "工單編號": [f"WO-{i+1:04d}" for i in range(N)],
    "客戶等級": np.random.choice(
        ["VIP", "重要客戶", "一般客戶"], N, p=[0.20, 0.30, 0.50]
    ),
    "訂單金額_KNTD": np.clip(
        np.random.normal(850, 350, N), 150, 1800
    ).round(0),
    "距離交期天數": np.clip(
        np.round(np.random.normal(10.0, 3.0, N)), 2, 18
    ).astype(int),
    "是否急單": np.random.choice(
        ["是", "否"], N, p=[0.14, 0.86]
    ),
    "生產工時": np.clip(
        np.random.normal(7.0, 2.5, N), 2.0, 14.0
    ).round(1),
    "換線工時": np.clip(
        np.random.normal(1.8, 0.7, N), 0.5, 4.0
    ).round(1),
    "機台負載率": np.clip(
        np.random.normal(78, 10, N), 50, 98
    ).round(1),
    "材料是否到齊": np.random.choice(
        ["是", "否"], N, p=[0.90, 0.10]
    ),
    "品質風險": np.random.choice(
        ["低", "中", "高"], N, p=[0.70, 0.20, 0.10]
    )
})


# ============================================================
# 3. Rule-Based Priority Score
# ============================================================

def score_due(d):
    if d <= 1: return 30
    if d <= 2: return 25
    if d <= 4: return 15
    if d <= 7: return 5
    return 0

df["急單分數"] = np.where(df["是否急單"] == "是", 30, 0)
df["交期分數"] = df["距離交期天數"].apply(score_due)
df["客戶分數"] = df["客戶等級"].map({
    "VIP": 20, "重要客戶": 10, "一般客戶": 0
})
df["材料分數"] = np.where(df["材料是否到齊"] == "是", 15, -35)
df["換線分數"] = np.select(
    [df["換線工時"] <= 1.5, df["換線工時"] >= 3.5],
    [10, -5],
    default=0
)
df["負載分數"] = np.where(df["機台負載率"] >= 90, -10, 0)
df["品質分數"] = df["品質風險"].map({
    "高": -15, "中": -5, "低": 0
})

score_cols = [
    "急單分數","交期分數","客戶分數","材料分數",
    "換線分數","負載分數","品質分數"
]

df["排程分數"] = df[score_cols].sum(axis=1)

def explain_rule(r):
    x = []
    if r["急單分數"] != 0:
        x.append(f"急單 {r['急單分數']:+d}")
    if r["交期分數"] != 0:
        x.append(f"交期 {r['交期分數']:+d}")
    if r["客戶分數"] != 0:
        x.append(f"{r['客戶等級']} {r['客戶分數']:+d}")
    x.append("材料到齊 +15" if r["材料分數"] > 0 else "材料未到 -35")
    if r["換線分數"] != 0:
        x.append(f"換線 {r['換線分數']:+d}")
    if r["負載分數"] != 0:
        x.append(f"高負載 {r['負載分數']:+d}")
    if r["品質分數"] != 0:
        x.append(f"品質風險 {r['品質分數']:+d}")
    return "；".join(x)

df["規則說明"] = df.apply(explain_rule, axis=1)

df["可立即排產"] = np.where(
    (df["材料是否到齊"] == "是") &
    (df["品質風險"] != "高"),
    "是", "否"
)

priority_df = (
    df.sort_values(
        ["可立即排產", "排程分數", "距離交期天數", "換線工時"],
        ascending=[False, False, True, True]
    )
    .reset_index(drop=True)
)

priority_df["排程順位"] = np.arange(1, len(priority_df) + 1)


# ============================================================
# 4. 真正排到機台與時間軸
# ============================================================

schedulable = priority_df[
    priority_df["可立即排產"] == "是"
].copy().reset_index(drop=True)

available = {m: 0.0 for m in MACHINE_NAMES}
rows = []

for _, r in schedulable.iterrows():
    machine = min(available, key=available.get)
    start = available[machine]
    duration = float(r["換線工時"] + r["生產工時"])
    end = start + duration

    rows.append({
        **r.to_dict(),
        "機台": machine,
        "開始工時": round(start, 1),
        "排程總工時": round(duration, 1),
        "完成工時": round(end, 1)
    })
    available[machine] = end

schedule = pd.DataFrame(rows)

schedule["交期工時"] = (
    schedule["距離交期天數"] * WORK_HOURS_PER_DAY
).astype(float)

schedule["距交期剩餘工時"] = (
    schedule["交期工時"] - schedule["完成工時"]
).round(1)

schedule["延遲工時"] = (
    schedule["完成工時"] - schedule["交期工時"]
).clip(lower=0).round(1)

def delivery_status(r):
    gap = r["距交期剩餘工時"]
    if gap < 0:
        return "Late"
    elif gap <= AT_RISK_WINDOW:
        return "At Risk"
    else:
        return "On Time"

schedule["交期狀態"] = schedule.apply(delivery_status, axis=1)

# ============================================================
# 5. FIFO baseline
# ============================================================

fifo_source = df[
    (df["材料是否到齊"] == "是") &
    (df["品質風險"] != "高")
].copy().reset_index(drop=True)

fifo_available = {m: 0.0 for m in MACHINE_NAMES}
fifo_rows = []

for _, r in fifo_source.iterrows():
    machine = min(fifo_available, key=fifo_available.get)
    start = fifo_available[machine]
    duration = float(r["換線工時"] + r["生產工時"])
    end = start + duration

    fifo_rows.append({
        **r.to_dict(),
        "機台": machine,
        "開始工時": round(start, 1),
        "完成工時": round(end, 1),
        "交期工時": float(r["距離交期天數"] * WORK_HOURS_PER_DAY)
    })
    fifo_available[machine] = end

fifo = pd.DataFrame(fifo_rows)
fifo["延遲工時"] = (
    fifo["完成工時"] - fifo["交期工時"]
).clip(lower=0).round(1)


# ============================================================
# 6. 保留所有分析表
# ============================================================

print("=" * 82)
print("Day 1｜Rule-Based Scheduling－AI 智慧工單排程與生產優先順序分析")
print("=" * 82)
print(f"總工單：{len(df)}｜可立即排產：{len(schedule)}｜機台：{N_MACHINES}")
print()

# 表 1
print("【表 1｜原始待排程工單－前 15 筆】")
display(style_table(
    df.head(15)[[
        "工單編號","客戶等級","訂單金額_KNTD","距離交期天數",
        "是否急單","生產工時","換線工時","機台負載率",
        "材料是否到齊","品質風險"
    ]],
    PASTEL_BLUE,
    {
        "訂單金額_KNTD":"{:,.0f}",
        "生產工時":"{:.1f}",
        "換線工時":"{:.1f}",
        "機台負載率":"{:.1f}%"
    }
))

# 表 2
rules_df = pd.DataFrame({
    "規則":["急單","交期 ≤ 1 天","交期 ≤ 2 天","交期 ≤ 4 天","交期 ≤ 7 天",
          "VIP 客戶","重要客戶","材料到齊","材料未到",
          "換線 ≤ 1.5 小時","換線 ≥ 3.5 小時","機台負載 ≥ 90%",
          "品質風險高","品質風險中"],
    "加扣分":["+30","+30","+25","+15","+5","+20","+10","+15","-35",
           "+10","-5","-10","-15","-5"],
    "管理意義":["提高急單優先度","最高交期壓力","交期即將到期","需提前安排","一般交期提醒",
              "核心客戶優先","重要客戶加權","可立即生產","暫緩等待材料",
              "降低換線損失","避免長換線","避免高負載","先確認品質","適度降低優先度"]
})
print("\n【表 2｜Rule-Based Scheduling 規則】")
display(style_table(rules_df, PASTEL_GREEN))

# 表 3
full_cols = [
    "排程順位","工單編號","客戶等級","距離交期天數","是否急單",
    "材料是否到齊","品質風險","換線工時","機台負載率",
    "排程分數"
]
print("\n【表 3｜完整優先排序－前 30 筆】")
display(style_table(
    priority_df.head(30)[full_cols],
    PASTEL_TEAL,
    {
        "換線工時":"{:.1f}",
        "機台負載率":"{:.1f}%",
        "排程分數":"{:.0f}"
    }
))

# 表 4
print("\n【表 4｜今日優先生產 Top 10】")
display(style_table(
    schedule.head(10)[[
        "排程順位","工單編號","機台","客戶等級",
        "距離交期天數","是否急單","開始工時","完成工時","排程分數"
    ]],
    PASTEL_ORANGE,
    {
        "開始工時":"{:.1f}",
        "完成工時":"{:.1f}",
        "排程分數":"{:.0f}"
    }
))

# 表 5
print("\n【表 5｜Top 10 排程分數來源拆解】")
display(style_table(
    priority_df.head(10)[[
        "排程順位","工單編號","急單分數","交期分數","客戶分數",
        "材料分數","換線分數","負載分數","品質分數","排程分數"
    ]],
    PASTEL_YELLOW,
    {
        "急單分數":"{:+.0f}",
        "交期分數":"{:+.0f}",
        "客戶分數":"{:+.0f}",
        "材料分數":"{:+.0f}",
        "換線分數":"{:+.0f}",
        "負載分數":"{:+.0f}",
        "品質分數":"{:+.0f}",
        "排程分數":"{:.0f}"
    }
))

# 表 6
print("\n【表 6｜Top 10 排程原因說明】")
display(style_table(
    priority_df.head(10)[[
        "排程順位","工單編號","排程分數","規則說明"
    ]],
    PASTEL_PINK,
    {"排程分數":"{:.0f}"}
))

# 表 7
urgent_df = priority_df[priority_df["是否急單"] == "是"].copy()
print("\n【表 7｜急單分析】")
display(style_table(
    urgent_df[[
        "排程順位","工單編號","客戶等級","距離交期天數",
        "材料是否到齊","品質風險","排程分數"
    ]],
    PASTEL_RED,
    {"排程分數":"{:.0f}"}
))

# 表 8
due_risk_df = priority_df[priority_df["距離交期天數"] <= 2].copy()
print("\n【表 8｜交期 ≤ 2 天風險工單】")
display(style_table(
    due_risk_df[[
        "排程順位","工單編號","客戶等級","距離交期天數",
        "是否急單","材料是否到齊","品質風險","排程分數"
    ]],
    PASTEL_ORANGE,
    {"排程分數":"{:.0f}"}
))

# 表 9
material_hold_df = priority_df[
    priority_df["材料是否到齊"] == "否"
].copy()
material_hold_df["材料到齊後預估分數"] = (
    material_hold_df["排程分數"] + 50
)
print("\n【表 9｜待料工單與材料到齊後潛在分數】")
display(style_table(
    material_hold_df[[
        "工單編號","客戶等級","距離交期天數","是否急單",
        "排程分數","材料到齊後預估分數"
    ]],
    PASTEL_BROWN,
    {
        "排程分數":"{:.0f}",
        "材料到齊後預估分數":"{:.0f}"
    }
))

# 表 10
quality_df = priority_df[priority_df["品質風險"] == "高"].copy()
print("\n【表 10｜高品質風險工單】")
display(style_table(
    quality_df[[
        "排程順位","工單編號","客戶等級","距離交期天數",
        "是否急單","排程分數"
    ]],
    PASTEL_PURPLE,
    {"排程分數":"{:.0f}"}
))

# 表 11
high_load_df = priority_df[
    priority_df["機台負載率"] >= 90
].copy()
print("\n【表 11｜機台高負載工單】")
display(style_table(
    high_load_df[[
        "排程順位","工單編號","機台負載率",
        "距離交期天數","是否急單","排程分數"
    ]],
    PASTEL_GRAY,
    {
        "機台負載率":"{:.1f}%",
        "排程分數":"{:.0f}"
    }
))

# 表 12
vip_df = priority_df[priority_df["客戶等級"] == "VIP"].copy()
print("\n【表 12｜VIP 客戶工單】")
display(style_table(
    vip_df[[
        "排程順位","工單編號","距離交期天數",
        "是否急單","材料是否到齊","品質風險","排程分數"
    ]],
    PASTEL_BLUE,
    {"排程分數":"{:.0f}"}
))

# 表 13
delivery_count = schedule["交期狀態"].value_counts().reindex(
    ["On Time","At Risk","Late"], fill_value=0
)
risk_summary = pd.DataFrame({
    "分析項目":[
        "總工單","可立即排產","待料工單","高品質風險",
        "急單","交期 2 天內","預計準時","交期風險","預計延遲"
    ],
    "工單數":[
        len(df),len(schedule),
        (df["材料是否到齊"]=="否").sum(),
        (df["品質風險"]=="高").sum(),
        (df["是否急單"]=="是").sum(),
        (df["距離交期天數"]<=2).sum(),
        delivery_count["On Time"],
        delivery_count["At Risk"],
        delivery_count["Late"]
    ],
    "管理意義":[
        "本次待排程總數","目前可投入生產","材料未到先暫緩","生產前先確認品質",
        "需提高優先度","近期交期壓力","預計準時完成","接近交期需注意","依目前排程可能延遲"
    ]
})
print("\n【表 13｜排程管理與風險摘要】")
display(style_table(risk_summary, PASTEL_GREEN))


# ============================================================
# 7. 保留並更新所有重要圖
# ============================================================

# 圖 1 Top 15
p = priority_df.head(15)
labels = [x.replace("WO-","WO\n") for x in p["工單編號"]]
fig, ax = plt.subplots(figsize=(13,6.1))
bars = ax.bar(np.arange(len(p)), p["排程分數"], width=0.52, color=COLORS[:len(p)])
ax.set_xticks(np.arange(len(p)))
ax.set_xticklabels(labels, rotation=0)
ax.set_title("Top 15 Work Orders by Priority Score")
ax.set_xlabel("Work Order")
ax.set_ylabel("Priority Score")
ax.grid(axis="y", alpha=0.22)
for b,v in zip(bars,p["排程分數"]):
    ax.text(b.get_x()+b.get_width()/2, v+1, f"{int(v)}", ha="center", fontsize=9)
plt.tight_layout()
plt.show()

# 圖 2 完整 Gantt
machine_y = {m:i for i,m in enumerate(MACHINE_NAMES)}
fig, ax = plt.subplots(figsize=(13,6.1))
for i,r in schedule.iterrows():
    y = machine_y[r["機台"]]
    ax.barh(y, r["排程總工時"], left=r["開始工時"],
            height=0.60, color=COLORS[i%len(COLORS)], edgecolor="white")
    if r["排程總工時"] >= 5:
        ax.text(r["開始工時"]+r["排程總工時"]/2, y,
                r["工單編號"].replace("WO-",""),
                ha="center", va="center", fontsize=7)
ax.set_yticks(range(len(MACHINE_NAMES)))
ax.set_yticklabels(MACHINE_NAMES)
ax.set_xlabel("Scheduled Production Hours")
ax.set_ylabel("Machine")
ax.set_title("Rule-Based Production Scheduling Gantt Chart")
ax.grid(axis="x", alpha=0.20)
for h in np.arange(8, schedule["完成工時"].max()+8, 8):
    ax.axvline(h, linewidth=0.8, linestyle="--", alpha=0.25)
plt.tight_layout()
plt.show()

# 圖 3 前 5 日
H = 5 * WORK_HOURS_PER_DAY
first = schedule[schedule["開始工時"] < H].copy()
fig, ax = plt.subplots(figsize=(13,6.1))
for i,r in first.iterrows():
    y = machine_y[r["機台"]]
    left = r["開始工時"]
    width = min(r["排程總工時"], H-left)
    if width <= 0:
        continue
    ax.barh(y, width, left=left, height=0.60,
            color=COLORS[i%len(COLORS)], edgecolor="white")
    if width >= 3.5:
        ax.text(left+width/2, y, r["工單編號"].replace("WO-",""),
                ha="center", va="center", fontsize=7)
ax.set_yticks(range(len(MACHINE_NAMES)))
ax.set_yticklabels(MACHINE_NAMES)
ticks = np.arange(0, H+1, 8)
ax.set_xticks(ticks)
ax.set_xticklabels([f"Day {int(h/8)+1}" if h < H else "" for h in ticks])
ax.set_xlim(0,H)
ax.set_xlabel("Production Day")
ax.set_ylabel("Machine")
ax.set_title("Five-Day Production Schedule")
ax.grid(axis="x", alpha=0.22)
plt.tight_layout()
plt.show()

# 圖 4 利用率
machine_hours = schedule.groupby("機台")["排程總工時"].sum().reindex(MACHINE_NAMES)
makespan = schedule["完成工時"].max()
util = (machine_hours / makespan * 100).round(1)
fig, ax = plt.subplots(figsize=(13,6.1))
bars = ax.bar(MACHINE_NAMES, util.values, width=0.52, color=COLORS[:len(MACHINE_NAMES)])
ax.set_title("Machine Utilization after Rule-Based Scheduling")
ax.set_ylabel("Utilization (%)")
ax.set_ylim(0,105)
ax.grid(axis="y", alpha=0.22)
for b,v in zip(bars, util.values):
    ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.show()

# 圖 5 Scheduled Completion vs Due Date
status_colors = {"On Time":"#59A14F","At Risk":"#F28E2B","Late":"#E15759"}
fig, ax = plt.subplots(figsize=(13,6.1))
for st in ["On Time","At Risk","Late"]:
    sub = schedule[schedule["交期狀態"] == st]
    ax.scatter(
        sub["交期工時"], sub["完成工時"],
        s=62, alpha=0.78,
        label=f"{st} ({len(sub)})",
        color=status_colors[st]
    )
mx = max(schedule["交期工時"].max(), schedule["完成工時"].max())
ax.plot([0,mx],[0,mx], linestyle="--", linewidth=1.5,
        color="#666666", label="Due-Date Boundary")
ax.set_title("Scheduled Completion vs Due Date - Updated")
ax.set_xlabel("Due Date (Production Hours)")
ax.set_ylabel("Scheduled Completion (Production Hours)")
ax.grid(alpha=0.22)
ax.legend()
plt.tight_layout()
plt.show()

# 圖 6 Delivery Risk
fig, ax = plt.subplots(figsize=(13,6.1))
bars = ax.bar(
    ["On Time","At Risk","Late"],
    delivery_count.values,
    width=0.52,
    color=[status_colors["On Time"],status_colors["At Risk"],status_colors["Late"]]
)
ax.set_title("Updated Delivery Risk Distribution")
ax.set_ylabel("Number of Work Orders")
ax.grid(axis="y", alpha=0.22)
for b,st in zip(bars,["On Time","At Risk","Late"]):
    v = delivery_count[st]
    pct = v / len(schedule) * 100
    ax.text(b.get_x()+b.get_width()/2, v+0.5,
            f"{v}\n({pct:.1f}%)", ha="center")
plt.tight_layout()
plt.show()

# 圖 7 Before vs After
def metrics(s):
    late = s["延遲工時"] > 0
    urgent = s["是否急單"] == "是"
    return {
        "Late Orders": late.sum(),
        "Avg Delay": s.loc[late,"延遲工時"].mean() if late.sum()>0 else 0,
        "On-Time Rate": (s["延遲工時"]<=0).mean()*100,
        "Urgent On-Time": (
            (s.loc[urgent,"延遲工時"]<=0).mean()*100
            if urgent.sum()>0 else 0
        )
    }

before = metrics(fifo)
after = metrics(schedule)

comp = pd.DataFrame({
    "Metric":[
        "Late\nOrders","Avg Delay\nHours",
        "On-Time\nRate (%)","Urgent On-Time\nRate (%)"
    ],
    "FIFO":[
        before["Late Orders"],before["Avg Delay"],
        before["On-Time Rate"],before["Urgent On-Time"]
    ],
    "Rule-Based":[
        after["Late Orders"],after["Avg Delay"],
        after["On-Time Rate"],after["Urgent On-Time"]
    ]
})

x = np.arange(len(comp))
w = 0.34
fig, ax = plt.subplots(figsize=(13,6.1))
b1 = ax.bar(x-w/2, comp["FIFO"], width=w, label="FIFO", color="#BAB0AC")
b2 = ax.bar(x+w/2, comp["Rule-Based"], width=w, label="Rule-Based", color="#4E79A7")
ax.set_xticks(x)
ax.set_xticklabels(comp["Metric"], rotation=0)
ax.set_title("Before vs After Rule-Based Scheduling")
ax.set_ylabel("Metric Value")
ax.grid(axis="y", alpha=0.22)
ax.legend()
for bars in [b1,b2]:
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, v+0.8, f"{v:.1f}",
                ha="center", fontsize=9)
plt.tight_layout()
plt.show()

# 圖 8 Day 1 Dispatch Board
dispatch = schedule[schedule["開始工時"] < 8].copy()
fig, ax = plt.subplots(figsize=(13,6.1))
for i,r in dispatch.iterrows():
    y = machine_y[r["機台"]]
    width = min(r["排程總工時"], 8-r["開始工時"])
    if width <= 0:
        continue
    ax.barh(y, width, left=r["開始工時"], height=0.58,
            color=COLORS[i%len(COLORS)], edgecolor="white")
    ax.text(r["開始工時"]+width/2, y,
            r["工單編號"].replace("WO-",""),
            ha="center", va="center", fontsize=8)
ax.set_yticks(range(len(MACHINE_NAMES)))
ax.set_yticklabels(MACHINE_NAMES)
ax.set_xlim(0,8)
ax.set_xticks(range(0,9))
ax.set_xlabel("Production Hour")
ax.set_ylabel("Machine")
ax.set_title("Day 1 Production Dispatch Board")
ax.grid(axis="x", alpha=0.22)
plt.tight_layout()
plt.show()


# ============================================================
# 8. 摘要與輸出
# ============================================================

print()
print("=" * 82)
print("Rule-Based Scheduling 分析摘要")
print("=" * 82)
print(f"總工單數：{len(df)}")
print(f"可排產工單：{len(schedule)}")
print(f"機台數量：{N_MACHINES}")
print(f"總排程工期：約 {makespan/8:.1f} 個生產日")
print(f"Rule-Based 預計準時率：{after['On-Time Rate']:.1f}%")
print(f"急單準時率：{after['Urgent On-Time']:.1f}%")
print(f"預計延遲工單：{after['Late Orders']} 筆")

priority_df.to_csv(
    "RuleBasedScheduling_100工單_完整優先排序.csv",
    index=False,
    encoding="utf-8-sig"
)

schedule.to_csv(
    "RuleBasedScheduling_更新後實際生產排程.csv",
    index=False,
    encoding="utf-8-sig"
)

print()
print("已輸出：RuleBasedScheduling_100工單_完整優先排序.csv")
print("已輸出：RuleBasedScheduling_更新後實際生產排程.csv")
