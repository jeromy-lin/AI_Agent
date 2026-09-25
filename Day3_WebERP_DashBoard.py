# WebBased AI 數據戰情室｜整段貼入 Google Colab 的一個程式碼儲存格
# 作者：國立雲林科技大學電機工程系 林家仁
import sys, subprocess, importlib.util
for mod, pkg in [("numpy","numpy"),("pandas","pandas"),("openpyxl","openpyxl"),
                 ("matplotlib","matplotlib"),("sklearn","scikit-learn"),
                 ("PIL","pillow"),("gradio","gradio>=6,<7"),("socksio","socksio")]:
    if importlib.util.find_spec(mod) is None:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])

import os, io, time, tempfile
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
from html import escape
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

RAW = {"CRM_Account_Export":{"AccountNo","AccountName"},
       "QUOTE_LOG":{"QuoteNo","CustCode","QuoteCreated","DiscPct","EstMarginPct","Outcome"},
       "SD_SalesHistory":{"SalesDoc","SoldTo","PostingDate","OrderQty","NetUnitPrice","Currency"},
       "MM_ItemMaster":{"ItemNo","BusinessFamily","StdMaterialCost","StdLaborCost","StdSubcontractCost","StdFactoryOH"},
       "CO_ActualCost":{"ManufacturingOrder","Item","CompletedQty","CncHours","AssemblyHours","InspectionHours","ChangeoverHours","ActualMfgCost"}}
ERP = {"客戶主檔","報價紀錄","銷售訂單","產品主檔","成本結算"}
NEEDED = {"客戶主檔":{"客戶名稱"},
          "報價紀錄":{"客戶ID","報價ID","報價日期","折扣率","預估毛利率","結果"},
          "銷售訂單":{"客戶ID","訂單ID","下單日期","訂購數量","成交單價_TWD","幣別"},
          "產品主檔":{"SKU","產品線","標準材料成本_TWD","標準人工成本_TWD","標準委外成本_TWD","標準製造費_TWD"},
          "成本結算":{"工單ID","SKU","完工數量","CNC加工工時_hr","組裝工時_hr","測試工時_hr","換線工時_hr","實際製造成本_TWD"}}
CACHE = {}

def kind(path):
    sheets=set(pd.ExcelFile(path).sheet_names)
    if set(RAW)<=sheets: return "原始英文檔"
    if ERP<=sheets: return "中文 ERP 檔"
    raise ValueError("無法辨識工作表；收到："+"、".join(sorted(sheets)))

def validate(path):
    x=pd.ExcelFile(path)
    if not ERP<=set(x.sheet_names): raise ValueError("缺少工作表："+"、".join(sorted(ERP-set(x.sheet_names))))
    for sheet,need in NEEDED.items():
        have=set(pd.read_excel(x,sheet,nrows=0).columns)
        if sheet=="客戶主檔" and not {"客戶ID","Customer_ID"}&have:
            raise ValueError("客戶主檔缺少 客戶ID 或 Customer_ID")
        if need-have: raise ValueError(f"{sheet} 缺少欄位："+"、".join(sorted(need-have)))

def prepare_raw(path):
    x=pd.ExcelFile(path); src={}
    for sheet,need in RAW.items():
        d=pd.read_excel(x,sheet); missing=need-set(d.columns)
        if missing: raise ValueError(f"{sheet} 缺少欄位："+"、".join(sorted(missing)))
        src[sheet]=d
    c=src["CRM_Account_Export"].dropna(subset=["AccountNo"])
    p=src["MM_ItemMaster"].dropna(subset=["ItemNo"])
    if c.AccountNo.duplicated().any() or p.ItemNo.duplicated().any():
        raise ValueError("來源客戶／產品主鍵重複，請先清理。")
    q=src["QUOTE_LOG"].dropna(subset=["QuoteNo","CustCode"])
    s=src["SD_SalesHistory"].dropna(subset=["SalesDoc","SoldTo"])
    t=src["CO_ActualCost"].dropna(subset=["ManufacturingOrder","Item"])
    q=q[q.CustCode.isin(c.AccountNo)].copy()
    s=s[s.SoldTo.isin(c.AccountNo)].copy()
    t=t[t.Item.isin(p.ItemNo)].copy()
    tables={
        "客戶主檔":c.rename(columns={"AccountNo":"客戶ID","AccountName":"客戶名稱",
            "CountryCode":"國別","SegmentLevel":"客戶等級","CreatedOn":"建立日期",
            "IndustryGroup":"產業別","AccountStatus":"客戶狀態"}),
        "報價紀錄":q.rename(columns={"QuoteNo":"報價ID","CustCode":"客戶ID",
            "MaterialCode":"SKU","QuoteCreated":"報價日期","DiscPct":"折扣率",
            "EstMarginPct":"預估毛利率","Outcome":"結果","Currency":"幣別"}),
        "銷售訂單":s.rename(columns={"SalesDoc":"訂單ID","SoldTo":"客戶ID",
            "PartNumber":"SKU","PostingDate":"下單日期","OrderQty":"訂購數量",
            "NetUnitPrice":"原幣成交單價","LineAmount":"原幣訂單金額",
            "Currency":"幣別","DocStatus":"單據狀態"}),
        "產品主檔":p.rename(columns={"ItemNo":"SKU","ItemDescription":"產品名稱",
            "BusinessFamily":"產品線","StdMaterialCost":"標準材料成本_TWD",
            "StdLaborCost":"標準人工成本_TWD","StdSubcontractCost":"標準委外成本_TWD",
            "StdFactoryOH":"標準製造費_TWD"}),
        "成本結算":t.rename(columns={"ManufacturingOrder":"工單ID","Item":"SKU",
            "CompletedQty":"完工數量","CncHours":"CNC加工工時_hr",
            "AssemblyHours":"組裝工時_hr","InspectionHours":"測試工時_hr",
            "ChangeoverHours":"換線工時_hr","ActualMfgCost":"實際製造成本_TWD"})}
    tables["報價紀錄"]["結果"]=tables["報價紀錄"]["結果"].replace({"Won":"成交","Lost":"未成交","Pending":"待確認"})
    sales=tables["銷售訂單"]
    sales["幣別"]=sales["幣別"].astype("string").str.upper().str.strip()
    sales["成交單價_TWD"]=pd.to_numeric(sales["原幣成交單價"],errors="coerce").where(sales["幣別"].eq("TWD").fillna(False))
    tables["資料說明"]=pd.DataFrame([
        ["來源","由英文原始匯出之客戶、報價、銷售、產品、成本五表逐列整理；不建立不存在的工單／訂單連結。"],
        ["匯率","未提供 USD/TWD 匯率；USD 訂單保留原幣，TWD 金額欄留空，不納入台幣採購額。"],
        ["成本幣別","英文來源的標準成本與 ActualMfgCost 無幣別欄，教學暫按 TWD 顯示；企業實務需核對幣別。"],
        ["模型","K-Means 四群；線性回歸按工單切分訓練及測試。"]],columns=["項目","說明"])
    return tables

def save_erp(tables,path):
    from openpyxl.styles import PatternFill,Font
    with pd.ExcelWriter(path,engine="openpyxl") as w:
        for sheet,df in tables.items():
            df.to_excel(w,sheet_name=sheet,index=False)
            ws=w.sheets[sheet]; ws.freeze_panes="A2"; ws.auto_filter.ref=ws.dimensions
            for cell in ws[1]:
                cell.fill=PatternFill("solid",fgColor="5B8F88")
                cell.font=Font(color="FFFFFF",bold=True)
    validate(path)

def path_of(value):
    if isinstance(value,(str,os.PathLike)): return os.fspath(value)
    if isinstance(value,dict): return value.get("path") or value.get("name")
    return getattr(value,"name",None)

def resolved(value):
    path=path_of(value)
    if not path: raise ValueError("請上傳 Excel。")
    path=os.path.abspath(path)
    if kind(path)=="中文 ERP 檔": validate(path); return path
    st=os.stat(path); key=(path,st.st_size,st.st_mtime_ns)
    if key not in CACHE:
        fd,target=tempfile.mkstemp(prefix="YUNFA_ERP_本次整理_",suffix=".xlsx"); os.close(fd)
        try: save_erp(prepare_raw(path),target)
        except Exception:
            os.unlink(target); raise
        CACHE[key]=target
    return CACHE[key]

FEATURES=["年化採購金額_TWD","訂單次數","平均訂單金額_TWD","報價次數",
          "報價成交率","平均折扣率","平均預估毛利率","最近交易天數"]
LABELS=["高價值活躍型","成長潛力型","價格敏感型","沉睡／流失風險型"]
EN=["High-Value Active","Growth Potential","Price Sensitive","Dormant / Churn Risk"]
COLORS=["#F2C6A0","#A8C69F","#AFC8E6","#D7C4E8"]
STRATEGY={LABELS[0]:"優先服務與交叉銷售",LABELS[1]:"提高成交率與產品組合",
          LABELS[2]:"檢查折扣與毛利",LABELS[3]:"安排回訪並確認近期需求"}

def customer_model(path):
    x=pd.ExcelFile(path)
    c=pd.read_excel(x,"客戶主檔").rename(columns={"Customer_ID":"客戶ID"}).dropna(subset=["客戶ID"]).drop_duplicates("客戶ID")
    q=pd.read_excel(x,"報價紀錄").dropna(subset=["客戶ID"])
    o=pd.read_excel(x,"銷售訂單").dropna(subset=["客戶ID"])
    q["報價日期"]=pd.to_datetime(q["報價日期"],errors="coerce")
    o["下單日期"]=pd.to_datetime(o["下單日期"],errors="coerce")
    for col in ["折扣率","預估毛利率"]: q[col]=pd.to_numeric(q[col],errors="coerce")
    for col in ["訂購數量","成交單價_TWD"]: o[col]=pd.to_numeric(o[col],errors="coerce")
    dates=pd.concat([q["報價日期"],o["下單日期"]]).dropna()
    ref=dates.max() if len(dates) else pd.Timestamp.today().normalize()
    first=o["下單日期"].min()
    years=max((ref-first).days/365.25,1) if pd.notna(first) else 1
    qs=q.groupby("客戶ID").agg(報價次數=("報價ID","count"),
        成交報價次數=("結果",lambda v:(v=="成交").sum()),平均折扣率=("折扣率","mean"),
        平均預估毛利率=("預估毛利率","mean"))
    qs["報價成交率"]=qs["成交報價次數"]/qs["報價次數"].replace(0,np.nan)
    twd=o[o["幣別"].astype("string").str.upper().str.strip().eq("TWD").fillna(False)].copy()
    twd["訂單金額_TWD"]=twd["訂購數量"]*twd["成交單價_TWD"]
    osum=twd.groupby("客戶ID").agg(總採購金額_TWD=("訂單金額_TWD","sum"),
        訂單次數=("訂單ID","count"),平均訂單金額_TWD=("訂單金額_TWD","mean"),最近交易日=("下單日期","max"))
    osum["年化採購金額_TWD"]=osum["總採購金額_TWD"]/years
    osum["最近交易天數"]=(ref-osum["最近交易日"]).dt.days
    d=c.merge(qs.reset_index(),on="客戶ID",how="left").merge(osum.reset_index(),on="客戶ID",how="left")
    d=d.rename(columns={"客戶等級":"既有客戶等級（ERP）"})
    d[FEATURES]=d[FEATURES].apply(pd.to_numeric,errors="coerce")
    d["最近交易天數"]=d["最近交易天數"].fillna((ref-first).days+1 if pd.notna(first) else 366)
    d[FEATURES]=d[FEATURES].fillna(0)
    if len(d)<5 or d[FEATURES].drop_duplicates().shape[0]<4:
        raise ValueError("至少五名客戶且具有四種不同交易行為。")
    z=StandardScaler().fit_transform(d[FEATURES]); km=KMeans(n_clusters=4,random_state=42,n_init=20)
    d["Cluster"]=km.fit_predict(z)
    elbow=[(i,KMeans(n_clusters=i,random_state=42,n_init=10).fit(z).inertia_)
           for i in range(1,min(8,len(d)-1)+1)]
    ctr=pd.DataFrame(km.cluster_centers_,columns=FEATURES); remaining=set(range(4)); names={}
    risk=(ctr["最近交易天數"]-.5*ctr["年化採購金額_TWD"]-.5*ctr["訂單次數"]).idxmax()
    names[risk]=LABELS[3];remaining.remove(risk)
    vip=(ctr["年化採購金額_TWD"]+ctr["訂單次數"]+ctr["報價次數"]-ctr["最近交易天數"]).loc[list(remaining)].idxmax()
    names[vip]=LABELS[0];remaining.remove(vip)
    sensitive=(ctr["平均折扣率"]-ctr["平均預估毛利率"]).loc[list(remaining)].idxmax()
    names[sensitive]=LABELS[2];remaining.remove(sensitive);names[remaining.pop()]=LABELS[1]
    d["K-Means行為分群"]=d["Cluster"].map(names)
    d["建議策略"]=d["K-Means行為分群"].map(STRATEGY)
    d[["PC1","PC2"]]=PCA(n_components=2).fit_transform(z)
    return d,elbow

CF=["完工數量","標準材料總成本","標準人工總成本","標準委外總成本",
    "標準製造費總額","CNC加工工時_hr","組裝工時_hr","測試工時_hr","換線工時_hr"]
def cost_data(path):
    x=pd.ExcelFile(path); p=pd.read_excel(x,"產品主檔"); c=pd.read_excel(x,"成本結算")
    pc=["SKU","產品線","標準材料成本_TWD","標準人工成本_TWD","標準委外成本_TWD","標準製造費_TWD"]
    cc=["工單ID","SKU","完工數量","CNC加工工時_hr","組裝工時_hr","測試工時_hr","換線工時_hr","實際製造成本_TWD"]
    d=c[cc].dropna(subset=["工單ID","SKU"]).merge(p[pc].dropna(subset=["SKU"]).drop_duplicates("SKU"),on="SKU")
    for col in set(pc+cc)-{"SKU","產品線","工單ID"}: d[col]=pd.to_numeric(d[col],errors="coerce")
    for name,source in [("標準材料總成本","標準材料成本_TWD"),("標準人工總成本","標準人工成本_TWD"),
                        ("標準委外總成本","標準委外成本_TWD"),("標準製造費總額","標準製造費_TWD")]:
        d[name]=d["完工數量"]*d[source]
    return d

def cost_model(d,line):
    s=d[d["產品線"]==line].dropna(subset=["工單ID","實際製造成本_TWD"]).copy()
    if len(s)<8 or s["工單ID"].nunique()<8: raise ValueError(f"{line} 至少需八筆有效且不同的工單。")
    idx1,idx2=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42).split(s,groups=s["工單ID"]))
    tr=s.iloc[idx1];te=s.iloc[idx2]
    fill=tr[CF].median(numeric_only=True).reindex(CF).fillna(0)
    scaler=StandardScaler(); train=scaler.fit_transform(tr[CF].fillna(fill));test=scaler.transform(te[CF].fillna(fill))
    model=LinearRegression().fit(train,tr["實際製造成本_TWD"])
    actual=te["實際製造成本_TWD"].to_numpy(); pred=model.predict(test)
    result=te[["工單ID","SKU","產品線"]].copy()
    result["實際製造成本"]=actual;result["預測製造成本"]=pred
    result["誤差率(%)"]=np.abs(actual-pred)/np.maximum(np.abs(actual),1)*100
    result["是否超出±3%"] = np.where(result["誤差率(%)"]>3,"是","否")
    score={"R²":r2_score(actual,pred),"MAE (KNTD)":mean_absolute_error(actual,pred)/1000,
           "RMSE (KNTD)":np.sqrt(mean_squared_error(actual,pred))/1000,
           "MAPE(%)":np.mean(np.abs(actual-pred)/np.maximum(np.abs(actual),1))*100,
           "±3%內比例(%)":np.mean(result["誤差率(%)"]<=3)*100}
    coef=pd.Series(model.coef_,index=CF).sort_values(key=abs,ascending=False)
    return result.sort_values("誤差率(%)",ascending=False),coef,score

def finish(fig):
    fig.tight_layout();buf=io.BytesIO();fig.savefig(buf,format="png",dpi=135,bbox_inches="tight",facecolor="white")
    buf.seek(0);img=Image.open(buf).convert("RGB").copy();plt.close(fig);return img
def axes(title,xlabel,ylabel):
    fig,ax=plt.subplots(figsize=(13,6.1),facecolor="white")
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.25)
    return fig,ax
def charts_customer(d,elbow):
    output=[]
    fig,ax=axes("ERP Customer Segmentation - K-Means","Principal Component 1","Principal Component 2")
    for name,en,color in zip(LABELS,EN,COLORS):
        part=d[d["K-Means行為分群"]==name]
        ax.scatter(part.PC1,part.PC2,label=en,color=color,s=85,edgecolor="white")
    ax.legend(frameon=False);output.append((finish(fig),"PCA 客戶分群位置：越近代表行為越相似。"))
    count=d["K-Means行為分群"].value_counts().reindex(LABELS).fillna(0)
    fig,ax=axes("Customer Segment Distribution","Customer Segment","Number of Customers")
    bars=ax.bar(EN,count.values,color=COLORS)
    for b,n in zip(bars,count): ax.text(b.get_x()+b.get_width()/2,n+.15,str(int(n)),ha="center")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    output.append((finish(fig),"各群人數可協助安排業務回訪與經營資源。"))
    z=pd.DataFrame(StandardScaler().fit_transform(d[FEATURES]),columns=FEATURES,index=d.index)
    z["群"]=d["K-Means行為分群"]
    profile=z.groupby("群")[FEATURES].mean().reindex(LABELS)
    fen=["Annual Purchase","Order Frequency","Avg. Order","Quote Count","Conversion","Discount","Gross Margin","Recency"]
    fig,ax=axes("Customer Segment Profile","Feature","Standardized Cluster Mean")
    xx=np.arange(len(FEATURES));width=.18
    for i,(name,en,color) in enumerate(zip(LABELS,EN,COLORS)):
        ax.bar(xx+(i-1.5)*width,profile.loc[name].values,width,label=en,color=color)
    ax.set_xticks(xx,fen,fontsize=9);ax.axhline(0,color="gray",linestyle="--");ax.legend(ncol=2,frameon=False)
    output.append((finish(fig),"比較八項行為特徵與全體平均的標準化差距。"))
    fig,ax=axes("Elbow Method for Customer Segmentation","Number of Clusters (K)","Within-Cluster SSE")
    ks,ss=zip(*elbow);ax.plot(ks,ss,marker="o",color="#698F85")
    ax.scatter([4],[dict(elbow)[4]],color="#D88B63",s=130,label="Teaching setting: K=4")
    ax.set_xticks(ks);ax.legend(frameon=False)
    output.append((finish(fig),"SSE 轉折可作為群數參考；四群是本課程示範設定。"))
    return output

def charts_cost(result,coef,line):
    output=[];a=result["實際製造成本"]/1000;p=result["預測製造成本"]/1000
    within=result["誤差率(%)"]<=3
    line_en={"汽車精密零組件":"Automotive Precision Parts","精密金屬加工件":"Precision Metal Components",
             "機電／自動化模組":"Mechatronics & Automation Modules","半導體設備零組件":"Semiconductor Equipment Parts"}.get(line,line)
    fig,ax=axes(f"Actual vs Predicted Cost - {line_en}","Actual Cost (KNTD)","Predicted Cost (KNTD)")
    ax.scatter(a[within],p[within],color="#F6B26B",label="Within ±3%",s=65)
    ax.scatter(a[~within],p[~within],color="#D9534F",label="Outside ±3%",s=65)
    low=min(a.min(),p.min());high=max(a.max(),p.max());x=np.linspace(low,high,100)
    ax.plot(x,x,"--",color="#6FA8DC",label="Perfect Prediction")
    ax.fill_between(x,x*.97,x*1.03,color="#93C47D",alpha=.2,label="±3% Range")
    ax.legend(frameon=False);output.append((finish(fig),"測試集實際與預測成本；顏色標示 ±3% 誤差範圍。"))
    en={"完工數量":"Completed Quantity","標準材料總成本":"Standard Material Cost",
        "標準人工總成本":"Standard Labor Cost","標準委外總成本":"Standard Outsourcing Cost",
        "標準製造費總額":"Standard Factory Overhead","CNC加工工時_hr":"CNC Hours",
        "組裝工時_hr":"Assembly Hours","測試工時_hr":"Inspection Hours","換線工時_hr":"Setup Hours"}
    fig,ax=axes(f"Manufacturing Cost Drivers - {line_en}","Standardized Coefficient (KNTD)","Feature")
    items=coef.iloc[::-1];ax.barh([en[k] for k in items.index],items.values/1000,color="#A8C69F")
    output.append((finish(fig),"係數為標準化後的相關性，不代表因果效果。"))
    fig,ax=axes(f"Prediction Error Distribution - {line_en}","Absolute Error Rate (%)","Work Orders")
    ax.hist(result["誤差率(%)"],bins=15,color="#B7D7C4",edgecolor="white")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    output.append((finish(fig),"檢視測試工單的絕對誤差率與離群案例。"))
    return output

CSS="""
.gradio-container{background:#fffdf9!important;color:#243338!important}
.hero{background:#eff6f2;border:1px solid #d7e8dc;border-radius:18px;padding:20px 26px;margin-bottom:16px}
.hero h1{margin:0;color:#315c50;font-size:30px}.hero p{margin:7px 0 0;color:#526960}
.method{background:#fff;border:1px solid #dde8dd;border-radius:14px;padding:15px;min-height:117px}
.method b{color:#315c50}.method p{margin:8px 0 0;line-height:1.65;color:#475569}
.kpis{display:flex;gap:10px;flex-wrap:wrap}.kpi{background:#f3f8f4;padding:12px 16px;border:1px solid #dce9df;border-radius:12px;min-width:125px}
.kpi strong{display:block;color:#315c50;font-size:21px}.kpi span{color:#65766f;font-size:13px}
.note{background:#f5f9f6;border:1px solid #e2e9e4;padding:14px;border-radius:12px;line-height:1.7;color:#34483e}
.tbl{overflow:auto;max-height:440px;border:1px solid #e0e6df;border-radius:10px}
.tbl table{border-collapse:collapse;width:100%;font-size:13px}.tbl th{background:#eaf3ed;white-space:nowrap}
.tbl td,.tbl th{padding:8px;border-bottom:1px solid #e9e9e9;text-align:left;white-space:nowrap}
"""
def cards(values):
    return '<div class="kpis">'+''.join(f'<div class="kpi"><span>{escape(str(k))}</span><strong>{v:,.1f}</strong></div>' for k,v in values.items())+'</div>'
def table(df):
    a=df.copy()
    for col in a.select_dtypes(include="number"):
        if col not in ["訂單次數","報價次數","最近交易天數","完工數量"]: a[col]=a[col].map(lambda n:"—" if pd.isna(n) else f"{n:,.1f}")
    return '<div class="tbl">'+a.to_html(index=False,escape=True,border=0)+'</div>'
def select(charts,index):
    if not charts:return None,'<div class="note">請先執行分析。</div>'
    i=int(index or 0);i=i if 0<=i<len(charts) else 0
    return charts[i][0],'<div class="note"><b>圖表意義與實務判讀</b><br>'+charts[i][1]+'</div>'
def lines(value):
    try:
        path=resolved(value);d=pd.read_excel(path,"產品主檔")
        options=sorted(d["產品線"].dropna().astype(str).unique().tolist())
        return gr.update(choices=options,value=options[0] if options else None),f"✅ {kind(path)}已讀取，共 {len(options)} 條產品線；開始分析。"
    except Exception as e:return gr.update(choices=[],value=None),f"❌ {type(e).__name__}: {e}"
def web_customer(value):
    try:
        start=time.time();d,elbow=customer_model(resolved(value));plots=charts_customer(d,elbow)
        cols=[v for v in ["客戶ID","客戶名稱","既有客戶等級（ERP）","年化採購金額_TWD",
              "訂單次數","報價成交率","平均折扣率","平均預估毛利率","最近交易天數","K-Means行為分群","建議策略"] if v in d]
        overview=table(d.sort_values("年化採購金額_TWD",ascending=False)[cols].head(20))
        note=f'<div class="note">Elbow Method：四群 SSE = {dict(elbow)[4]:,.1f}；四群為教學設定，須搭配實際業務判讀。</div>'
        return (f"✅ K-Means 客戶分群完成｜{len(d)} 位客戶｜{time.time()-start:.1f} 秒",
                cards({"客戶數":len(d),"分群數":4}),note,overview,plots,gr.update(value=0),*select(plots,0))
    except Exception as e:return (f"❌ 客戶分群失敗：{type(e).__name__}: {e}","","","",[],gr.update(value=0),None,"")
def web_cost(value,line):
    try:
        start=time.time();d=cost_data(resolved(value))
        if line not in set(d["產品線"].dropna()): line=sorted(d["產品線"].dropna().unique())[0]
        r,coef,score=cost_model(d,line);plots=charts_cost(r,coef,line)
        explanation='<div class="note"><b>模型評估（測試集）</b><br>R²：解釋變異；MAE：平均絕對誤差；RMSE：較重視大誤差；MAPE：平均絕對百分比誤差。成本以 KNTD 呈現。僅供教學。</div>'
        return (f"✅ Multiple Linear Regression 完成｜{line}｜測試工單 {len(r)} 筆｜{time.time()-start:.1f} 秒",
                cards(score),explanation,table(r.head(15)),plots,gr.update(value=0),*select(plots,0))
    except Exception as e:return (f"❌ 成本回歸失敗：{type(e).__name__}: {e}","","","",[],gr.update(value=0),None,"")

def dashboard(initial):
    with gr.Blocks(title="WebBased AI 數據戰情室應用實作") as app:
        gr.HTML('<div class="hero"><h1>WebBased AI 數據戰情室應用實作</h1><p>YUNFA ERP｜客戶分群 · 製造成本預測｜K-Means + Multiple Linear Regression</p></div>')
        gr.HTML('<div class="note">上傳英文原始資料或已整理的中文 ERP。USD 保留原幣且不納入 TWD 採購額；來源成本若未標幣別，正式使用前請核對。</div>')
        source=gr.File(label="選擇原始英文檔／已整理的中文 ERP Excel",type="filepath",file_types=[".xlsx"],value=initial)
        status=gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="method"><b>方法 1｜K-Means Customer Segmentation</b><p>八項報價與交易特徵；PCA、群組人數、特徵比較及 Elbow Method，共四張圖。</p></div>')
                b1=gr.Button("▶ 執行 K-Means 客戶分群",variant="primary")
            with gr.Column():
                gr.HTML('<div class="method"><b>方法 2｜Multiple Linear Regression</b><p>標準成本、完工數量及加工工時；實際與預測、成本影響因素及誤差分布，共三張圖。</p></div>')
                line=gr.Dropdown(choices=[],label="選擇產品線")
                b2=gr.Button("▶ 執行 Multiple Regression")
        with gr.Tab("K-Means 客戶分群結果"):
            s1=gr.Markdown()
            with gr.Row():k1=gr.HTML();h1=gr.HTML()
            gr.Markdown("### 客戶分群明細（金額 KNTD；比率依資料原值顯示）")
            t1=gr.HTML();v1=gr.Dropdown(choices=[("PCA 客戶分群位置",0),("各群人數",1),("客戶群特徵比較",2),("Elbow Method",3)],value=0,label="選擇圖表")
            state1=gr.State([]);img1=gr.Image(type="pil",height=540,label="客戶分群圖表");mean1=gr.HTML()
        with gr.Tab("Multiple Regression 結果"):
            s2=gr.Markdown()
            with gr.Row():k2=gr.HTML();h2=gr.HTML()
            gr.Markdown("### 工單預測明細（製造成本 KNTD）")
            t2=gr.HTML();v2=gr.Dropdown(choices=[("實際與預測成本",0),("製造成本影響因素",1),("預測誤差分布",2)],value=0,label="選擇圖表")
            state2=gr.State([]);img2=gr.Image(type="pil",height=540,label="製造成本圖表");mean2=gr.HTML()
        a=[s1,k1,h1,t1,state1,v1,img1,mean1];b=[s2,k2,h2,t2,state2,v2,img2,mean2]
        b1.click(web_customer,[source],a);b2.click(web_cost,[source,line],b)
        v1.change(select,[state1,v1],[img1,mean1]);v2.change(select,[state2,v2],[img2,mean2])
        source.change(lines,[source],[line,status]).then(web_customer,[source],a).then(web_cost,[source,line],b)
        app.load(lines,[source],[line,status]).then(web_customer,[source],a).then(web_cost,[source,line],b)
    return app

def main():
    from google.colab import files
    print("📤 請上傳一份 Test_Raw_Enterprise_Data.xlsx 或 YUNFA_ERP.xlsx")
    uploaded=files.upload()
    excel=[name for name in uploaded if name.lower().endswith(".xlsx")]
    if len(excel)!=1:raise ValueError("每次只上傳一個 .xlsx 檔案。")
    src=excel[0];print("✅ 辨識：",kind(src))
    if kind(src)=="原始英文檔":
        target=os.path.abspath("YUNFA_ERP_本次整理.xlsx")
        save_erp(prepare_raw(src),target)
        print("📥 已整理並檢查五張分析表，提供下載。")
        files.download(target)
    else:
        target=os.path.abspath(src);validate(target)
        print("✅ 原中文 ERP 已檢查，直接分析。")
    d,e=customer_model(target);cost=cost_data(target)
    print(f"✅ 客戶 {len(d)} 位；成本 {len(cost)} 筆；四張分群圖。")
    for line in sorted(cost["產品線"].dropna().unique()):
        result,coef,score=cost_model(cost,line)
        print(f"  {line}：測試集 {len(result)} 筆，R²={score['R²']:.3f}，MAE={score['MAE (KNTD)']:.1f} KNTD")
    print("🚀 啟動原戰情室的兩項分析與七張圖；不使用 API、GPU。")
    dashboard(target).queue().launch(share=True,show_error=True,debug=False,allowed_paths=[target],css=CSS)

main()
