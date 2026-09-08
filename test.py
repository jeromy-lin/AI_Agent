# ============================================================
# Manufacturing AI Intelligence
# Product → Profit → Inventory → Channel → KMeans → Agent
#
# 適用：
# ① 生活家電製造商_232SKU_加入成本價格主檔.xlsx
# ② Day1_Lab2_Smart_Home_Appliance_Product_Channel_Mapping_v2.xlsx
#
# 第二份 Excel 為選配：
# 只放第一份，也能先使用 Product / Profit / Inventory / Agent
# ============================================================


# ============================================================
# 0｜安裝套件
# ============================================================

!pip install -q -U \
    openpyxl \
    plotly \
    gradio \
    scikit-learn \
    accelerate \
    "transformers>=4.51.0"


# ============================================================
# 1｜Imports
# ============================================================

import re
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import gradio as gr

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")


# ============================================================
# 2｜APP State
# ============================================================

APP = {

    "product_ready": False,
    "channel_ready": False,
    "kmeans_ready": False,
    "local_ai_ready": False,

    "product_df": None,
    "cost_df": None,
    "inventory_df": None,
    "sku360": None,

    "channel_df": None,
    "channel_feature_df": None,

    "km_result": None,
    "km_profile": None,
    "km_scores": None,
    "km_feature_cols": None,
    "best_k": None,

    "local_tokenizer": None,
    "local_model": None
}


# ============================================================
# 3｜固定教材選項
# ============================================================

PRODUCTS = [
    "全部",
    "咖啡機",
    "氣炸鍋",
    "果汁調理機",
    "電子鍋",
    "吸塵器",
    "空氣清淨機",
    "除濕機",
    "吹風機"
]

TIERS = [
    "全部",
    "入門型",
    "中階型",
    "高階型"
]

MODES = [
    "全部",
    "OBM",
    "OEM",
    "ODM"
]

LIFECYCLES = [
    "全部",
    "新品",
    "成長機種",
    "主力/成熟",
    "歷史停產"
]

RISKS = [
    "全部",
    "正常",
    "偏高",
    "呆滯觀察",
    "高風險呆滯"
]


# ============================================================
# 4｜通用工具
# ============================================================

def get_path(file_obj):

    if file_obj is None:
        return None

    if isinstance(file_obj, str):
        return file_obj

    if hasattr(file_obj, "name"):
        return file_obj.name

    return str(file_obj)


def clean_sku(df):

    df = df.copy()

    df["SKU"] = (
        df["SKU"]
        .astype(str)
        .str.strip()
    )

    df = df[
        ~df["SKU"].isin(
            ["", "nan", "None"]
        )
    ].copy()

    return df


def smart_read(
    path,
    sheet_name,
    key="SKU",
    scan_rows=10
):

    probe = pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=None,
        nrows=scan_rows
    )

    header_row = None

    for i in range(len(probe)):

        values = [
            str(x).strip()
            if pd.notna(x)
            else ""
            for x in probe.iloc[i].tolist()
        ]

        if key in values:
            header_row = i
            break

    if header_row is None:

        raise ValueError(
            f"❌ 工作表「{sheet_name}」"
            f"前 {scan_rows} 列找不到「{key}」欄位"
        )

    df = pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=header_row
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    return df


def require_columns(
    df,
    columns,
    sheet_name
):

    missing = [
        c
        for c in columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"❌「{sheet_name}」缺少欄位："
            +
            "、".join(missing)
        )


def parse_date(series):

    result = pd.to_datetime(
        series,
        errors="coerce"
    )

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    mask = numeric.between(
        20000,
        60000
    )

    if mask.any():

        result.loc[mask] = pd.to_datetime(
            numeric.loc[mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce"
        )

    return result


def fmt_pct(x):

    if pd.isna(x):
        return "—"

    return f"{float(x) * 100:.1f}%"


def fmt_num(x):

    if pd.isna(x):
        return "0"

    return f"{float(x):,.0f}"


def blank_html(
    text="等待資料"
):

    return f"""
    <div style="
        padding:40px;
        border:1px dashed #bbb;
        border-radius:14px;
        text-align:center;
        color:#777;
        background:white;">
        {text}
    </div>
    """


def plot_html(
    fig,
    height=430
):

    fig.update_layout(

        template="plotly_white",

        height=height,

        margin=dict(
            l=30,
            r=20,
            t=60,
            b=40
        ),

        legend_title_text=""
    )

    return fig.to_html(

        full_html=False,

        include_plotlyjs="cdn",

        config={
            "displayModeBar": False,
            "displaylogo": False,
            "responsive": True
        }
    )


def make_cards(items):

    blocks = []

    for title, value, note in items:

        blocks.append(
            f"""
            <div style="
                flex:1;
                min-width:145px;
                padding:16px;
                border:1px solid #e0e0e0;
                border-radius:14px;
                background:white;
                box-shadow:0 1px 3px rgba(0,0,0,.04);
            ">

                <div style="
                    font-size:13px;
                    color:#666;">
                    {title}
                </div>

                <div style="
                    font-size:27px;
                    font-weight:700;
                    margin-top:4px;">
                    {value}
                </div>

                <div style="
                    font-size:12px;
                    color:#888;
                    margin-top:3px;">
                    {note}
                </div>

            </div>
            """
        )

    return (
        "<div style='"
        "display:flex;"
        "gap:12px;"
        "flex-wrap:wrap;"
        "'>"
        +
        "".join(blocks)
        +
        "</div>"
    )


# ============================================================
# 5｜Product Master Loader
# ============================================================

def load_product_master(path):

    xls = pd.ExcelFile(path)

    required_sheets = [
        "SKU總表",
        "成本價格主檔",
        "庫存概念"
    ]

    missing = [
        s
        for s in required_sheets
        if s not in xls.sheet_names
    ]

    if missing:

        if "產品通路對照" in xls.sheet_names:

            raise ValueError(
                "❌ 第一個檔案似乎是 Product × Channel，"
                "兩份 Excel 可能傳反。"
            )

        raise ValueError(
            "❌ Product Master 缺少："
            +
            "、".join(missing)
        )


    # --------------------------------------------------------
    # 讀三張表
    # --------------------------------------------------------

    product = smart_read(
        path,
        "SKU總表"
    )

    cost = smart_read(
        path,
        "成本價格主檔"
    )

    inventory = smart_read(
        path,
        "庫存概念"
    )


    # --------------------------------------------------------
    # 必要欄位
    # --------------------------------------------------------

    require_columns(
        product,
        [
            "SKU",
            "產品別",
            "產品定位",
            "業務模式",
            "生命週期"
        ],
        "SKU總表"
    )


    require_columns(
        cost,
        [
            "SKU",
            "標準成本",
            "基準售價/報價",
            "單位毛利",
            "基準毛利率"
        ],
        "成本價格主檔"
    )


    require_columns(
        inventory,
        [
            "SKU",
            "現有庫存量",
            "庫齡(日)",
            "庫存風險"
        ],
        "庫存概念"
    )


    # --------------------------------------------------------
    # SKU
    # --------------------------------------------------------

    product = (
        clean_sku(product)
        .drop_duplicates("SKU")
    )

    cost = (
        clean_sku(cost)
        .drop_duplicates("SKU")
    )

    inventory = (
        clean_sku(inventory)
        .drop_duplicates("SKU")
    )


    # --------------------------------------------------------
    # 日期
    # --------------------------------------------------------

    for col in [
        "上市年月",
        "停產年月"
    ]:

        if col in product.columns:

            product[col] = parse_date(
                product[col]
            )


    for col in [
        "上市年月",
        "停產年月",
        "最後異動日"
    ]:

        if col in inventory.columns:

            inventory[col] = parse_date(
                inventory[col]
            )


    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    for col in [
        "標準成本",
        "基準售價/報價",
        "單位毛利",
        "基準毛利率"
    ]:

        cost[col] = pd.to_numeric(
            cost[col],
            errors="coerce"
        )


    for col in [
        "現有庫存量",
        "庫齡(日)"
    ]:

        inventory[col] = pd.to_numeric(
            inventory[col],
            errors="coerce"
        ).fillna(0)


    # --------------------------------------------------------
    # Cost Merge
    # --------------------------------------------------------

    cost_cols = [

        c

        for c in [
            "SKU",
            "標準成本",
            "價格類型",
            "基準售價/報價",
            "單位毛利",
            "基準毛利率",
            "毛利狀態",
            "教學情境"
        ]

        if c in cost.columns
    ]


    # --------------------------------------------------------
    # Inventory Merge
    # --------------------------------------------------------

    inventory_cols = [

        c

        for c in [
            "SKU",
            "現有庫存量",
            "最後異動日",
            "庫齡(日)",
            "庫齡區間",
            "庫存風險",
            "後繼機種SKU"
        ]

        if c in inventory.columns
    ]


    sku360 = (

        product

        .merge(
            cost[cost_cols],
            on="SKU",
            how="left"
        )

        .merge(
            inventory[inventory_cols],
            on="SKU",
            how="left"
        )
    )


    sku360["現有庫存量"] = pd.to_numeric(
        sku360.get(
            "現有庫存量",
            0
        ),
        errors="coerce"
    ).fillna(0)


    sku360["庫齡(日)"] = pd.to_numeric(
        sku360.get(
            "庫齡(日)",
            0
        ),
        errors="coerce"
    ).fillna(0)


    return (
        product,
        cost,
        inventory,
        sku360
    )


# ============================================================
# 6｜Channel Loader
# ============================================================

def load_channel_master(path):

    xls = pd.ExcelFile(path)

    if "產品通路對照" not in xls.sheet_names:

        if "SKU總表" in xls.sheet_names:

            raise ValueError(
                "❌ 第二個檔案似乎是 Product Master，"
                "兩份 Excel 可能傳反。"
            )

        raise ValueError(
            "❌ 找不到「產品通路對照」"
        )


    channel = smart_read(
        path,
        "產品通路對照"
    )


    require_columns(
        channel,
        [
            "SKU",
            "產品別",
            "產品定位",
            "業務模式",
            "產品上市年月",
            "上市期間(月)",
            "通路別",
            "市場別",
            "通路上架年月",
            "通路狀態"
        ],
        "產品通路對照"
    )


    channel = clean_sku(
        channel
    )


    for col in [
        "產品上市年月",
        "通路上架年月",
        "通路下架年月"
    ]:

        if col in channel.columns:

            channel[col] = parse_date(
                channel[col]
            )


    channel["上市期間(月)"] = pd.to_numeric(
        channel["上市期間(月)"],
        errors="coerce"
    ).fillna(0)


    return channel


# ============================================================
# 7｜Product Filter
# ============================================================

def filter_product(
    product="全部",
    tier="全部",
    mode="全部",
    lifecycle="全部"
):

    if not APP["product_ready"]:
        return pd.DataFrame()


    df = APP[
        "sku360"
    ].copy()


    rules = [
        ("產品別", product),
        ("產品定位", tier),
        ("業務模式", mode),
        ("生命週期", lifecycle)
    ]


    for col, value in rules:

        if value != "全部":

            df = df[
                df[col]
                .astype(str)
                ==
                value
            ]


    return df


# ============================================================
# 8｜Channel Feature Engineering
#
# 注意：
# 一個 SKU 原本有多個 Channel Rows
# 必須先轉成「一個 SKU 一列」
# ============================================================

def build_channel_features(channel):

    ch = channel.copy()


    # --------------------------------------------------------
    # 上架延遲
    # --------------------------------------------------------

    ch["上架延遲(月)"] = (

        (
            ch["通路上架年月"].dt.year
            -
            ch["產品上市年月"].dt.year
        )
        *
        12

        +

        (
            ch["通路上架年月"].dt.month
            -
            ch["產品上市年月"].dt.month
        )
    )


    ch["上架延遲(月)"] = (

        ch["上架延遲(月)"]

        .clip(lower=0)

        .fillna(0)
    )


    # --------------------------------------------------------
    # 只拿目前在售通路算 coverage
    # --------------------------------------------------------

    active = ch[
        ch["通路狀態"].astype(str)
        ==
        "在售"
    ].copy()


    # --------------------------------------------------------
    # SKU 基本資料
    # --------------------------------------------------------

    base = (

        ch.groupby("SKU")

        .agg(

            產品別=(
                "產品別",
                "first"
            ),

            產品定位=(
                "產品定位",
                "first"
            ),

            業務模式=(
                "業務模式",
                "first"
            ),

            上市期間月=(
                "上市期間(月)",
                "first"
            ),

            平均上架延遲月=(
                "上架延遲(月)",
                "mean"
            )

        )

        .reset_index()
    )


    # --------------------------------------------------------
    # 通路數
    # --------------------------------------------------------

    active_count = (

        active

        .groupby("SKU")["通路別"]

        .nunique()

        .rename(
            "在售通路數"
        )
    )


    domestic_count = (

        active[
            active["市場別"]
            .astype(str)
            ==
            "國內"
        ]

        .groupby("SKU")["通路別"]

        .nunique()

        .rename(
            "國內在售通路數"
        )
    )


    export_count = (

        active[
            active["市場別"]
            .astype(str)
            ==
            "外銷"
        ]

        .groupby("SKU")["通路別"]

        .nunique()

        .rename(
            "外銷在售通路數"
        )
    )


    base = (

        base

        .merge(
            active_count,
            on="SKU",
            how="left"
        )

        .merge(
            domestic_count,
            on="SKU",
            how="left"
        )

        .merge(
            export_count,
            on="SKU",
            how="left"
        )
    )


    for col in [
        "在售通路數",
        "國內在售通路數",
        "外銷在售通路數"
    ]:

        base[col] = (
            base[col]
            .fillna(0)
        )


    # --------------------------------------------------------
    # 定位 Encoding
    #
    # 只是 ML Encoding
    # 不是「分數」
    # --------------------------------------------------------

    tier_map = {
        "入門型": 1,
        "中階型": 2,
        "高階型": 3
    }


    base["產品定位代碼"] = (

        base["產品定位"]

        .map(tier_map)

        .fillna(0)
    )


    # --------------------------------------------------------
    # Channel Dummy
    # --------------------------------------------------------

    if len(active) > 0:

        channel_dummy = pd.crosstab(

            active["SKU"],

            active["通路別"]
        )


        channel_dummy = (
            channel_dummy > 0
        ).astype(int)


        channel_dummy.columns = [

            "通路_" + str(c)

            for c in channel_dummy.columns
        ]


        channel_dummy = (
            channel_dummy
            .reset_index()
        )


        base = base.merge(

            channel_dummy,

            on="SKU",

            how="left"
        )


    # --------------------------------------------------------
    # Business Mode Dummy
    # --------------------------------------------------------

    mode_dummy = pd.get_dummies(

        base["業務模式"],

        prefix="模式",

        dtype=int
    )


    base = pd.concat(

        [
            base,
            mode_dummy
        ],

        axis=1
    )


    # Dummy 空值
    dummy_cols = [

        c

        for c in base.columns

        if (
            c.startswith("通路_")
            or
            c.startswith("模式_")
        )
    ]


    base[dummy_cols] = (

        base[dummy_cols]

        .fillna(0)
    )


    return base


# ============================================================
# 9｜Global KMeans
# ============================================================

def build_global_kmeans(
    channel,
    sku360
):

    base = build_channel_features(
        channel
    )


    core_features = [

        "上市期間月",

        "產品定位代碼",

        "在售通路數",

        "國內在售通路數",

        "外銷在售通路數",

        "平均上架延遲月"
    ]


    dummy_features = [

        c

        for c in base.columns

        if (
            c.startswith("通路_")
            or
            c.startswith("模式_")
        )
    ]


    feature_cols = (
        core_features
        +
        dummy_features
    )


    X = (

        base[feature_cols]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )

        .fillna(0)
    )


    # --------------------------------------------------------
    # 移除沒有變化的欄位
    # --------------------------------------------------------

    variable_cols = [

        c

        for c in X.columns

        if X[c].nunique() > 1
    ]


    X = X[
        variable_cols
    ]


    if len(X) < 5:

        raise ValueError(
            "❌ SKU 數量不足，無法建立 KMeans"
        )


    if X.shape[1] < 2:

        raise ValueError(
            "❌ KMeans 有效特徵不足"
        )


    # --------------------------------------------------------
    # StandardScaler
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )


    # --------------------------------------------------------
    # K = 2 / 3 / 4
    # --------------------------------------------------------

    scores = []


    for k in [
        2,
        3,
        4
    ]:

        if len(X) <= k:
            continue


        km = KMeans(

            n_clusters=k,

            random_state=42,

            n_init=10
        )


        labels = km.fit_predict(
            X_scaled
        )


        unique_labels = np.unique(
            labels
        )


        if (
            len(unique_labels) > 1
            and
            len(unique_labels) < len(X)
        ):

            score = silhouette_score(
                X_scaled,
                labels
            )


            scores.append({

                "K": k,

                "Silhouette": round(
                    float(score),
                    4
                )
            })


    if scores:

        score_df = pd.DataFrame(
            scores
        )


        best_k = int(

            score_df.loc[
                score_df[
                    "Silhouette"
                ].idxmax(),
                "K"
            ]
        )


    else:

        score_df = pd.DataFrame(
            columns=[
                "K",
                "Silhouette"
            ]
        )

        best_k = 2


    # --------------------------------------------------------
    # Final KMeans
    # --------------------------------------------------------

    final_model = KMeans(

        n_clusters=best_k,

        random_state=42,

        n_init=10
    )


    labels = final_model.fit_predict(
        X_scaled
    )


    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca = PCA(
        n_components=2
    )


    coords = pca.fit_transform(
        X_scaled
    )


    result = base.copy()


    result["PC1"] = (
        coords[:, 0]
    )

    result["PC2"] = (
        coords[:, 1]
    )


    result["Cluster"] = [

        f"C{x}"

        for x in labels
    ]


    # --------------------------------------------------------
    # 分群完才接回毛利、庫存
    #
    # 這些沒有參與 KMeans
    # --------------------------------------------------------

    business_cols = [

        c

        for c in [
            "SKU",
            "基準毛利率",
            "現有庫存量",
            "庫齡(日)",
            "庫存風險",
            "生命週期"
        ]

        if c in sku360.columns
    ]


    result = result.merge(

        sku360[business_cols],

        on="SKU",

        how="left"
    )


    # --------------------------------------------------------
    # Cluster Profile
    # --------------------------------------------------------

    profile = (

        result

        .groupby("Cluster")

        .agg(

            SKU數=(
                "SKU",
                "nunique"
            ),

            平均上市期間月=(
                "上市期間月",
                "mean"
            ),

            平均在售通路數=(
                "在售通路數",
                "mean"
            ),

            平均國內通路數=(
                "國內在售通路數",
                "mean"
            ),

            平均外銷通路數=(
                "外銷在售通路數",
                "mean"
            ),

            平均上架延遲月=(
                "平均上架延遲月",
                "mean"
            ),

            平均毛利率=(
                "基準毛利率",
                "mean"
            ),

            平均庫存量=(
                "現有庫存量",
                "mean"
            ),

            平均庫齡日=(
                "庫齡(日)",
                "mean"
            )

        )

        .reset_index()
    )


    profile["平均毛利率(%)"] = (
        profile["平均毛利率"]
        *
        100
    )


    round_cols = [

        "平均上市期間月",
        "平均在售通路數",
        "平均國內通路數",
        "平均外銷通路數",
        "平均上架延遲月",
        "平均毛利率(%)",
        "平均庫存量",
        "平均庫齡日"
    ]


    profile[round_cols] = (
        profile[round_cols]
        .round(1)
    )


    profile.drop(
        columns=[
            "平均毛利率"
        ],
        inplace=True
    )


    return (

        base,
        variable_cols,
        result,
        profile,
        score_df,
        best_k
    )


# ============================================================
# 10｜Project Loader
# ============================================================

def load_project(
    product_file,
    channel_file
):

    product_path = get_path(
        product_file
    )

    channel_path = get_path(
        channel_file
    )


    if not product_path:

        raise gr.Error(
            "請先上傳第一份 Product Master"
        )


    try:

        (
            product,
            cost,
            inventory,
            sku360

        ) = load_product_master(
            product_path
        )


        APP.update({

            "product_ready": True,

            "product_df": product,
            "cost_df": cost,
            "inventory_df": inventory,
            "sku360": sku360,

            "channel_ready": False,
            "kmeans_ready": False,

            "channel_df": None,
            "channel_feature_df": None,

            "km_result": None,
            "km_profile": None,
            "km_scores": None,
            "km_feature_cols": None,
            "best_k": None
        })


        status = (

            "### ✅ STEP 1｜Product Intelligence Ready\n"

            f"- Product Master："
            f"**{product['SKU'].nunique()} SKU**\n"

            f"- Cost："
            f"**{cost['SKU'].nunique()} SKU**\n"

            f"- Inventory："
            f"**{inventory['SKU'].nunique()} SKU**"
        )


        # ====================================================
        # Optional Channel
        # ====================================================

        if channel_path:


            channel = load_channel_master(
                channel_path
            )


            product_skus = set(
                sku360["SKU"]
            )

            channel_skus = set(
                channel["SKU"]
            )


            overlap = (
                product_skus
                &
                channel_skus
            )


            match_rate = (

                len(overlap)

                /

                max(
                    len(channel_skus),
                    1
                )
            )


            if match_rate < 0.5:

                raise ValueError(

                    f"兩份 Excel SKU 對應率只有 "
                    f"{match_rate:.1%}，"
                    "請確認檔案是否正確。"
                )


            (
                feature_df,
                feature_cols,
                km_result,
                km_profile,
                km_scores,
                best_k

            ) = build_global_kmeans(

                channel,
                sku360
            )


            APP.update({

                "channel_ready": True,

                "channel_df": channel,

                "channel_feature_df": feature_df,

                "kmeans_ready": True,

                "km_feature_cols": feature_cols,

                "km_result": km_result,

                "km_profile": km_profile,

                "km_scores": km_scores,

                "best_k": best_k
            })


            status += (

                "\n\n"
                "### ✅ STEP 2｜Channel + KMeans Ready\n"

                f"- Product × Channel："
                f"**{len(channel):,} records**\n"

                f"- Channel SKU："
                f"**{channel['SKU'].nunique()}**\n"

                f"- SKU Match："
                f"**{match_rate:.1%}**\n"

                f"- Global KMeans："
                f"**Best K = {best_k}**"
            )


        else:

            status += (

                "\n\n"
                "### 🔒 STEP 2 尚未解鎖\n"

                "上傳第二份 Product × Channel Excel，"
                "即可解鎖 Channel 與 KMeans。"
            )


        return refresh_dashboard(
            status
        )


    except Exception as e:

        raise gr.Error(
            str(e)
        )


# ============================================================
# 11｜Dashboard
# ============================================================

def refresh_dashboard(
    status="### Ready"
):

    if not APP["product_ready"]:

        return (

            status,

            make_cards([
                (
                    "狀態",
                    "未載入",
                    "請上傳 Excel"
                )
            ]),

            blank_html(),
            blank_html(),
            blank_html(),
            blank_html(),

            "### 🔒 KMeans 尚未解鎖",

            blank_html(),

            pd.DataFrame(),

            pd.DataFrame()
        )


    df = APP[
        "sku360"
    ]


    historical = (

        df["生命週期"]
        .astype(str)
        ==
        "歷史停產"
    )


    high_risk = (

        df["庫存風險"]
        .astype(str)
        .str.contains(
            "高風險",
            na=False
        )
    )


    cards = make_cards([

        (
            "總 SKU",
            df["SKU"].nunique(),
            "Product Master"
        ),

        (
            "產品別",
            df["產品別"].nunique(),
            "產品線"
        ),

        (
            "現行 SKU",
            int(
                (~historical).sum()
            ),
            "非停產產品"
        ),

        (
            "歷史 SKU",
            int(
                historical.sum()
            ),
            "歷史停產"
        ),

        (
            "平均毛利率",
            fmt_pct(
                df["基準毛利率"]
                .mean()
            ),
            "基準"
        ),

        (
            "高風險庫存",
            int(
                high_risk.sum()
            ),
            "SKU"
        )

    ])


    # --------------------------------------------------------
    # Product
    # --------------------------------------------------------

    p1 = (

        df

        .groupby(
            [
                "產品別",
                "產品定位"
            ]
        )

        .size()

        .reset_index(
            name="SKU數"
        )
    )


    fig1 = px.bar(

        p1,

        x="產品別",

        y="SKU數",

        color="產品定位",

        text="SKU數",

        title="Product Portfolio"
    )


    # --------------------------------------------------------
    # Profit
    # --------------------------------------------------------

    p2 = (

        df

        .groupby(
            "業務模式"
        )["基準毛利率"]

        .mean()

        .reset_index()
    )


    p2["平均毛利率(%)"] = (
        p2["基準毛利率"]
        *
        100
    )


    fig2 = px.bar(

        p2,

        x="業務模式",

        y="平均毛利率(%)",

        text_auto=".1f",

        title="OBM / OEM / ODM Profit"
    )


    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    p3 = (

        df

        .groupby(
            "庫存風險"
        )["現有庫存量"]

        .sum()

        .reset_index()
    )


    fig3 = px.bar(

        p3,

        x="庫存風險",

        y="現有庫存量",

        text_auto=".0f",

        title="Inventory Risk"
    )


    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    p4 = (

        df

        .groupby(
            "生命週期"
        )

        .size()

        .reset_index(
            name="SKU數"
        )
    )


    fig4 = px.pie(

        p4,

        names="生命週期",

        values="SKU數",

        hole=.45,

        title="Product Lifecycle"
    )


    # ========================================================
    # KMeans
    # ========================================================

    if APP["kmeans_ready"]:

        km = APP[
            "km_result"
        ]


        km_fig = px.scatter(

            km,

            x="PC1",

            y="PC2",

            color="Cluster",

            symbol="業務模式",

            hover_name="SKU",

            hover_data=[
                "產品別",
                "產品定位",
                "業務模式",
                "在售通路數",
                "國內在售通路數",
                "外銷在售通路數",
                "平均上架延遲月"
            ],

            title=(
                f"Global KMeans Strategy Map｜"
                f"Best K = {APP['best_k']}"
            )
        )


        km_status = (

            f"### 🧠 KMeans Ready｜"
            f"Best K = {APP['best_k']}\n\n"

            "分群使用：上市期間、產品定位、"
            "通路覆蓋、國內/外銷、上架延遲、"
            "通路組合與 OBM/OEM/ODM。\n\n"

            "**毛利與庫存沒有參與分群，"
            "只用來解讀分群結果。**"
        )


        km_html = plot_html(
            km_fig,
            560
        )


        km_profile = (
            APP["km_profile"]
            .copy()
        )


        km_scores = (
            APP["km_scores"]
            .copy()
        )


    else:

        km_status = (
            "### 🔒 KMeans 尚未解鎖"
        )

        km_html = blank_html(
            "請上傳第二份 Product × Channel Excel"
        )

        km_profile = pd.DataFrame()

        km_scores = pd.DataFrame()


    return (

        status,

        cards,

        plot_html(fig1),

        plot_html(fig2),

        plot_html(fig3),

        plot_html(fig4),

        km_status,

        km_html,

        km_profile,

        km_scores
    )


# ============================================================
# 12｜Product Tool
# ============================================================

def product_tool(
    product,
    tier,
    mode,
    lifecycle
):

    df = filter_product(
        product,
        tier,
        mode,
        lifecycle
    )


    if df.empty:

        return (
            "### 找不到符合條件的產品",
            pd.DataFrame(),
            blank_html()
        )


    summary = (

        "### 🔎 Product Intelligence\n"

        f"- SKU："
        f"**{df['SKU'].nunique()}**\n"

        f"- 平均毛利率："
        f"**{fmt_pct(df['基準毛利率'].mean())}**\n"

        f"- 現有庫存："
        f"**{fmt_num(df['現有庫存量'].sum())}**"
    )


    g = (

        df

        .groupby(
            [
                "產品定位",
                "業務模式"
            ]
        )

        .size()

        .reset_index(
            name="SKU數"
        )
    )


    fig = px.bar(

        g,

        x="產品定位",

        y="SKU數",

        color="業務模式",

        barmode="group",

        text="SKU數",

        title="產品定位 × 業務模式"
    )


    cols = [

        c

        for c in [
            "SKU",
            "產品別",
            "產品定位",
            "業務模式",
            "品牌/委託客戶",
            "主規格值",
            "單位",
            "產品概念",
            "上市年月",
            "生命週期",
            "基準毛利率",
            "現有庫存量",
            "庫存風險"
        ]

        if c in df.columns
    ]


    return (

        summary,

        df[cols]
        .head(80),

        plot_html(fig)
    )


# ============================================================
# 13｜Profit Tool
# ============================================================

def profit_tool(
    product,
    mode
):

    df = filter_product(
        product=product,
        mode=mode
    )


    df = df[
        df["基準毛利率"]
        .notna()
    ].copy()


    if df.empty:

        return (
            "### 找不到毛利資料",
            pd.DataFrame(),
            blank_html()
        )


    minimum = df.loc[
        df["基準毛利率"]
        .idxmin()
    ]


    summary = (

        "### 💰 Profit Intelligence\n"

        f"- 平均毛利率："
        f"**{fmt_pct(df['基準毛利率'].mean())}**\n"

        f"- 毛利率低於 20%："
        f"**{(df['基準毛利率'] < .20).sum()} SKU**\n"

        f"- 最低毛利 SKU："
        f"**{minimum['SKU']} / "
        f"{fmt_pct(minimum['基準毛利率'])}**"
    )


    g = (

        df

        .groupby(
            [
                "產品別",
                "業務模式"
            ]
        )["基準毛利率"]

        .mean()

        .reset_index()
    )


    g["平均毛利率(%)"] = (
        g["基準毛利率"]
        *
        100
    )


    fig = px.bar(

        g,

        x="產品別",

        y="平均毛利率(%)",

        color="業務模式",

        barmode="group",

        text_auto=".1f",

        title="Product × Business Model Profit"
    )


    cols = [

        c

        for c in [
            "SKU",
            "產品別",
            "產品定位",
            "業務模式",
            "標準成本",
            "基準售價/報價",
            "單位毛利",
            "基準毛利率",
            "毛利狀態",
            "教學情境"
        ]

        if c in df.columns
    ]


    return (

        summary,

        df
        .sort_values(
            "基準毛利率"
        )[cols]
        .head(60),

        plot_html(fig)
    )


# ============================================================
# 14｜Inventory Tool
# ============================================================

def inventory_tool(
    product,
    risk
):

    df = filter_product(
        product=product
    )


    df = df[
        df["現有庫存量"]
        >
        0
    ].copy()


    if risk != "全部":

        df = df[
            df["庫存風險"]
            .astype(str)
            ==
            risk
        ]


    if df.empty:

        return (
            "### 找不到符合條件的庫存",
            pd.DataFrame(),
            blank_html()
        )


    summary = (

        "### 📦 Inventory Intelligence\n"

        f"- 有庫存 SKU："
        f"**{df['SKU'].nunique()}**\n"

        f"- 庫存總量："
        f"**{fmt_num(df['現有庫存量'].sum())}**\n"

        f"- 平均庫齡："
        f"**{df['庫齡(日)'].mean():.0f} 天**"
    )


    g = (

        df

        .groupby(
            [
                "產品別",
                "庫存風險"
            ]
        )["現有庫存量"]

        .sum()

        .reset_index()
    )


    fig = px.bar(

        g,

        x="產品別",

        y="現有庫存量",

        color="庫存風險",

        title="Product × Inventory Risk"
    )


    cols = [

        c

        for c in [
            "SKU",
            "產品別",
            "業務模式",
            "生命週期",
            "現有庫存量",
            "庫齡(日)",
            "庫齡區間",
            "庫存風險",
            "後繼機種SKU"
        ]

        if c in df.columns
    ]


    return (

        summary,

        df
        .sort_values(
            [
                "庫齡(日)",
                "現有庫存量"
            ],
            ascending=[
                False,
                False
            ]
        )[cols]
        .head(60),

        plot_html(fig)
    )


# ============================================================
# 15｜Channel Tool
# ============================================================

def channel_tool(
    product,
    mode,
    market
):

    if not APP["channel_ready"]:

        return (

            "### 🔒 Channel 尚未解鎖\n"
            "請上傳第二份 Product × Channel Excel。",

            pd.DataFrame(),

            blank_html(
                "Product × Channel 尚未載入"
            )
        )


    df = APP[
        "channel_df"
    ].copy()


    if product != "全部":

        df = df[
            df["產品別"]
            ==
            product
        ]


    if mode != "全部":

        df = df[
            df["業務模式"]
            ==
            mode
        ]


    if market != "全部":

        df = df[
            df["市場別"]
            ==
            market
        ]


    active = df[
        df["通路狀態"]
        .astype(str)
        ==
        "在售"
    ]


    summary = (

        "### 🏬 Channel Intelligence\n"

        f"- Product × Channel Records："
        f"**{len(df):,}**\n"

        f"- SKU："
        f"**{df['SKU'].nunique()}**\n"

        f"- 在售通路種類："
        f"**{active['通路別'].nunique()}**"
    )


    g = (

        active

        .groupby(
            "通路別"
        )["SKU"]

        .nunique()

        .reset_index(
            name="在售SKU數"
        )
    )


    fig = px.bar(

        g,

        x="通路別",

        y="在售SKU數",

        text="在售SKU數",

        title="Active Channel Coverage"
    )


    cols = [

        c

        for c in [
            "SKU",
            "產品別",
            "產品定位",
            "業務模式",
            "通路別",
            "市場別",
            "通路上架年月",
            "通路下架年月",
            "通路狀態",
            "交易型態"
        ]

        if c in df.columns
    ]


    return (

        summary,

        df[cols]
        .head(100),

        plot_html(fig)
    )


# ============================================================
# 16｜單一產品 KMeans
# ============================================================

def calculate_product_kmeans(
    product
):

    if not APP["kmeans_ready"]:

        return None


    base = APP[
        "channel_feature_df"
    ].copy()


    df = base[
        base["產品別"]
        ==
        product
    ].copy()


    if len(df) < 4:
        return None


    feature_cols = [

        c

        for c in APP[
            "km_feature_cols"
        ]

        if c in df.columns
    ]


    X = (

        df[feature_cols]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )

        .fillna(0)
    )


    variable_cols = [

        c

        for c in X.columns

        if X[c].nunique() > 1
    ]


    X = X[
        variable_cols
    ]


    if X.shape[1] < 2:

        return None


    scaler = StandardScaler()


    X_scaled = scaler.fit_transform(
        X
    )


    # --------------------------------------------------------
    # 教學版
    # 各產品固定 3 群
    # 資料太少則 2 群
    # --------------------------------------------------------

    if len(df) >= 6:
        k = 3
    else:
        k = 2


    km = KMeans(

        n_clusters=k,

        random_state=42,

        n_init=10
    )


    labels = km.fit_predict(
        X_scaled
    )


    pca = PCA(
        n_components=2
    )


    coords = pca.fit_transform(
        X_scaled
    )


    result = df.copy()


    result["PC1"] = (
        coords[:, 0]
    )


    result["PC2"] = (
        coords[:, 1]
    )


    result["Cluster"] = [

        f"C{x}"

        for x in labels
    ]


    business_cols = [

        c

        for c in [
            "SKU",
            "基準毛利率",
            "現有庫存量",
            "庫齡(日)",
            "庫存風險"
        ]

        if c in APP[
            "sku360"
        ].columns
    ]


    result = result.merge(

        APP[
            "sku360"
        ][business_cols],

        on="SKU",

        how="left"
    )


    profile = (

        result

        .groupby(
            "Cluster"
        )

        .agg(

            SKU數=(
                "SKU",
                "nunique"
            ),

            平均在售通路數=(
                "在售通路數",
                "mean"
            ),

            平均國內通路數=(
                "國內在售通路數",
                "mean"
            ),

            平均外銷通路數=(
                "外銷在售通路數",
                "mean"
            ),

            平均上架延遲月=(
                "平均上架延遲月",
                "mean"
            ),

            平均毛利率=(
                "基準毛利率",
                "mean"
            ),

            平均庫存量=(
                "現有庫存量",
                "mean"
            ),

            平均庫齡日=(
                "庫齡(日)",
                "mean"
            )

        )

        .reset_index()
    )


    profile["平均毛利率(%)"] = (
        profile["平均毛利率"]
        *
        100
    )


    profile.drop(
        columns=[
            "平均毛利率"
        ],
        inplace=True
    )


    profile = profile.round(
        1
    )


    return (
        result,
        profile,
        k
    )


def product_kmeans_tool(
    product
):

    data = calculate_product_kmeans(
        product
    )


    if data is None:

        return (

            "### 🔒 此產品暫時無法建立 KMeans",

            pd.DataFrame(),

            blank_html(),

            pd.DataFrame()
        )


    (
        result,
        profile,
        k
    ) = data


    fig = px.scatter(

        result,

        x="PC1",

        y="PC2",

        color="Cluster",

        symbol="業務模式",

        hover_name="SKU",

        hover_data=[
            "產品定位",
            "業務模式",
            "在售通路數",
            "國內在售通路數",
            "外銷在售通路數"
        ],

        title=(
            f"{product}｜"
            f"KMeans Strategy Map｜"
            f"K = {k}"
        )
    )


    summary = (

        f"### 🧠 {product} KMeans\n"

        f"- SKU："
        f"**{len(result)}**\n"

        f"- 分成："
        f"**{k} 群**\n\n"

        "Cluster 編號只代表群組，"
        "沒有高低、好壞或排名意義。"
    )


    detail_cols = [

        c

        for c in [
            "SKU",
            "產品定位",
            "業務模式",
            "在售通路數",
            "國內在售通路數",
            "外銷在售通路數",
            "平均上架延遲月",
            "Cluster",
            "基準毛利率",
            "現有庫存量",
            "庫齡(日)",
            "庫存風險"
        ]

        if c in result.columns
    ]


    return (

        summary,

        profile,

        plot_html(
            fig,
            560
        ),

        result[detail_cols]
    )


# ============================================================
# 17｜Local Qwen3-0.6B
# ============================================================

def load_local_ai():

    if APP["local_ai_ready"]:

        return (
            "### ✅ Local Qwen3-0.6B 已經啟動"
        )


    try:

        import torch

        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM
        )


        MODEL_NAME = (
            "Qwen/Qwen3-0.6B"
        )


        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                MODEL_NAME
            )
        )


        model = (
            AutoModelForCausalLM
            .from_pretrained(

                MODEL_NAME,

                torch_dtype="auto",

                device_map="auto"
            )
        )


        APP[
            "local_tokenizer"
        ] = tokenizer


        APP[
            "local_model"
        ] = model


        APP[
            "local_ai_ready"
        ] = True


        return (
            "### ✅ Local Qwen3-0.6B Ready\n"
            "不需要 Gemini API Key。"
        )


    except Exception as e:

        APP[
            "local_ai_ready"
        ] = False


        return (
            "### ⚠️ Local AI 載入失敗\n\n"
            "前面的 Product / Profit / Inventory / "
            "Channel / KMeans / Agent 仍然可以使用。\n\n"
            +
            str(e)
        )


# ============================================================
# 18｜Qwen Explanation
# ============================================================

def local_explain(
    question,
    python_answer,
    evidence
):

    if not APP[
        "local_ai_ready"
    ]:

        return python_answer


    try:

        import torch


        tokenizer = APP[
            "local_tokenizer"
        ]


        model = APP[
            "local_model"
        ]


        if isinstance(
            evidence,
            pd.DataFrame
        ):

            evidence_text = (

                evidence

                .head(8)

                .to_string(
                    index=False
                )
            )


        else:

            evidence_text = str(
                evidence
            )


        prompt = (

            "你是一位製造業產品與營運分析助理。\n"

            "請務必遵守：\n"

            "1. 只能根據 Python 已算出的結果與資料證據回答。\n"

            "2. 不得自行捏造任何數字。\n"

            "3. 使用繁體中文。\n"

            "4. 內容簡潔，約 100～180 字。\n"

            "5. 使用『觀察、可能原因、管理建議』三段說明。\n\n"

            f"使用者問題：\n{question}\n\n"

            f"Python 結果：\n{python_answer}\n\n"

            f"資料證據：\n{evidence_text}"
        )


        messages = [

            {
                "role": "user",
                "content": prompt
            }

        ]


        text = tokenizer.apply_chat_template(

            messages,

            tokenize=False,

            add_generation_prompt=True,

            enable_thinking=False
        )


        inputs = tokenizer(

            text,

            return_tensors="pt"
        )


        device = next(
            model.parameters()
        ).device


        inputs = {

            k: v.to(device)

            for k, v in inputs.items()
        }


        with torch.no_grad():

            outputs = model.generate(

                **inputs,

                max_new_tokens=220,

                do_sample=False
            )


        new_tokens = outputs[
            0,
            inputs["input_ids"].shape[1]:
        ]


        reply = tokenizer.decode(

            new_tokens,

            skip_special_tokens=True

        ).strip()


        if not reply:

            return python_answer


        return (

            python_answer

            +

            "\n\n---\n\n"

            "### 🧠 Local AI 解讀\n\n"

            +

            reply
        )


    except Exception:

        return python_answer


# ============================================================
# 19｜Agent Parser
# ============================================================

def detect_product(question):

    for product in PRODUCTS[1:]:

        if product in question:
            return product

    return None


def detect_modes(question):

    q = question.upper()

    return [

        x

        for x in [
            "OBM",
            "OEM",
            "ODM"
        ]

        if x in q
    ]


# ============================================================
# 20｜AI Agent Controller
# ============================================================

def ask_agent(
    question,
    use_local_ai
):

    if not APP[
        "product_ready"
    ]:

        return (

            "### ⛔ 請先載入 Product Master",

            pd.DataFrame(),

            blank_html()
        )


    q = str(
        question or ""
    ).strip()


    if not q:

        return (

            "### 請輸入管理問題",

            pd.DataFrame(),

            blank_html()
        )


    product = detect_product(
        q
    )


    modes = detect_modes(
        q
    )


    # ========================================================
    # A｜KMeans / 分群
    # ========================================================

    if any(

        keyword in q.lower()

        for keyword in [
            "kmeans",
            "cluster",
            "分群",
            "群組"
        ]
    ):


        if not APP[
            "kmeans_ready"
        ]:

            return (

                "### 🔒 KMeans 尚未解鎖\n"
                "請加入第二份 Product × Channel Excel。",

                pd.DataFrame(),

                blank_html()
            )


        # ----------------------------------------------------
        # 有指定產品 → 使用該產品自己的 KMeans
        # ----------------------------------------------------

        if product:

            data = calculate_product_kmeans(
                product
            )


            if data is None:

                return (

                    "### 找不到足夠資料進行產品分群",

                    pd.DataFrame(),

                    blank_html()
                )


            (
                result,
                profile,
                k
            ) = data


            answer = (

                f"### 🤖 {product} Cluster Agent\n"

                f"{product} 共分為 "
                f"**{k} 群**。\n\n"

                "這些群組是依通路覆蓋、"
                "國內/外銷結構、上市時間、"
                "定位與業務模式等行為形成。"
            )


            evidence = profile


            fig = px.scatter(

                result,

                x="PC1",

                y="PC2",

                color="Cluster",

                hover_name="SKU",

                title=(
                    f"Agent｜"
                    f"{product} KMeans"
                )
            )


        # ----------------------------------------------------
        # 沒指定產品 → Global KMeans
        # ----------------------------------------------------

        else:

            result = APP[
                "km_result"
            ].copy()


            cluster_match = re.search(

                r"(?:cluster|群|c)\s*([0-9]+)",

                q.lower()
            )


            if cluster_match:

                cluster = (
                    "C"
                    +
                    cluster_match.group(1)
                )


                result = result[
                    result["Cluster"]
                    ==
                    cluster
                ]


            evidence = (

                result

                .groupby(
                    "Cluster"
                )

                .agg(

                    SKU數=(
                        "SKU",
                        "nunique"
                    ),

                    平均在售通路數=(
                        "在售通路數",
                        "mean"
                    ),

                    平均國內通路數=(
                        "國內在售通路數",
                        "mean"
                    ),

                    平均外銷通路數=(
                        "外銷在售通路數",
                        "mean"
                    ),

                    平均毛利率=(
                        "基準毛利率",
                        "mean"
                    ),

                    平均庫存量=(
                        "現有庫存量",
                        "mean"
                    ),

                    平均庫齡日=(
                        "庫齡(日)",
                        "mean"
                    )

                )

                .reset_index()
            )


            if "平均毛利率" in evidence.columns:

                evidence[
                    "平均毛利率(%)"
                ] = (
                    evidence[
                        "平均毛利率"
                    ]
                    *
                    100
                )

                evidence.drop(
                    columns=[
                        "平均毛利率"
                    ],
                    inplace=True
                )


            evidence = evidence.round(
                1
            )


            answer = (

                "### 🤖 Global KMeans Agent\n"

                f"公司目前分成 "
                f"**{APP['best_k']} 個主要產品通路群**。\n\n"

                "KMeans 本身只找出"
                "行為相似的產品，"
                "Cluster 編號不代表好壞。"
            )


            fig = px.scatter(

                result,

                x="PC1",

                y="PC2",

                color="Cluster",

                hover_name="SKU",

                title="Agent｜Global KMeans"
            )


    # ========================================================
    # B｜Channel
    # ========================================================

    elif any(

        keyword in q

        for keyword in [
            "通路",
            "外銷",
            "上架",
            "下架",
            "市場"
        ]
    ):


        if not APP[
            "channel_ready"
        ]:

            return (

                "### 🔒 Channel 尚未解鎖\n"
                "請加入第二份 Excel。",

                pd.DataFrame(),

                blank_html()
            )


        df = APP[
            "channel_df"
        ].copy()


        if product:

            df = df[
                df["產品別"]
                ==
                product
            ]


        if modes:

            df = df[
                df["業務模式"]
                .isin(modes)
            ]


        if "外銷" in q:

            df = df[
                df["市場別"]
                ==
                "外銷"
            ]


        if "國內" in q:

            df = df[
                df["市場別"]
                ==
                "國內"
            ]


        if "在售" in q:

            df = df[
                df["通路狀態"]
                ==
                "在售"
            ]


        if df.empty:

            return (

                "### 找不到符合條件的 Channel 資料",

                pd.DataFrame(),

                blank_html()
            )


        active = df[
            df["通路狀態"]
            ==
            "在售"
        ]


        answer = (

            "### 🤖 Channel Agent\n"

            f"- SKU："
            f"**{df['SKU'].nunique()}**\n"

            f"- 通路種類："
            f"**{df['通路別'].nunique()}**\n"

            f"- 在售通路種類："
            f"**{active['通路別'].nunique()}**"
        )


        evidence = (

            df

            .groupby(
                [
                    "通路別",
                    "市場別",
                    "通路狀態"
                ]
            )["SKU"]

            .nunique()

            .reset_index(
                name="SKU數"
            )
        )


        fig = px.bar(

            evidence,

            x="通路別",

            y="SKU數",

            color="通路狀態",

            title="Agent｜Channel Structure"
        )


    # ========================================================
    # C｜Inventory
    # ========================================================

    elif any(

        keyword in q

        for keyword in [
            "庫存",
            "庫齡",
            "呆滯"
        ]
    ):


        df = APP[
            "sku360"
        ].copy()


        if product:

            df = df[
                df["產品別"]
                ==
                product
            ]


        if modes:

            df = df[
                df["業務模式"]
                .isin(modes)
            ]


        df = df[
            df["現有庫存量"]
            >
            0
        ]


        if "高風險" in q:

            df = df[
                df["庫存風險"]
                .astype(str)
                .str.contains(
                    "高風險",
                    na=False
                )
            ]


        if "呆滯" in q:

            df = df[
                df["庫存風險"]
                .astype(str)
                .str.contains(
                    "呆滯",
                    na=False
                )
            ]


        if "停產" in q:

            df = df[
                df["生命週期"]
                ==
                "歷史停產"
            ]


        answer = (

            "### 🤖 Inventory Agent\n"

            f"- SKU："
            f"**{df['SKU'].nunique()}**\n"

            f"- 庫存總量："
            f"**{fmt_num(df['現有庫存量'].sum())}**\n"

            f"- 平均庫齡："
            f"**{df['庫齡(日)'].mean():.0f} 天**"
        )


        cols = [

            c

            for c in [
                "SKU",
                "產品別",
                "生命週期",
                "現有庫存量",
                "庫齡(日)",
                "庫存風險",
                "後繼機種SKU"
            ]

            if c in df.columns
        ]


        evidence = (

            df

            .sort_values(
                "庫齡(日)",
                ascending=False
            )[cols]

            .head(60)
        )


        g = (

            df

            .groupby(
                "產品別"
            )["現有庫存量"]

            .sum()

            .reset_index()
        )


        fig = px.bar(

            g,

            x="產品別",

            y="現有庫存量",

            title="Agent｜Inventory Risk"
        )


    # ========================================================
    # D｜Profit
    # ========================================================

    elif any(

        keyword in q

        for keyword in [
            "毛利",
            "成本",
            "售價",
            "報價"
        ]
    ):


        df = APP[
            "sku360"
        ].copy()


        if product:

            df = df[
                df["產品別"]
                ==
                product
            ]


        if modes:

            df = df[
                df["業務模式"]
                .isin(modes)
            ]


        df = df[
            df["基準毛利率"]
            .notna()
        ]


        threshold = re.search(

            r"(?:低於|小於|<)\s*(\d+(?:\.\d+)?)\s*%?",

            q
        )


        if threshold:

            value = float(
                threshold.group(1)
            )


            if value > 1:
                value /= 100


            df = df[
                df["基準毛利率"]
                <
                value
            ]


        if df.empty:

            return (

                "### 找不到符合條件的毛利資料",

                pd.DataFrame(),

                blank_html()
            )


        mode_summary = (

            df

            .groupby(
                "業務模式"
            )

            .agg(

                SKU數=(
                    "SKU",
                    "nunique"
                ),

                平均毛利率=(
                    "基準毛利率",
                    "mean"
                )

            )

            .reset_index()
        )


        best = mode_summary.loc[
            mode_summary[
                "平均毛利率"
            ].idxmax()
        ]


        worst = mode_summary.loc[
            mode_summary[
                "平均毛利率"
            ].idxmin()
        ]


        answer = (

            "### 🤖 Profit Agent\n"

            f"- SKU："
            f"**{df['SKU'].nunique()}**\n"

            f"- 平均毛利率："
            f"**{fmt_pct(df['基準毛利率'].mean())}**\n"

            f"- 較高平均毛利模式："
            f"**{best['業務模式']}**\n"

            f"- 較低平均毛利模式："
            f"**{worst['業務模式']}**"
        )


        cols = [

            c

            for c in [
                "SKU",
                "產品別",
                "產品定位",
                "業務模式",
                "標準成本",
                "基準售價/報價",
                "單位毛利",
                "基準毛利率",
                "毛利狀態"
            ]

            if c in df.columns
        ]


        evidence = (

            df

            .sort_values(
                "基準毛利率"
            )[cols]

            .head(60)
        )


        mode_summary[
            "平均毛利率(%)"
        ] = (
            mode_summary[
                "平均毛利率"
            ]
            *
            100
        )


        fig = px.bar(

            mode_summary,

            x="業務模式",

            y="平均毛利率(%)",

            text_auto=".1f",

            title="Agent｜Profit"
        )


    # ========================================================
    # E｜Product
    # ========================================================

    else:


        df = APP[
            "sku360"
        ].copy()


        if product:

            df = df[
                df["產品別"]
                ==
                product
            ]


        if modes:

            df = df[
                df["業務模式"]
                .isin(modes)
            ]


        if "高階" in q:

            df = df[
                df["產品定位"]
                ==
                "高階型"
            ]


        elif "中階" in q:

            df = df[
                df["產品定位"]
                ==
                "中階型"
            ]


        elif "入門" in q:

            df = df[
                df["產品定位"]
                ==
                "入門型"
            ]


        if "停產" in q:

            df = df[
                df["生命週期"]
                ==
                "歷史停產"
            ]


        answer = (

            "### 🤖 Product Agent\n"

            f"找到 **{df['SKU'].nunique()} SKU**。\n\n"

            f"產品結構："
            f"{df['產品別'].value_counts().to_dict()}\n\n"

            f"業務模式："
            f"{df['業務模式'].value_counts().to_dict()}"
        )


        cols = [

            c

            for c in [
                "SKU",
                "產品別",
                "產品定位",
                "業務模式",
                "品牌/委託客戶",
                "主規格值",
                "單位",
                "產品概念",
                "生命週期"
            ]

            if c in df.columns
        ]


        evidence = (
            df[cols]
            .head(60)
        )


        g = (

            df

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


        fig = px.bar(

            g,

            x="產品別",

            y="SKU數",

            color="業務模式",

            title="Agent｜Product Portfolio"
        )


    # ========================================================
    # Local Qwen 可選
    # ========================================================

    if use_local_ai:

        answer = local_explain(

            q,

            answer,

            evidence
        )


    return (

        answer,

        evidence,

        plot_html(
            fig,
            500
        )
    )


# ============================================================
# 21｜建立 Gradio Web
# ============================================================

try:
    gr.close_all()
except:
    pass


CSS = """

.gradio-container {
    max-width: 1450px !important;
}

.main-title {
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    color: #666;
    font-size: 15px;
    margin-bottom: 15px;
}

"""


with gr.Blocks(

    title="Manufacturing AI Intelligence",

    css=CSS

) as demo:


    # ========================================================
    # Header
    # ========================================================

    gr.HTML(
        """
        <div class="main-title">
        🏭 Manufacturing AI Intelligence
        </div>

        <div class="subtitle">
        Product → Profit → Inventory → Channel → KMeans → AI Agent
        </div>
        """
    )


    # ========================================================
    # TAB 1｜LOAD
    # ========================================================

    with gr.Tab(
        "📥 1｜資料載入"
    ):


        gr.Markdown(
            """
### 使用方式

**第一份 Excel｜必要**

`生活家電製造商_232SKU_加入成本價格主檔.xlsx`

解鎖：

- Product
- Profit
- Inventory
- Product Agent


**第二份 Excel｜可稍後加入**

`Day1_Lab2_Smart_Home_Appliance_Product_Channel_Mapping_v2.xlsx`

再解鎖：

- Channel
- Global KMeans
- 各產品 KMeans
- Cluster Agent
            """
        )


        with gr.Row():


            product_file = gr.File(

                label="① Product Master",

                file_types=[
                    ".xlsx",
                    ".xls"
                ],

                type="filepath"
            )


            channel_file = gr.File(

                label="② Product × Channel｜可選",

                file_types=[
                    ".xlsx",
                    ".xls"
                ],

                type="filepath"
            )


        load_btn = gr.Button(

            "🚀 建立 / 更新 Manufacturing AI",

            variant="primary"
        )


        load_status = gr.Markdown(
            "### ⏳ 尚未載入資料"
        )


    # ========================================================
    # TAB 2｜Dashboard
    # ========================================================

    with gr.Tab(
        "🏠 2｜Dashboard"
    ):


        home_cards = gr.HTML(

            make_cards([
                (
                    "資料狀態",
                    "等待",
                    "請先上傳 Excel"
                )
            ])
        )


        with gr.Row():

            home_product = gr.HTML(
                blank_html()
            )

            home_profit = gr.HTML(
                blank_html()
            )


        with gr.Row():

            home_inventory = gr.HTML(
                blank_html()
            )

            home_lifecycle = gr.HTML(
                blank_html()
            )


    # ========================================================
    # TAB 3｜Product
    # ========================================================

    with gr.Tab(
        "📦 3｜Product"
    ):


        with gr.Row():


            p_product = gr.Dropdown(

                PRODUCTS,

                value="全部",

                label="產品別"
            )


            p_tier = gr.Dropdown(

                TIERS,

                value="全部",

                label="產品定位"
            )


            p_mode = gr.Dropdown(

                MODES,

                value="全部",

                label="業務模式"
            )


            p_life = gr.Dropdown(

                LIFECYCLES,

                value="全部",

                label="生命週期"
            )


        p_btn = gr.Button(
            "分析 Product"
        )


        p_summary = gr.Markdown()

        p_chart = gr.HTML(
            blank_html()
        )

        p_table = gr.Dataframe(
            interactive=False
        )


    # ========================================================
    # TAB 4｜Profit
    # ========================================================

    with gr.Tab(
        "💰 4｜Profit"
    ):


        with gr.Row():


            profit_product = gr.Dropdown(

                PRODUCTS,

                value="全部",

                label="產品別"
            )


            profit_mode = gr.Dropdown(

                MODES,

                value="全部",

                label="業務模式"
            )


        profit_btn = gr.Button(
            "分析 Profit"
        )


        profit_summary = gr.Markdown()

        profit_chart = gr.HTML(
            blank_html()
        )

        profit_table = gr.Dataframe(
            interactive=False
        )


    # ========================================================
    # TAB 5｜Inventory
    # ========================================================

    with gr.Tab(
        "📦 5｜Inventory"
    ):


        with gr.Row():


            inv_product = gr.Dropdown(

                PRODUCTS,

                value="全部",

                label="產品別"
            )


            inv_risk = gr.Dropdown(

                RISKS,

                value="全部",

                label="庫存風險"
            )


        inv_btn = gr.Button(
            "分析 Inventory"
        )


        inv_summary = gr.Markdown()

        inv_chart = gr.HTML(
            blank_html()
        )

        inv_table = gr.Dataframe(
            interactive=False
        )


    # ========================================================
    # TAB 6｜Channel
    # ========================================================

    with gr.Tab(
        "🏬 6｜Channel"
    ):


        gr.Markdown(
            "第二份 Product × Channel Excel 載入後自動解鎖。"
        )


        with gr.Row():


            ch_product = gr.Dropdown(

                PRODUCTS,

                value="全部",

                label="產品別"
            )


            ch_mode = gr.Dropdown(

                MODES,

                value="全部",

                label="業務模式"
            )


            ch_market = gr.Dropdown(

                [
                    "全部",
                    "國內",
                    "外銷"
                ],

                value="全部",

                label="市場"
            )


        ch_btn = gr.Button(
            "分析 Channel"
        )


        ch_summary = gr.Markdown()

        ch_chart = gr.HTML(
            blank_html(
                "Channel 尚未解鎖"
            )
        )

        ch_table = gr.Dataframe(
            interactive=False
        )


    # ========================================================
    # TAB 7｜Global KMeans
    # ========================================================

    with gr.Tab(
        "🧠 7｜Global KMeans"
    ):


        km_status = gr.Markdown(
            "### 🔒 KMeans 尚未解鎖"
        )


        km_chart = gr.HTML(
            blank_html(
                "請加入第二份 Excel"
            )
        )


        gr.Markdown(
            "### Cluster Profile"
        )


        km_profile = gr.Dataframe(
            interactive=False
        )


        gr.Markdown(
            "### K Selection｜Silhouette Score"
        )


        km_scores = gr.Dataframe(
            interactive=False
        )


    # ========================================================
    # TAB 8｜產品 KMeans
    # ========================================================

    with gr.Tab(
        "🔬 8｜各產品分群"
    ):


        km_product = gr.Dropdown(

            PRODUCTS[1:],

            value="咖啡機",

            label="產品別"
        )


        km_product_btn = gr.Button(

            "🧠 執行產品 KMeans",

            variant="primary"
        )


        km_product_summary = gr.Markdown()


        km_product_chart = gr.HTML(
            blank_html()
        )


        km_product_profile = gr.Dataframe(
            label="Cluster Profile",
            interactive=False
        )


        km_product_table = gr.Dataframe(
            label="SKU Cluster",
            interactive=False
        )


    # ========================================================
    # TAB 9｜Agent
    # ========================================================

    with gr.Tab(
        "🤖 9｜AI Agent"
    ):


        gr.Markdown(
            """
### 可以試著問

`有哪些高階咖啡機？`

`咖啡機毛利率低於20%的SKU？`

`有哪些停產但仍有庫存？`

`列出高風險呆滯庫存`

`咖啡機目前有哪些在售通路？`

`哪些產品有外銷？`

`KMeans各群有什麼差異？`

`咖啡機分群有什麼特色？`
            """
        )


        question = gr.Textbox(

            label="管理問題",

            lines=2,

            placeholder=(
                "例如："
                "咖啡機目前有哪些在售通路？"
            )
        )


        use_local = gr.Checkbox(

            value=False,

            label="使用 Local Qwen3-0.6B 幫我重新解釋"
        )


        ask_btn = gr.Button(

            "🤖 Agent Analyze",

            variant="primary"
        )


        agent_answer = gr.Markdown()


        agent_chart = gr.HTML(
            blank_html()
        )


        agent_table = gr.Dataframe(

            label="Evidence｜資料證據",

            interactive=False
        )


    # ========================================================
    # TAB 10｜Local Qwen
    # ========================================================

    with gr.Tab(
        "🧠 10｜Local Qwen｜選配"
    ):


        gr.Markdown(
            """
### Qwen3-0.6B

這個模型不是拿來計算 KMeans。

它的角色是：

**Python 算 → Qwen 解釋**

所以即使不啟動 Local AI：

- Product 可以用
- Profit 可以用
- Inventory 可以用
- Channel 可以用
- KMeans 可以用
- Agent 也可以用

第一次按啟動時需要下載模型。
            """
        )


        local_btn = gr.Button(

            "⬇️ 下載並啟動 Local Qwen3-0.6B"
        )


        local_status = gr.Markdown(
            "### ⏳ Local AI 尚未啟動"
        )


    # ========================================================
    # EVENTS
    # ========================================================

    load_btn.click(

        fn=load_project,

        inputs=[
            product_file,
            channel_file
        ],

        outputs=[
            load_status,
            home_cards,
            home_product,
            home_profit,
            home_inventory,
            home_lifecycle,
            km_status,
            km_chart,
            km_profile,
            km_scores
        ]
    )


    p_btn.click(

        fn=product_tool,

        inputs=[
            p_product,
            p_tier,
            p_mode,
            p_life
        ],

        outputs=[
            p_summary,
            p_table,
            p_chart
        ]
    )


    profit_btn.click(

        fn=profit_tool,

        inputs=[
            profit_product,
            profit_mode
        ],

        outputs=[
            profit_summary,
            profit_table,
            profit_chart
        ]
    )


    inv_btn.click(

        fn=inventory_tool,

        inputs=[
            inv_product,
            inv_risk
        ],

        outputs=[
            inv_summary,
            inv_table,
            inv_chart
        ]
    )


    ch_btn.click(

        fn=channel_tool,

        inputs=[
            ch_product,
            ch_mode,
            ch_market
        ],

        outputs=[
            ch_summary,
            ch_table,
            ch_chart
        ]
    )


    km_product_btn.click(

        fn=product_kmeans_tool,

        inputs=[
            km_product
        ],

        outputs=[
            km_product_summary,
            km_product_profile,
            km_product_chart,
            km_product_table
        ]
    )


    ask_btn.click(

        fn=ask_agent,

        inputs=[
            question,
            use_local
        ],

        outputs=[
            agent_answer,
            agent_table,
            agent_chart
        ]
    )


    question.submit(

        fn=ask_agent,

        inputs=[
            question,
            use_local
        ],

        outputs=[
            agent_answer,
            agent_table,
            agent_chart
        ]
    )


    local_btn.click(

        fn=load_local_ai,

        inputs=[],

        outputs=[
            local_status
        ]
    )


# ============================================================
# 22｜Launch
# ============================================================

print("==============================================")
print("🏭 Manufacturing AI Intelligence")
print("==============================================")
print("✅ Product")
print("✅ Profit")
print("✅ Inventory")
print("✅ Channel")
print("✅ Global KMeans")
print("✅ Product KMeans")
print("✅ Agent")
print("✅ Local Qwen3-0.6B Optional")
print("==============================================")
print("🚀 正在啟動 Web App...")


demo.launch(
    share=True,
    debug=False,
    show_error=True
)
