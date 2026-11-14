
import html
import json
import re
import sqlite3
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

PROGRAM_DB = DATA / "boatrace.db"
TRAIN_DB = DATA / "train_dataset_v1.db"
MODEL_FILE = MODELS / "phase08_1_b_model.txt"
REPORT_FILE = MODELS / "phase08_1_b_report.json"

VENUES = {
    1: "桐生", 2: "戸田", 3: "江戸川",
    4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津",
    10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀",
    16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋",
    22: "福岡", 23: "唐津", 24: "大村",
}

BOAT_COLORS = {
    1: ("#FFFFFF", "#111827"),
    2: ("#111827", "#FFFFFF"),
    3: ("#EF4444", "#FFFFFF"),
    4: ("#2563EB", "#FFFFFF"),
    5: ("#FACC15", "#111827"),
    6: ("#16A34A", "#FFFFFF"),
}

GROUP_NAMES = {
    "p": "番組表",
    "f0": "選手の過去成績",
    "f1": "コース・当地成績",
    "f2": "艇番別成績",
    "f3": "モーター・ボート",
    "f4": "競艇場の傾向",
    "f5": "直近成績",
    "f6": "今節成績",
}

FEATURE_NAMES = {
    "national_win_rate": "全国勝率",
    "national_2rate": "全国2連対率",
    "local_win_rate": "当地勝率",
    "local_2rate": "当地2連対率",
    "avg_st": "平均ST",
    "grade": "級別",
    "win_rate": "1着率",
    "top2_rate": "2着以内率",
    "top3_rate": "3着以内率",
    "avg_finish": "平均着順",
    "avg_st": "平均ST",
    "st_std": "STのばらつき",
    "starts": "出走数",
    "records": "記録数",
    "valid_finishes": "有効着順数",
    "course_win_rate": "進入コース別1着率",
    "lane_win_rate": "艇番別1着率",
    "lane_top3_rate": "艇番別3着以内率",
    "motor_win_rate": "モーター1着率",
    "motor_top3_rate": "モーター3着以内率",
    "boat_win_rate": "ボート1着率",
    "boat_top3_rate": "ボート3着以内率",
    "meet_day": "開催日目",
    "meet_avg_st": "今節平均ST",
    "meet_top3_rate": "今節3着以内率",
    "days_since_last_race": "前走からの日数",
}

st.set_page_config(
    page_title="BOAT RACE AI | 新ダッシュボード",
    page_icon="🚤",
    layout="wide",
)


def style():
    st.markdown("""
    <style>
    .stApp {
        background-color: #F2F6FB;
        color: #172A46;
    }
    [data-testid="stSidebar"] {
        background-color: #10233D;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }
    h1, h2, h3 {
        color: #152C49;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        padding: 16px;
        border: 1px solid #E0E8F1;
        border-radius: 12px;
    }
    div[data-testid="stMetricValue"] {
        color: #087AB8;
    }
    .boat-row {
        display: flex;
        align-items: center;
        gap: 12px;
        background: #FFFFFF;
        padding: 11px 14px;
        border: 1px solid #E3EAF3;
        border-radius: 10px;
        margin-bottom: 7px;
    }
    .boat-number {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 7px;
        font-weight: bold;
        flex-shrink: 0;
        border: 1px solid #CBD5E1;
    }
    .boat-name {
        flex: 1;
        font-weight: 600;
        min-width: 0;
    }
    .boat-details {
        color: #52647A;
        font-size: 13px;
        text-align: right;
    }
    @media (max-width: 640px) {
        .boat-row {
            flex-wrap: wrap;
        }
        .boat-details {
            width: 100%;
            text-align: left;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def connect(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    con = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True,
        timeout=30
    )
    con.execute("PRAGMA query_only=ON")
    return con


@st.cache_resource
def load_model():
    return lgb.Booster(
        model_file=str(MODEL_FILE)
    )


@st.cache_data
def load_report():
    return json.loads(
        REPORT_FILE.read_text(encoding="utf-8")
    )


@st.cache_data
def get_dates():
    with connect(TRAIN_DB) as con:
        rows = con.execute("""
            SELECT DISTINCT race_date
            FROM train_races
            WHERE race_date BETWEEN
                '2024-01-01' AND '2024-12-31'
            ORDER BY race_date DESC
        """).fetchall()
    return [r[0] for r in rows]


@st.cache_data
def get_venues(day):
    with connect(TRAIN_DB) as con:
        rows = con.execute("""
            SELECT DISTINCT venue_code
            FROM train_races
            WHERE race_date = ?
            ORDER BY venue_code
        """, (day,)).fetchall()
    return [int(r[0]) for r in rows]


@st.cache_data
def get_races(day, venue):
    with connect(TRAIN_DB) as con:
        rows = con.execute("""
            SELECT race_no
            FROM train_races
            WHERE race_date = ?
              AND venue_code = ?
            ORDER BY race_no
        """, (day, venue)).fetchall()
    return [int(r[0]) for r in rows]


@st.cache_data(max_entries=50)
def load_entries(day, venue, race):
    with connect(PROGRAM_DB) as con:
        return pd.read_sql_query("""
            SELECT *
            FROM program_entries
            WHERE race_date = ?
              AND venue_code = ?
              AND race_no = ?
            ORDER BY lane
        """, con, params=(day, venue, race))


@st.cache_data(max_entries=50)
def load_race(day, venue, race, feature_tuple):
    cols = ", ".join(
        [quote("label")] +
        [quote(f) for f in feature_tuple]
    )
    with connect(TRAIN_DB) as con:
        rows = con.execute(f"""
            SELECT {cols}
            FROM train_races
            WHERE race_date = ?
              AND venue_code = ?
              AND race_no = ?
        """, (day, venue, race)).fetchall()

    if len(rows) != 1:
        raise RuntimeError(
            f"レースが一意に取得できません: {len(rows)}件"
        )

    actual = str(rows[0][0])
    x = np.asarray([
        np.nan if v is None else float(v)
        for v in rows[0][1:]
    ], dtype=np.float32)

    x[~np.isfinite(x)] = np.nan
    return actual, x


def japanese_feature(name):
    match = re.match(
        r"^b([1-6])_(p|f[0-6])_(.+)$",
        name
    )
    if not match:
        return name

    boat, group, field = match.groups()

    # 期間の表記を日本語にする
    period = ""
    m = re.search(r"_(\\d+)d$", field)
    if m:
        period = f"・過去{m.group(1)}日"
        field = field[:m.start()]

    m = re.search(r"^recent_(\\d+)_(.+)$", field)
    if m:
        period = f"・直近{m.group(1)}走"
        field = m.group(2)

    label = FEATURE_NAMES.get(field)

    if label is None:
        # 未登録項目は英語を残す
        label = field.replace("_", " ")

    return (
        f"{boat}号艇｜"
        f"{GROUP_NAMES[group]}｜"
        f"{label}{period}"
    )


def number_text(value, digits=2):
    try:
        v = float(value)
        if not np.isfinite(v):
            return "—"
        return f"{v:.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def show_boats(entries):
    st.subheader("出走表・6艇比較")

    if len(entries) != 6:
        st.warning(
            f"出走表は{len(entries)}艇分です。"
        )

    for _, row in entries.iterrows():
        lane = int(row["lane"])
        bg, fg = BOAT_COLORS[lane]

        name = row.get(
            "racer_name",
            row.get("name", "")
        )
        if pd.isna(name) or not str(name).strip():
            name = f"登録番号 {row.get('racer_id', '不明')}"

        name = html.escape(str(name))

        grade = html.escape(
            str(row.get("grade", ""))
        )

        national = number_text(
            row.get("national_win_rate")
        )
        motor = number_text(
            row.get("motor_2rate")
        )

        st.markdown(f"""
        <div class="boat-row">
          <span class="boat-number"
                style="background:{bg};color:{fg}">
            {lane}
          </span>
          <span class="boat-name">{name}</span>
          <span class="boat-details">
            {grade}｜
            全国勝率 {national}｜
            モーター2連対率 {motor}
          </span>
        </div>
        """, unsafe_allow_html=True)

    options = {
        "全国勝率": "national_win_rate",
        "当地勝率": "local_win_rate",
        "モーター2連対率": "motor_2rate",
        "ボート2連対率": "boat_2rate",
    }

    available = {
        label: col
        for label, col in options.items()
        if col in entries.columns
    }

    if available:
        selected = st.selectbox(
            "6艇を比較する項目",
            list(available.keys())
        )
        col = available[selected]

        chart = pd.DataFrame({
            "艇番": (
                entries["lane"].astype(str) + "号艇"
            ),
            selected: pd.to_numeric(
                entries[col], errors="coerce"
            )
        }).set_index("艇番")

        st.bar_chart(chart)


def show_prediction(model, report, day, venue, race):
    features = report["feature_names"]
    classes = report["class_order"]

    actual, x = load_race(
        day, venue, race, tuple(features)
    )

    probabilities = np.asarray(
        model.predict(
            x.reshape(1, -1),
            num_threads=4
        )
    ).reshape(-1)

    if len(probabilities) != 120:
        raise RuntimeError(
            "予測結果が120通りではありません"
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "予測確率に不正な値があります"
        )

    result = pd.DataFrame({
        "3連単": list(map(str, classes)),
        "予測確率": probabilities
    }).sort_values(
        "予測確率", ascending=False
    ).reset_index(drop=True)

    result.insert(
        0, "順位",
        np.arange(1, len(result) + 1)
    )

    st.subheader("3連単AI予測")

    first = result.iloc[0]
    top3_sum = result.head(3)["予測確率"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("AI本命", first["3連単"])
    c2.metric(
        "本命の予測確率",
        f"{first['予測確率'] * 100:.2f}%"
    )
    c3.metric(
        "上位3点の確率合計",
        f"{top3_sum * 100:.2f}%"
    )

    count = st.slider(
        "ランキング表示件数",
        min_value=3,
        max_value=120,
        value=15
    )

    shown = result.head(count).copy()
    shown["予測確率 (%)"] = (
        shown["予測確率"] * 100
    ).round(3)

    st.dataframe(
        shown[
            ["順位", "3連単", "予測確率 (%)"]
        ],
        hide_index=True,
        use_container_width=True
    )

    chart = shown.head(15).set_index(
        "3連単"
    )["予測確率 (%)"]
    st.bar_chart(chart)

    st.markdown("#### レース結果")
    matched = result[
        result["3連単"] == actual
    ]

    if not matched.empty:
        rank = int(matched.iloc[0]["順位"])
        st.write(
            f"実際の3連単：**{actual}** "
            f"（AI予測 {rank}位）"
        )
        if rank == 1:
            st.success("Top1的中")
        elif rank <= 3:
            st.success("Top3的中")
        else:
            st.info("Top3には含まれていません")
    else:
        st.warning(
            "実際の結果が予測クラスにありません"
        )


def show_importance(model, report):
    st.subheader("AIが重視した特徴量")

    names = report["feature_names"]
    gain = model.feature_importance(
        importance_type="gain"
    )

    df = pd.DataFrame({
        "特徴量（日本語）": [
            japanese_feature(n) for n in names
        ],
        "元の特徴量名": names,
        "重要度": gain,
    })

    df = df.sort_values(
        "重要度", ascending=False
    ).reset_index(drop=True)

    df.insert(
        0, "順位",
        np.arange(1, len(df) + 1)
    )

    st.dataframe(
        df.head(30),
        hide_index=True,
        use_container_width=True
    )

    st.bar_chart(
        df.head(12).set_index(
            "特徴量（日本語）"
        )["重要度"]
    )

    st.caption(
        "日本語名は表示専用です。"
        "モデルの内部特徴量名は変更していません。"
        "Gain重要度はモデル全体の値であり、"
        "このレース固有の寄与度ではありません。"
    )


def main():
    style()

    st.title("🚤 BOAT RACE AI")
    st.caption(
        "WEB-5 | 競艇3連単AI 分析ダッシュボード"
    )

    st.info(
        "2024年の過去レース専用。"
        "現在の安定モデル8.1-Bを使用。"
        "AI本体・DBへの書き込みは行いません。"
    )

    try:
        report = load_report()
        model = load_model()

        features = report["feature_names"]
        classes = report["class_order"]

        if len(features) != 618 or len(classes) != 120:
            raise RuntimeError("モデル情報が不正")

        if model.feature_name() != features:
            raise RuntimeError(
                "特徴量の順序がモデルと一致しません"
            )

        dates = get_dates()
        if not dates:
            st.warning("2024年のレースがありません")
            return

    except Exception as exc:
        st.error(f"初期化エラー：{exc}")
        return

    metrics = report["lightgbm"]

    a, b, c = st.columns(3)
    a.metric(
        "Top1的中率",
        f"{metrics['top1'] * 100:.2f}%"
    )
    b.metric(
        "Top3的中率",
        f"{metrics['top3'] * 100:.2f}%"
    )
    c.metric(
        "Log Loss",
        f"{metrics['logloss']:.6f}"
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        day = st.selectbox("開催日", dates)

    venues = get_venues(day)

    with c2:
        venue = st.selectbox(
            "競艇場",
            venues,
            format_func=lambda v:
                f"{VENUES.get(v, str(v))} ({v:02d})"
        )

    races = get_races(day, venue)

    with c3:
        race = st.selectbox(
            "レース番号",
            races,
            format_func=lambda r: f"{r}R"
        )

    if st.button(
        "レース分析を表示",
        type="primary",
        use_container_width=True
    ):
        st.session_state["web5_selection"] = (
            day, venue, race
        )

    selection = st.session_state.get(
        "web5_selection"
    )

    if selection is None:
        st.write(
            "開催日・競艇場・レースを選択して実行してください。"
        )
        return

    day, venue, race = selection

    st.divider()
    st.header(
        f"{day} | {VENUES.get(venue, venue)} {race}R"
    )

    try:
        entries = load_entries(day, venue, race)
        if entries.empty:
            st.warning("出走表がありません")
            return

        tab1, tab2, tab3 = st.tabs([
            "出走表・選手比較",
            "3連単AI予測",
            "特徴量ランキング",
        ])

        with tab1:
            show_boats(entries)

        with tab2:
            show_prediction(
                model, report, day, venue, race
            )

        with tab3:
            show_importance(model, report)

    except Exception as exc:
        st.error(f"表示エラー：{exc}")

    st.caption(
        "予測確率は未校正です。"
        "過去の的中率は将来の利益を保証しません。"
    )


if __name__ == "__main__":
    main()
