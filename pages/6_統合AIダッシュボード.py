
import html
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st


# =====================================================
# BOAT RACE AI - WEB 6
# 統合ダッシュボード / 2024年検証専用 / READ ONLY
# =====================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

PROGRAM_DB = DATA / "boatrace.db"
TRAIN_DB = DATA / "train_dataset_v1.db"
MODEL_PATH = MODELS / "phase08_1_b_model.txt"
REPORT_PATH = MODELS / "phase08_1_b_report.json"

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

# filename, table, 艇番を表す列, 日本語名
FEATURE_DBS = {
    "F0": (
        "features_fast.db",
        "racer_features_fast",
        "boat_no",
        "選手の過去成績",
    ),
    "F1": (
        "features_fast.db",
        "racer_course_features_fast",
        "lane",
        "コース・当地成績",
    ),
    "F2": (
        "lane_features.db",
        "racer_lane_features_fast",
        "lane",
        "艇番別成績",
    ),
    "F3": (
        "motor_features.db",
        "equipment_features_fast",
        "lane",
        "モーター・ボート成績",
    ),
    "F4": (
        "venue_detail_features.db",
        "venue_detail_features_fast",
        "lane",
        "競艇場・コース傾向",
    ),
    "F5": (
        "recent_form_features.db",
        "racer_recent_form_fast",
        "lane",
        "直近5・10・20走",
    ),
    "F6": (
        "meet_features.db",
        "racer_meet_features",
        "lane",
        "今節成績・ST",
    ),
}

GROUP_NAMES = {
    "p": "番組表",
    **{
        group.lower(): info[3]
        for group, info in FEATURE_DBS.items()
    },
}

TERMS = {
    "national": "全国",
    "local": "当地",
    "venue": "競艇場",
    "course": "進入コース",
    "lane": "艇番",
    "racer": "選手",
    "motor": "モーター",
    "boat": "ボート",
    "win": "1着",
    "top2": "2着以内",
    "top3": "3着以内",
    "rate": "率",
    "starts": "出走数",
    "records": "記録数",
    "valid": "有効",
    "finishes": "着順",
    "finish": "着順",
    "avg": "平均",
    "st": "ST",
    "std": "標準偏差",
    "trend": "変化",
    "meet": "今節",
    "day": "日目",
    "days": "日数",
    "since": "経過",
    "last": "前回",
    "race": "レース",
    "count": "件数",
    "change": "変更",
    "partial": "一部欠損",
    "f": "F",
    "l": "L",
}

KEYS = [
    "race_date", "venue_code",
    "race_no", "lane", "racer_id"
]

st.set_page_config(
    page_title="BOAT RACE AI | 統合版",
    page_icon="🚤",
    layout="wide",
)


# -----------------------------
# デザイン
# -----------------------------

def apply_style():
    st.markdown("""
    <style>
    .stApp {
        background: #F3F7FC;
        color: #152B48;
    }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #DFE8F2;
        border-radius: 12px;
        padding: 15px;
    }
    .web6-header {
        background: linear-gradient(
            110deg, #0B1C36, #173C63
        );
        color: white;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 16px;
    }
    .web6-header h2 {
        color: white !important;
        margin: 0;
    }
    .web6-header p {
        color: #B9D3EF;
        margin: 6px 0 0 0;
    }
    .boat-row {
        display: flex;
        align-items: center;
        gap: 12px;
        background: white;
        color: #14263F;
        border: 1px solid #E0E8F2;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
    }
    .boat-number {
        display: inline-flex;
        justify-content: center;
        align-items: center;
        width: 36px;
        height: 36px;
        border-radius: 6px;
        border: 1px solid #CBD5E1;
        font-weight: 700;
        flex-shrink: 0;
    }
    .boat-name {
        font-weight: 600;
        flex: 1;
    }
    .boat-stats {
        color: #51647A;
        font-size: 13px;
    }
    @media (max-width: 680px) {
        .boat-row { flex-wrap: wrap; }
        .boat-stats { width: 100%; }
    }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------
# データの読み込み
# -----------------------------

def q(name):
    return '"' + name.replace('"', '""') + '"'


def connect(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    con = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True,
        timeout=30,
    )
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA busy_timeout=30000")
    return con


@st.cache_resource
def load_model():
    return lgb.Booster(
        model_file=str(MODEL_PATH)
    )


@st.cache_data
def load_report():
    return json.loads(
        REPORT_PATH.read_text(encoding="utf-8")
    )


@st.cache_data(ttl=600)
def dates():
    with closing(connect(TRAIN_DB)) as con:
        rows = con.execute("""
            SELECT DISTINCT race_date
            FROM train_races
            WHERE race_date BETWEEN
                  '2024-01-01' AND '2024-12-31'
            ORDER BY race_date DESC
        """).fetchall()
    return [r[0] for r in rows]


@st.cache_data(ttl=600)
def venues(day):
    with closing(connect(TRAIN_DB)) as con:
        rows = con.execute("""
            SELECT DISTINCT venue_code
            FROM train_races
            WHERE race_date = ?
            ORDER BY venue_code
        """, (day,)).fetchall()
    return [int(r[0]) for r in rows]


@st.cache_data(ttl=600)
def races(day, venue):
    with closing(connect(TRAIN_DB)) as con:
        rows = con.execute("""
            SELECT race_no
            FROM train_races
            WHERE race_date = ?
              AND venue_code = ?
            ORDER BY race_no
        """, (day, venue)).fetchall()
    return [int(r[0]) for r in rows]


@st.cache_data(ttl=600, max_entries=60)
def program(day, venue, race):
    with closing(connect(PROGRAM_DB)) as con:
        return pd.read_sql_query("""
            SELECT *
            FROM program_entries
            WHERE race_date = ?
              AND venue_code = ?
              AND race_no = ?
            ORDER BY lane
        """, con, params=(day, venue, race))


@st.cache_data(ttl=600, max_entries=60)
def race_features(day, venue, race, feature_names):
    cols = ", ".join(
        [q("label")] +
        [q(c) for c in feature_names]
    )

    with closing(connect(TRAIN_DB)) as con:
        rows = con.execute(f"""
            SELECT {cols}
            FROM train_races
            WHERE race_date = ?
              AND venue_code = ?
              AND race_no = ?
        """, (day, venue, race)).fetchall()

    if len(rows) != 1:
        raise RuntimeError(
            f"学習データのレース数が不正: {len(rows)}"
        )

    actual = str(rows[0][0])

    x = np.array(
        [
            np.nan if v is None else float(v)
            for v in rows[0][1:]
        ],
        dtype=np.float32,
    )
    x[~np.isfinite(x)] = np.nan

    return actual, x


@st.cache_data(ttl=600, max_entries=140)
def group_features(group, day, venue, race):
    filename, table, lane_col, _ = FEATURE_DBS[group]

    with closing(connect(DATA / filename)) as con:
        df = pd.read_sql_query(
            f"""
            SELECT *
            FROM {q(table)}
            WHERE race_date = ?
              AND venue_code = ?
              AND race_no = ?
            ORDER BY {q(lane_col)}
            """,
            con,
            params=(day, venue, race),
        )

    if lane_col != "lane":
        df = df.rename(
            columns={lane_col: "lane"}
        )

    return df


# -----------------------------
# 特徴量の日本語名
# -----------------------------

def translate_field(field):
    period = ""

    match = re.search(r"_([0-9]+)d$", field)
    if match:
        period = f"・過去{match.group(1)}日"
        field = field[:match.start()]

    match = re.match(
        r"recent_([0-9]+)_(.+)", field
    )
    if match:
        period = f"・直近{match.group(1)}走"
        field = match.group(2)

    exact = {
        "national_win_rate": "全国勝率",
        "local_win_rate": "当地勝率",
        "avg_st": "平均ST",
        "days_since_last_race": "前走からの日数",
        "meet_day": "開催日目",
        "meet_st_partial": "今節STの一部欠損",
    }

    if field in exact:
        return exact[field] + period

    words = field.split("_")
    translated = [
        TERMS.get(word, word) for word in words
    ]

    return "・".join(translated) + period


def japanese_feature(name):
    match = re.match(
        r"^b([1-6])_(p|f[0-6])_(.+)$",
        name,
    )

    if not match:
        return name

    boat, group, field = match.groups()

    return (
        f"{boat}号艇｜{GROUP_NAMES[group]}｜"
        f"{translate_field(field)}"
    )


# -----------------------------
# 出走表
# -----------------------------

def safe_number(value, digits=2):
    try:
        number = float(value)
        if np.isfinite(number):
            return f"{number:.{digits}f}"
    except (TypeError, ValueError):
        pass
    return "—"


def show_boats(df):
    st.subheader("🚤 出走表")

    if len(df) != 6:
        st.warning(
            f"出走表の行数が6ではありません: {len(df)}"
        )

    for _, row in df.sort_values("lane").iterrows():
        lane = int(row["lane"])
        bg, fg = BOAT_COLORS[lane]

        name = row.get(
            "racer_name",
            row.get("name", "")
        )

        if pd.isna(name) or not str(name).strip():
            name = f"登録番号 {row['racer_id']}"

        name = html.escape(str(name))
        grade = html.escape(str(row.get("grade", "")))

        win = safe_number(
            row.get("national_win_rate")
        )
        motor = safe_number(
            row.get("motor_2rate")
        )

        st.markdown(f"""
        <div class="boat-row">
          <div class="boat-number"
               style="background:{bg};color:{fg}">
            {lane}
          </div>
          <span class="boat-name">{name}</span>
          <span class="boat-stats">
            {grade} ｜ 全国勝率 {win}
            ｜ モーター2連対率 {motor}
          </span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("出走表の詳細数値"):
        selected = [
            c for c in [
                "lane", "racer_id", "grade",
                "national_win_rate",
                "local_win_rate", "avg_st",
                "motor_no", "motor_2rate",
                "boat_no", "boat_2rate",
            ]
            if c in df.columns
        ]
        st.dataframe(
            df[selected],
            hide_index=True,
            use_container_width=True,
        )


# -----------------------------
# 予測
# -----------------------------

def predict(model, report, day, venue, race):
    actual, x = race_features(
        day, venue, race,
        tuple(report["feature_names"])
    )

    probabilities = np.asarray(
        model.predict(
            x.reshape(1, -1),
            num_threads=4,
        )
    ).reshape(-1)

    classes = report["class_order"]

    if (
        len(probabilities) != 120
        or not np.isfinite(probabilities).all()
    ):
        raise RuntimeError("予測確率が不正です")

    df = pd.DataFrame({
        "3連単": [str(c) for c in classes],
        "予測確率": probabilities,
    }).sort_values(
        "予測確率", ascending=False
    ).reset_index(drop=True)

    df.insert(
        0, "順位",
        np.arange(1, len(df) + 1)
    )

    return actual, df


def show_prediction(actual, predictions):
    st.subheader("🎯 3連単AI予測")

    top = predictions.iloc[0]
    top3_sum = predictions.head(3)[
        "予測確率"
    ].sum()

    c1, c2, c3 = st.columns(3)

    c1.metric("AI本命", top["3連単"])
    c2.metric(
        "本命確率",
        f"{top['予測確率'] * 100:.2f}%"
    )
    c3.metric(
        "TOP3確率合計",
        f"{top3_sum * 100:.2f}%"
    )

    count = st.slider(
        "ランキングの表示件数",
        3, 120, 15,
        key="web6_rank_count",
    )

    df = predictions.head(count).copy()

    df["予測確率 (%)"] = (
        df["予測確率"] * 100
    ).round(3)

    st.dataframe(
        df[["順位", "3連単", "予測確率 (%)"]],
        hide_index=True,
        use_container_width=True,
    )

    st.bar_chart(
        df.head(15).set_index(
            "3連単"
        )["予測確率 (%)"]
    )

    st.subheader("実際のレース結果")

    result = predictions[
        predictions["3連単"] == actual
    ]

    if result.empty:
        st.warning(
            f"実際の結果 {actual} が見つかりません"
        )
        return

    rank = int(result.iloc[0]["順位"])
    st.write(
        f"実際の3連単：**{actual}** "
        f"｜ 予測順位：**{rank}位**"
    )

    if rank == 1:
        st.success("Top1 的中")
    elif rank <= 3:
        st.success("Top3 的中")
    else:
        st.info("Top3には含まれていません")


# -----------------------------
# 選手・モーター比較
# -----------------------------


def show_comparison(df, key_suffix="main"):
    st.subheader("📊 6艇の成績比較")

    candidates = {
        "全国勝率": ("national_win_rate", "max"),
        "当地勝率": ("local_win_rate", "max"),
        "モーター2連対率": ("motor_2rate", "max"),
        "ボート2連対率": ("boat_2rate", "max"),
        "平均ST": ("avg_st", "min"),
    }

    available = {
        label: (col, direction)
        for label, (col, direction) in candidates.items()
        if col in df.columns
    }

    if not available:
        st.warning("比較できる数値がありません")
        return

    # 6艇の全指標を表にまとめる
    table = pd.DataFrame(
        index=df["lane"].astype(str) + "号艇"
    )
    table.index.name = "艇番"

    for label, (col, _) in available.items():
        table[label] = pd.to_numeric(
            df[col], errors="coerce"
        ).to_numpy()

    # 各指標の最高値（STは最低値）を強調
    def highlight_best(column):
        direction = available[column.name][1]
        valid = column.dropna()

        if valid.empty:
            return [""] * len(column)

        best = (
            valid.max()
            if direction == "max"
            else valid.min()
        )

        return [
            (
                "background-color: #BBF7D0;"
                "color: #14532D;"
                "font-weight: bold;"
            )
            if pd.notna(value) and np.isclose(
                value, best, rtol=1e-9, atol=1e-9
            )
            else ""
            for value in column
        ]

    styled = (
        table.style
        .apply(highlight_best, axis=0)
        .format(precision=3, na_rep="—")
    )

    st.markdown("#### 6艇の総合比較表")
    st.dataframe(
        styled,
        use_container_width=True
    )

    st.caption(
        "緑色は各項目の6艇中1位です。"
        "同率1位の場合は複数艇を強調します。"
        "平均STは小さい方を強調します。"
    )

    st.divider()

    # 指標を選んでグラフを表示
    metric = st.selectbox(
        "グラフで比較する項目",
        list(available.keys()),
        key=f"web8_metric_{key_suffix}",
    )

    direction = available[metric][1]
    values = table[metric].dropna()

    if values.empty:
        st.info("この指標の有効な数値がありません")
        return

    best = (
        values.max()
        if direction == "max"
        else values.min()
    )

    winners = values.index[
        np.isclose(
            values.to_numpy(),
            best,
            rtol=1e-9,
            atol=1e-9
        )
    ].tolist()

    st.success(
        f"🏆 {metric}の1位："
        f"{'・'.join(winners)} "
        f"（{best:.3f}）"
    )

    st.bar_chart(table[[metric]])

    st.caption(
        "数値は過去レースの保存済みデータです。"
        "指標単独の順位であり、AIの総合予測順位"
        "とは異なります。"
    )


# -----------------------------
# F0〜F6の詳細分析
# -----------------------------

def show_groups(entries, day, venue, race):
    st.subheader("🔬 詳細特徴量分析")

    tabs = st.tabs([
        f"{group} {info[3]}"
        for group, info in FEATURE_DBS.items()
    ])

    for tab, (group, info) in zip(
        tabs, FEATURE_DBS.items()
    ):
        with tab:
            try:
                df = group_features(
                    group, day, venue, race
                )

                if len(df) != 6:
                    raise RuntimeError(
                        f"行数が6ではありません: {len(df)}"
                    )

                if (
                    df.duplicated(KEYS).any()
                    or not df[KEYS].sort_values(
                        KEYS
                    ).reset_index(drop=True).equals(
                        entries[KEYS].sort_values(
                            KEYS
                        ).reset_index(drop=True)
                    )
                ):
                    raise RuntimeError(
                        "出走表と特徴量のキーが不一致"
                    )

                cols = [
                    c for c in df.columns
                    if c not in KEYS
                ]

                numeric = [
                    c for c in cols
                    if pd.to_numeric(
                        df[c], errors="coerce"
                    ).notna().any()
                ]

                if not numeric:
                    st.warning(
                        "数値特徴量がありません"
                    )
                    continue

                chosen = st.selectbox(
                    "比較する特徴量",
                    numeric,
                    format_func=translate_field,
                    key=f"web6_{group}_metric",
                )

                chart = pd.DataFrame({
                    "艇番": (
                        df["lane"].astype(str)
                        + "号艇"
                    ),
                    translate_field(chosen):
                        pd.to_numeric(
                            df[chosen], errors="coerce"
                        ),
                }).set_index("艇番")

                st.bar_chart(chart)

                with st.expander(
                    "全特徴量の一覧を表示"
                ):
                    display = df[
                        ["lane"] + cols
                    ].copy()

                    display.columns = [
                        "艇番"
                    ] + [
                        translate_field(c)
                        for c in cols
                    ]

                    st.dataframe(
                        display,
                        hide_index=True,
                        use_container_width=True,
                    )

                st.caption(
                    f"{group}: {len(cols)}特徴量"
                )

            except Exception as exc:
                st.warning(
                    f"{group}の読み込みに失敗: {exc}"
                )


# -----------------------------
# モデル特徴量重要度
# -----------------------------

def show_importance(model, report):
    st.subheader("🧠 AIの特徴量重要度")

    names = report["feature_names"]
    gains = model.feature_importance(
        importance_type="gain"
    )

    df = pd.DataFrame({
        "特徴量": [
            japanese_feature(n) for n in names
        ],
        "内部名": names,
        "重要度": gains,
    }).sort_values(
        "重要度", ascending=False
    ).reset_index(drop=True)

    df.insert(
        0, "順位",
        np.arange(1, len(df) + 1)
    )

    term = st.text_input(
        "特徴量を検索",
        placeholder="例：モーター / 1号艇 / ST",
        key="web6_search",
    )

    if term:
        mask = (
            df["特徴量"].str.contains(
                term, case=False, regex=False
            )
            | df["内部名"].str.contains(
                term, case=False, regex=False
            )
        )
        df = df.loc[mask]

    st.dataframe(
        df.head(100),
        hide_index=True,
        use_container_width=True,
    )

    if not df.empty:
        st.bar_chart(
            df.head(12).set_index(
                "特徴量"
            )["重要度"]
        )

    st.caption(
        "重要度はモデル全体のGain値です。"
        "選択中のレース固有の寄与度ではありません。"
    )


# -----------------------------
# メインページ
# -----------------------------

def main():
    apply_style()

    st.markdown("""
    <div class="web6-header">
      <h2>🚤 BOAT RACE AI</h2>
      <p>
        3連単予測・選手分析・モーター分析
        を一つにまとめた統合ダッシュボード
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "WEB-6 | 2024年過去レース検証用 "
        "| 安定版モデル8.1-B"
    )

    try:
        report = load_report()
        model = load_model()

        feature_names = report["feature_names"]
        classes = report["class_order"]

        if (
            len(feature_names) != 618
            or len(classes) != 120
            or model.feature_name() != feature_names
        ):
            raise RuntimeError(
                "モデルと特徴量の構造が不一致"
            )

        all_dates = dates()

        if not all_dates:
            raise RuntimeError(
                "2024年のレースがありません"
            )

    except Exception as exc:
        st.error(f"初期化エラー: {exc}")
        st.stop()

    metrics = report["lightgbm"]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Top1 的中率",
        f"{metrics['top1'] * 100:.2f}%"
    )
    c2.metric(
        "Top3 的中率",
        f"{metrics['top3'] * 100:.2f}%"
    )
    c3.metric(
        "Log Loss",
        f"{metrics['logloss']:.6f}"
    )

    st.divider()
    st.subheader("レース選択")


    # 全機能で同じレース選択を使用
    c1, c2, c3 = st.columns(3)

    
    with c1:
        selected_date = st.date_input(
            "開催日",
            value=pd.to_datetime(all_dates[0]).date(),
            min_value=pd.to_datetime(all_dates[-1]).date(),
            max_value=pd.to_datetime(all_dates[0]).date(),
            format="YYYY/MM/DD",
            key="web8_calendar",
        )

        day = selected_date.isoformat()

    if day not in all_dates:
        st.warning(
            "選択された日付には保存済みレースがありません。"
            "別の日付を選択してください。"
        )
        return


    with c2:
        day_venues = venues(day)
        venue = st.selectbox(
            "競艇場",
            day_venues,
            format_func=lambda v:
                f"{VENUES.get(v, v)} ({v:02d})",
            key="web6_venue"
        )

    with c3:
        day_races = races(day, venue)
        race = st.selectbox(
            "レース番号",
            day_races,
            format_func=lambda r: f"{r}R",
            key="web6_race"
        )

    submitted = st.button(
        "このレースを分析",
        type="primary",
        use_container_width=True
    )

    if submitted:
        st.session_state["web6_selected_race"] = (
            day, venue, race
        )

    selected = st.session_state.get(
        "web6_selected_race"
    )

    if selected is None:
        st.info(
            "日付・競艇場・レース番号を選択し、"
            "「このレースを分析」を押してください。"
        )
        return

    day, venue, race = selected

    st.header(
        f"{day}｜{VENUES.get(venue, venue)} {race}R"
    )

    try:
        entries = program(
            day, venue, race
        )

        if len(entries) != 6:
            raise RuntimeError(
                f"出走表が6艇ではありません: {len(entries)}"
            )

        actual, prediction = predict(
            model, report, day, venue, race
        )

    except Exception as exc:
        st.error(f"レース読み込みエラー: {exc}")
        return

    tabs = st.tabs([
        "総合分析",
        "3連単120通り",
        "選手・機材",
        "F0〜F6詳細",
        "AI重要度",
    ])


    with tabs[0]:
        show_boats(entries)

        show_comparison(
            entries,
            key_suffix="overview"
        )

        show_prediction(
            actual, prediction
        )


    with tabs[1]:
        st.subheader("3連単120通り")

        full = prediction.copy()
        full["確率 (%)"] = (
            full["予測確率"] * 100
        ).round(4)

        st.dataframe(
            full[["順位", "3連単", "確率 (%)"]],
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            f"予測確率の合計："
            f"{prediction['予測確率'].sum():.6f}"
        )
        
    with tabs[2]:
        show_boats(entries)
        show_comparison(
            entries,
            key_suffix="details"
        )

    with tabs[3]:
        show_groups(
            entries, day, venue, race
        )

    with tabs[4]:
        show_importance(
            model, report
        )

    st.divider()

    st.caption(
        "このサイトは保存済み2024年データを"
        "読み取り専用で分析します。"
        "2025〜2026年の未使用テストデータは扱いません。"
        "表示確率は未校正であり、実戦の利益を"
        "保証するものではありません。"
    )


if __name__ == "__main__":
    main()
