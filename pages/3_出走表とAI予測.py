
import json
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

st.set_page_config(
    page_title="出走表とAI予測 | BOAT RACE AI",
    page_icon="🚤",
    layout="wide",
)


def quote(name):
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
def get_model():
    return lgb.Booster(model_file=str(MODEL_PATH))


@st.cache_data
def get_report():
    return json.loads(
        REPORT_PATH.read_text(encoding="utf-8")
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
            SELECT DISTINCT race_no
            FROM train_races
            WHERE race_date = ?
              AND venue_code = ?
            ORDER BY race_no
        """, (day, venue)).fetchall()

    return [int(r[0]) for r in rows]


@st.cache_data(max_entries=100)
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


@st.cache_data(max_entries=100)
def load_features(day, venue, race, feature_tuple):
    features = list(feature_tuple)

    cols = ", ".join(
        [quote("label")] +
        [quote(f) for f in features]
    )

    with connect(TRAIN_DB) as con:
        rows = con.execute(
            f"""
            SELECT {cols}
            FROM train_races
            WHERE race_date = ?
              AND venue_code = ?
              AND race_no = ?
            """,
            (day, venue, race)
        ).fetchall()

    if len(rows) != 1:
        raise RuntimeError(
            f"学習データのレース件数が不正: {len(rows)}"
        )

    label = str(rows[0][0])

    x = np.array(
        [
            np.nan if v is None else float(v)
            for v in rows[0][1:]
        ],
        dtype=np.float32
    )

    x[~np.isfinite(x)] = np.nan

    return label, x


def show_entries(df):
    st.subheader("出走表・選手比較")

    candidates = [
        ("lane", "艇番"),
        ("racer_id", "登録番号"),
        ("racer_name", "選手名"),
        ("name", "選手名"),
        ("grade", "級別"),
        ("national_win_rate", "全国勝率"),
        ("local_win_rate", "当地勝率"),
        ("avg_st", "平均ST"),
        ("motor_no", "モーター番号"),
        ("motor_2rate", "モーター2連対率"),
        ("boat_no", "ボート番号"),
        ("boat_2rate", "ボート2連対率"),
    ]

    columns = []
    renamed = {}
    seen_labels = set()

    for col, label in candidates:
        if col in df.columns and label not in seen_labels:
            columns.append(col)
            renamed[col] = label
            seen_labels.add(label)

    if len(df) != 6:
        st.warning(
            f"出走表が6艇ではありません: {len(df)}艇"
        )

    if columns:
        st.dataframe(
            df[columns].rename(columns=renamed),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True
        )

    chart_columns = [
        ("national_win_rate", "全国勝率"),
        ("local_win_rate", "当地勝率"),
        ("motor_2rate", "モーター2連対率"),
        ("boat_2rate", "ボート2連対率"),
    ]

    options = [
        (col, label)
        for col, label in chart_columns
        if col in df.columns
    ]

    if options:
        chosen = st.selectbox(
            "比較指標",
            options,
            format_func=lambda v: v[1]
        )

        col, label = chosen

        chart = df[["lane", col]].copy()
        chart[col] = pd.to_numeric(
            chart[col], errors="coerce"
        )
        chart["lane"] = (
            chart["lane"].astype(str) + "号艇"
        )

        chart = chart.set_index("lane")
        chart = chart.rename(columns={col: label})
        st.bar_chart(chart)


def predict(model, x, class_order):
    probabilities = np.asarray(
        model.predict(
            x.reshape(1, -1),
            num_threads=4
        )
    ).reshape(-1)

    if probabilities.shape != (120,):
        raise RuntimeError("予測結果の形状が不正です")

    if not np.isfinite(probabilities).all():
        raise RuntimeError("予測確率に異常があります")

    result = pd.DataFrame({
        "3連単": list(map(str, class_order)),
        "確率": probabilities
    })

    result = result.sort_values(
        "確率", ascending=False
    ).reset_index(drop=True)

    result.insert(
        0, "順位",
        np.arange(1, len(result) + 1)
    )

    return result


def show_prediction(model, report, day, venue, race):
    features = report["feature_names"]
    classes = report["class_order"]

    actual, x = load_features(
        day, venue, race, tuple(features)
    )

    result = predict(model, x, classes)

    st.divider()
    st.subheader("AI 3連単予測")

    top = result.iloc[0]
    top3 = result.head(3)

    a, b, c = st.columns(3)

    a.metric(
        "AI本命",
        top["3連単"]
    )
    b.metric(
        "本命の予測確率",
        f"{top['確率'] * 100:.2f}%"
    )
    c.metric(
        "上位3点の合計確率",
        f"{top3['確率'].sum() * 100:.2f}%"
    )

    st.subheader("3連単ランキング")

    count = st.slider(
        "表示件数",
        min_value=3,
        max_value=120,
        value=15
    )

    display = result.head(count).copy()
    display["予測確率 (%)"] = (
        display["確率"] * 100
    ).round(3)

    st.dataframe(
        display[
            ["順位", "3連単", "予測確率 (%)"]
        ],
        hide_index=True,
        use_container_width=True
    )

    st.bar_chart(
        display.head(15).set_index(
            "3連単"
        )["予測確率 (%)"]
    )

    st.subheader("実際の結果との照合")
    st.write(f"実際の3連単：**{actual}**")

    matched = result[
        result["3連単"] == actual
    ]

    if not matched.empty:
        rank = int(matched.iloc[0]["順位"])
        st.write(f"実際の結果の予測順位：**{rank}位**")

        if rank == 1:
            st.success("Top1的中")
        elif rank <= 3:
            st.success("Top3的中")
        else:
            st.info("Top3には含まれていません")
    else:
        st.warning(
            "実際の結果がクラス一覧に見つかりません"
        )

    st.caption(
        "2024年の過去レースに対するモデル予測です。"
        "確率は未校正であり、回収率や利益を"
        "保証するものではありません。"
    )


def main():
    st.title("🚤 出走表 × 3連単AI予測")
    st.caption("BOAT RACE AI | WEB-3")

    st.info(
        "2024年の保存済みレース専用。"
        "AI本体・モデル・DBは変更しません。"
    )

    try:
        report = get_report()
        model = get_model()

        features = report["feature_names"]
        classes = report["class_order"]

        if len(features) != 618 or len(classes) != 120:
            raise RuntimeError(
                "モデルのメタデータが不正です"
            )

        if model.feature_name() != features:
            raise RuntimeError(
                "モデルと特徴量の順序が一致しません"
            )

        dates = get_dates()

        if not dates:
            st.warning("対象データがありません")
            return

    except Exception as e:
        st.error(f"初期化エラー: {e}")
        return

    c1, c2, c3 = st.columns(3)

    with c1:
        day = st.selectbox("日付", dates)

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
            "レース",
            races,
            format_func=lambda r: f"{r}R"
        )

    if st.button(
        "出走表とAI予測を表示",
        type="primary",
        use_container_width=True
    ):
        st.session_state["web3_selection"] = (
            day, venue, race
        )

    selection = st.session_state.get(
        "web3_selection"
    )

    if selection is None:
        st.write("レースを選んで実行してください。")
        return

    day, venue, race = selection

    st.divider()
    st.header(
        f"{day} | {VENUES.get(venue, venue)} "
        f"{race}R"
    )

    try:
        entries = load_entries(
            day, venue, race
        )

        if entries.empty:
            st.warning("出走表が見つかりません")
            return

        show_entries(entries)

        show_prediction(
            model, report, day, venue, race
        )

    except Exception as e:
        st.error(f"表示エラー: {e}")


if __name__ == "__main__":
    main()
