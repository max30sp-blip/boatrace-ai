import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "boatrace.db"

st.set_page_config(
    page_title="出走表・選手比較 | BOAT RACE AI",
    page_icon="🚤",
    layout="wide",
)

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


def connect():
    if not DB_PATH.is_file():
        raise FileNotFoundError(DB_PATH)

    con = sqlite3.connect(
        DB_PATH.resolve().as_uri() + "?mode=ro",
        uri=True,
        timeout=20,
    )

    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA busy_timeout=20000")

    return con


@st.cache_data(ttl=600)
def get_columns():
    with connect() as con:
        return [
            r[1]
            for r in con.execute(
                "PRAGMA table_info(program_entries)"
            )
        ]


@st.cache_data(ttl=600)
def get_dates():
    with connect() as con:
        rows = con.execute("""
            SELECT DISTINCT race_date
            FROM program_entries
            WHERE race_date BETWEEN
                  '2024-01-01' AND '2024-12-31'
            ORDER BY race_date DESC
        """).fetchall()

    return [r[0] for r in rows]


@st.cache_data(ttl=600)
def get_venues(day):
    with connect() as con:
        rows = con.execute("""
            SELECT DISTINCT venue_code
            FROM program_entries
            WHERE race_date = ?
            ORDER BY venue_code
        """, (day,)).fetchall()

    return [int(r[0]) for r in rows]


@st.cache_data(ttl=600)
def get_races(day, venue):
    with connect() as con:
        rows = con.execute("""
            SELECT DISTINCT race_no
            FROM program_entries
            WHERE race_date = ?
              AND venue_code = ?
            ORDER BY race_no
        """, (day, venue)).fetchall()

    return [int(r[0]) for r in rows]


@st.cache_data(ttl=600, max_entries=100)
def get_entries(day, venue, race):
    with connect() as con:
        df = pd.read_sql_query("""
            SELECT *
            FROM program_entries
            WHERE race_date = ?
              AND venue_code = ?
              AND race_no = ?
            ORDER BY lane
        """, con, params=(day, venue, race))

    return df


def available(df, candidates):
    return [c for c in candidates if c in df.columns]


def display_entries(df):
    preferred = [
        "lane",
        "racer_id",
        "racer_name",
        "name",
        "grade",
        "age",
        "national_win_rate",
        "national_2rate",
        "local_win_rate",
        "local_2rate",
        "avg_st",
        "motor_no",
        "motor_2rate",
        "boat_no",
        "boat_2rate",
    ]

    cols = available(df, preferred)

    if not cols:
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True
        )
        return

    labels = {
        "lane": "艇番",
        "racer_id": "選手登録番号",
        "racer_name": "選手名",
        "name": "選手名",
        "grade": "級別",
        "age": "年齢",
        "national_win_rate": "全国勝率",
        "national_2rate": "全国2連対率",
        "local_win_rate": "当地勝率",
        "local_2rate": "当地2連対率",
        "avg_st": "平均ST",
        "motor_no": "モーター番号",
        "motor_2rate": "モーター2連対率",
        "boat_no": "ボート番号",
        "boat_2rate": "ボート2連対率",
    }

    table = df[cols].rename(columns=labels)

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True
    )


def show_comparison(df):
    st.subheader("選手・機材の数値比較")

    candidates = [
        ("national_win_rate", "全国勝率"),
        ("local_win_rate", "当地勝率"),
        ("motor_2rate", "モーター2連対率"),
        ("boat_2rate", "ボート2連対率"),
    ]

    options = [
        (col, label)
        for col, label in candidates
        if col in df.columns
    ]

    if not options:
        st.info(
            "比較対象の数値列が見つかりません。"
            "元DBの列名を確認してください。"
        )
        return

    selected = st.selectbox(
        "比較する指標",
        options,
        format_func=lambda x: x[1],
    )

    col, label = selected

    chart = df[["lane", col]].copy()

    chart[col] = pd.to_numeric(
        chart[col], errors="coerce"
    )

    chart["lane"] = (
        chart["lane"].astype(str) + "号艇"
    )

    chart = chart.rename(columns={
        "lane": "艇番",
        col: label,
    }).set_index("艇番")

    st.bar_chart(chart)

    st.caption(
        "元DBの値をそのまま表示しています。"
        "百分率などの単位は元データの定義に従います。"
    )


def main():
    st.title("🚤 出走表・選手比較")
    st.caption(
        "BOAT RACE AI | WEB-2"
        " / 過去データの閲覧専用"
    )

    st.info(
        "2024年の保存済み番組表を表示します。"
        "当日の最新出走表ではありません。"
        "AIモデルやデータベースは変更しません。"
    )

    try:
        columns = get_columns()
        dates = get_dates()
    except Exception as exc:
        st.error(f"DB接続エラー: {exc}")
        st.stop()

    required = {
        "race_date", "venue_code",
        "race_no", "lane",
    }

    if not required.issubset(columns):
        st.error(
            "program_entriesの必須列が不足しています。"
        )
        st.stop()

    if not dates:
        st.warning("対象期間にデータがありません。")
        st.stop()

    c1, c2, c3 = st.columns(3)

    with c1:
        day = st.selectbox("開催日", dates)

    venues = get_venues(day)

    with c2:
        venue = st.selectbox(
            "競艇場",
            venues,
            format_func=lambda v:
                f"{VENUES.get(v, str(v))} ({v:02d})",
        )

    races = get_races(day, venue)

    with c3:
        race = st.selectbox(
            "レース",
            races,
            format_func=lambda r: f"{r}R",
        )

    if st.button(
        "出走表を表示",
        type="primary",
        use_container_width=True,
    ):
        st.session_state["web2_selection"] = (
            day, venue, race
        )

    selection = st.session_state.get(
        "web2_selection"
    )

    if selection is None:
        st.write("開催日・競艇場・レースを選択してください。")
        return

    day, venue, race = selection

    st.divider()
    st.subheader(
        f"{day} | {VENUES.get(venue, venue)} "
        f"{race}R"
    )

    try:
        df = get_entries(day, venue, race)
    except Exception as exc:
        st.error(f"出走表取得エラー: {exc}")
        return

    if df.empty:
        st.warning("出走表が見つかりません。")
        return

    st.metric("出走艇数", len(df))

    st.subheader("出走表")
    display_entries(df)

    show_comparison(df)

    with st.expander("DBの全項目を表示"):
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True
        )

    st.caption(
        "このページは読み取り専用です。"
        "学習・再計算・モデル保存は行いません。"
    )


if __name__ == "__main__":
    main()