
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ==================================================
# WEB-4  選手・モーター詳細分析
# 保存済み特徴量DBの閲覧専用
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

PROGRAM_DB = DATA / "boatrace.db"
TRAIN_DB = DATA / "train_dataset_v1.db"

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

KEYS = [
    "race_date",
    "venue_code",
    "race_no",
    "lane",
    "racer_id",
]

st.set_page_config(
    page_title="選手・モーター詳細分析",
    page_icon="📊",
    layout="wide",
)


def q(name):
    return '"' + name.replace('"', '""') + '"'


def connect_readonly(path):
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


@st.cache_data(ttl=600)
def get_dates():
    with closing(connect_readonly(TRAIN_DB)) as con:
        rows = con.execute("""
            SELECT DISTINCT race_date
            FROM train_races
            WHERE race_date BETWEEN
                '2024-01-01' AND '2024-12-31'
            ORDER BY race_date DESC
        """).fetchall()

    return [r[0] for r in rows]


@st.cache_data(ttl=600)
def get_venues(day):
    with closing(connect_readonly(TRAIN_DB)) as con:
        rows = con.execute("""
            SELECT DISTINCT venue_code
            FROM train_races
            WHERE race_date = ?
            ORDER BY venue_code
        """, (day,)).fetchall()

    return [int(r[0]) for r in rows]


@st.cache_data(ttl=600)
def get_races(day, venue):
    with closing(connect_readonly(TRAIN_DB)) as con:
        rows = con.execute("""
            SELECT DISTINCT race_no
            FROM train_races
            WHERE race_date = ?
              AND venue_code = ?
            ORDER BY race_no
        """, (day, venue)).fetchall()

    return [int(r[0]) for r in rows]


@st.cache_data(ttl=600, max_entries=60)
def get_program(day, venue, race):
    with closing(connect_readonly(PROGRAM_DB)) as con:
        df = pd.read_sql_query("""
            SELECT *
            FROM program_entries
            WHERE race_date = ?
              AND venue_code = ?
              AND race_no = ?
            ORDER BY lane
        """, con, params=(day, venue, race))

    return df


@st.cache_data(ttl=600, max_entries=120)
def get_features(group, day, venue, race):
    filename, table, lane_column, description = (
        FEATURE_DBS[group]
    )

    path = DATA / filename

    with closing(connect_readonly(path)) as con:
        actual_tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table'"
            )
        }

        if table not in actual_tables:
            raise RuntimeError(
                f"{filename}に{table}がありません"
            )

        df = pd.read_sql_query(
            f"""
            SELECT *
            FROM {q(table)}
            WHERE race_date = ?
              AND venue_code = ?
              AND race_no = ?
            ORDER BY {q(lane_column)}
            """,
            con,
            params=(day, venue, race)
        )

    if lane_column != "lane":
        df = df.rename(
            columns={lane_column: "lane"}
        )

    return df


def selected_columns(df, columns):
    return [c for c in columns if c in df.columns]


def build_analysis(day, venue, race):
    program = get_program(
        day, venue, race
    )

    if program.empty:
        raise RuntimeError(
            "番組表が見つかりません"
        )

    if len(program) != 6:
        raise RuntimeError(
            f"出走艇数が6ではありません: {len(program)}"
        )

    if program["lane"].duplicated().any():
        raise RuntimeError("出走表の艇番が重複")

    data = program.copy()
    errors = []
    status = []

    for group, spec in FEATURE_DBS.items():
        description = spec[3]

        try:
            feat = get_features(
                group, day, venue, race
            )

            if len(feat) != 6:
                raise RuntimeError(
                    f"特徴量行数={len(feat)}"
                )

            if not set(KEYS).issubset(feat.columns):
                raise RuntimeError(
                    "必要なキー列が不足"
                )

            if feat.duplicated(KEYS).any():
                raise RuntimeError(
                    "特徴量キーが重複"
                )

            check = program[
                KEYS
            ].merge(
                feat[KEYS],
                how="left",
                on=KEYS,
                indicator=True
            )

            if not (
                check["_merge"] == "both"
            ).all():
                raise RuntimeError(
                    "番組表と選手ID・艇番が不一致"
                )

            feature_cols = [
                c for c in feat.columns
                if c not in KEYS
            ]

            feat = feat[
                KEYS + feature_cols
            ].rename(
                columns={
                    c: f"{group}_{c}"
                    for c in feature_cols
                }
            )

            data = data.merge(
                feat,
                on=KEYS,
                how="left",
                validate="one_to_one"
            )

            status.append({
                "グループ": group,
                "内容": description,
                "列数": len(feature_cols),
                "状態": "読み込み成功",
            })

        except Exception as exc:
            errors.append(
                f"{group} {description}: {exc}"
            )

            status.append({
                "グループ": group,
                "内容": description,
                "列数": 0,
                "状態": "読み込み失敗",
            })

    return (
        data.sort_values("lane"),
        pd.DataFrame(status),
        errors
    )


def display_program(df):
    st.subheader("6艇の基本情報")

    names = {
        "lane": "艇番",
        "racer_id": "選手登録番号",
        "racer_name": "選手名",
        "name": "選手名",
        "grade": "級別",
        "national_win_rate": "全国勝率",
        "local_win_rate": "当地勝率",
        "avg_st": "平均ST",
        "motor_no": "モーター番号",
        "motor_2rate": "モーター2連対率",
        "boat_no": "ボート番号",
        "boat_2rate": "ボート2連対率",
    }

    columns = []
    seen = set()

    for col, label in names.items():
        if col in df.columns and label not in seen:
            columns.append(col)
            seen.add(label)

    st.dataframe(
        df[columns].rename(columns=names),
        hide_index=True,
        use_container_width=True
    )


def group_numeric_columns(df, group):
    prefix = group + "_"

    columns = []

    for col in df.columns:
        if not col.startswith(prefix):
            continue

        values = pd.to_numeric(
            df[col], errors="coerce"
        )

        if values.notna().any():
            columns.append(col)

    return columns


def display_group(df, group):
    description = FEATURE_DBS[group][3]

    st.subheader(
        f"{group}：{description}"
    )

    columns = group_numeric_columns(
        df, group
    )

    if not columns:
        st.warning(
            "表示可能な数値特徴量がありません"
        )
        return

    selection = st.selectbox(
        "比較する特徴量",
        columns,
        format_func=lambda c:
            c.replace(group + "_", ""),
        key=f"metric_{group}"
    )

    values = pd.to_numeric(
        df[selection],
        errors="coerce"
    )

    chart = pd.DataFrame({
        "艇番": (
            df["lane"].astype(str) + "号艇"
        ),
        "値": values,
    }).set_index("艇番")

    st.bar_chart(chart)

    st.caption(
        "特徴量DBの値をそのまま表示。"
        "勝率などは0〜1、平均STは秒など、"
        "元の特徴量定義に従います。"
    )

    table = df[
        ["lane"] + columns
    ].copy()

    table = table.rename(
        columns={
            "lane": "艇番",
            **{
                col: col.replace(
                    group + "_", ""
                )
                for col in columns
            }
        }
    )

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        f"表示対象：{len(columns)}個の数値特徴量"
    )


def display_all_comparison(df):
    st.subheader("全グループ横断比較")

    numeric_cols = []

    candidates = [
        "national_win_rate",
        "local_win_rate",
        "motor_2rate",
        "boat_2rate",
    ]

    candidates += [
        col
        for col in df.columns
        if any(
            col.startswith(group + "_")
            for group in FEATURE_DBS
        )
    ]

    for col in candidates:
        if col not in df.columns:
            continue

        values = pd.to_numeric(
            df[col], errors="coerce"
        )

        if values.notna().any():
            numeric_cols.append(col)

    if not numeric_cols:
        st.info(
            "比較可能な指標がありません"
        )
        return

    chosen = st.multiselect(
        "比較する指標（最大3つ）",
        numeric_cols,
        default=numeric_cols[:1],
        max_selections=3
    )

    if not chosen:
        return

    table = df[
        ["lane"] + chosen
    ].copy()

    table["lane"] = (
        table["lane"].astype(str) + "号艇"
    )

    table = table.set_index("lane")

    for col in chosen:
        table[col] = pd.to_numeric(
            table[col], errors="coerce"
        )

    st.dataframe(
        table,
        use_container_width=True
    )

    st.bar_chart(table)

    st.warning(
        "異なる単位の特徴量を同じグラフに"
        "表示すると比較しづらくなります。"
        "必要に応じて1項目ずつ選んでください。",
        icon="ℹ️"
    )


def main():
    st.title("📊 選手・モーター詳細分析")

    st.caption(
        "BOAT RACE AI / WEB-4"
        " / 保存済み特徴量の閲覧"
    )

    st.info(
        "2024年の過去レース専用。"
        "既存DBを読み取り専用で使用し、"
        "学習済みAIや特徴量は変更しません。"
    )

    try:
        dates = get_dates()
    except Exception as exc:
        st.error(
            f"データ取得エラー: {exc}"
        )
        st.stop()

    if not dates:
        st.warning(
            "2024年のデータがありません"
        )
        st.stop()

    c1, c2, c3 = st.columns(3)

    with c1:
        day = st.selectbox(
            "開催日", dates
        )

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
            format_func=lambda x: f"{x}R"
        )

    if st.button(
        "詳細分析を表示",
        type="primary",
        use_container_width=True
    ):
        st.session_state["web4_selection"] = (
            day, venue, race
        )

    selection = st.session_state.get(
        "web4_selection"
    )

    if selection is None:
        st.write(
            "レースを選んで詳細分析を実行してください。"
        )
        return

    day, venue, race = selection

    st.divider()
    st.header(
        f"{day} | {VENUES.get(venue, venue)} "
        f"{race}R"
    )

    try:
        data, status, errors = build_analysis(
            day, venue, race
        )
    except Exception as exc:
        st.error(
            f"分析データ取得エラー: {exc}"
        )
        return

    display_program(data)

    st.subheader("特徴量DBの読み込み状況")
    st.dataframe(
        status,
        hide_index=True,
        use_container_width=True
    )

    if errors:
        with st.expander(
            "読み込みエラーの詳細"
        ):
            for error in errors:
                st.warning(error)

    st.divider()

    tabs = st.tabs([
        "全体比較",
        "F0 過去成績",
        "F1 コース",
        "F2 艇番",
        "F3 モーター",
        "F4 競艇場",
        "F5 直近成績",
        "F6 今節",
    ])

    with tabs[0]:
        display_all_comparison(data)

    for index, group in enumerate(
        FEATURE_DBS, start=1
    ):
        with tabs[index]:
            display_group(data, group)

    st.divider()

    with st.expander(
        "全特徴量データを表示"
    ):
        st.dataframe(
            data,
            hide_index=True,
            use_container_width=True
        )

    st.caption(
        "分析専用画面。"
        "データベースへの書き込み、"
        "特徴量の再計算、AI再学習は行いません。"
    )


if __name__ == "__main__":
    main()
