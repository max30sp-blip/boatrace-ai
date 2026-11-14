
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ==================================================
# WEB-9 : 当日・未来レース出走表
# 既存DB・学習済みAIには一切書き込まない
# ==================================================

JST = ZoneInfo("Asia/Tokyo")

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

COLORS = {
    1: ("#FFFFFF", "#111827"),
    2: ("#111827", "#FFFFFF"),
    3: ("#EF4444", "#FFFFFF"),
    4: ("#2563EB", "#FFFFFF"),
    5: ("#FACC15", "#111827"),
    6: ("#16A34A", "#FFFFFF"),
}

FIELDS = {
    "艇番": ("entry_number", None),
    "選手名": ("name", None),
    "登録番号": ("number", None),
    "級別コード": ("rank_number", None),
    "年齢": ("age", None),
    "全国勝率": ("national_win_rate", None),
    "全国2連対率(%)": (
        "national_top_2_percent", None
    ),
    "当地勝率": ("local_win_rate", None),
    "当地2連対率(%)": (
        "local_top_2_percent", None
    ),
    "平均ST": ("average_start_timing", None),
    "モーター番号": ("motor_number", None),
    "モーター2連対率(%)": (
        "motor_top_2_percent", None
    ),
    "ボート番号": ("boat_number", None),
    "ボート2連対率(%)": (
        "boat_top_2_percent", None
    ),
}

COMPARE_RULES = {
    "全国勝率": "max",
    "全国2連対率(%)": "max",
    "当地勝率": "max",
    "当地2連対率(%)": "max",
    "平均ST": "min",
    "モーター2連対率(%)": "max",
    "ボート2連対率(%)": "max",
}

st.set_page_config(
    page_title="当日・未来レース | BOAT RACE AI",
    page_icon="📅",
    layout="wide",
)


def api_url(day):
    stamp = day.strftime("%Y%m%d")
    return (
        "https://boatraceopenapi.github.io"
        f"/api/v1/{day.year}/{stamp}.json"
    )


def official_url(day, venue, race):
    return (
        "https://www.boatrace.jp/owpc/pc/race/racelist"
        f"?hd={day:%Y%m%d}"
        f"&jcd={venue:02d}"
        f"&rno={race}"
    )


@st.cache_data(ttl=180, show_spinner=False)
def download_daily(day_iso):
    day = datetime.strptime(
        day_iso, "%Y-%m-%d"
    ).date()

    url = api_url(day)

    response = requests.get(
        url,
        timeout=20,
        headers={
            "User-Agent": "BOAT-RACE-AI-Research/1.0",
            "Accept": "application/json",
        }
    )

    if response.status_code == 404:
        return None

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, dict):
        raise ValueError(
            "APIが想定外のデータを返しました"
        )

    return data


def get_stadiums(data):
    return (
        data.get("programs", {})
        .get("stadiums", {})
    )


def get_races(data, venue):
    stadium = get_stadiums(data).get(
        str(venue), {}
    )

    return stadium.get("races", {})


def build_entries(race_data):
    racers = race_data.get("racers") or {}

    if not isinstance(racers, dict):
        raise ValueError("選手情報の形式が不正です")

    rows = []

    for lane in range(1, 7):
        racer = racers.get(str(lane))

        if not isinstance(racer, dict):
            raise ValueError(
                f"{lane}号艇の出走情報がありません"
            )

        row = {}

        for title, (source, _) in FIELDS.items():
            row[title] = racer.get(source)

        row["艇番"] = lane

        # 元の級別文字列がある場合は優先
        grade = racer.get("rank_number_source")
        if grade:
            row["級別コード"] = grade

        rows.append(row)

    df = pd.DataFrame(rows)

    if df["登録番号"].isna().any():
        raise ValueError(
            "選手登録番号が欠損しています"
        )

    if df["登録番号"].duplicated().any():
        raise ValueError(
            "選手登録番号が重複しています"
        )

    return df


def highlight(df):
    def color_column(column):
        name = column.name

        if name not in COMPARE_RULES:
            return [""] * len(column)

        nums = pd.to_numeric(
            column, errors="coerce"
        )

        valid = nums.dropna()

        if valid.empty:
            return [""] * len(column)

        direction = COMPARE_RULES[name]

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
            if pd.notna(v)
            and np.isclose(v, best)
            else ""
            for v in nums
        ]

    return (
        df.style
        .apply(color_column, axis=0)
        .format(
            {
                name: "{:.2f}"
                for name in COMPARE_RULES
            },
            na_rep="—"
        )
    )


def display_entries(df):
    st.subheader("🚤 出走表")

    st.dataframe(
        highlight(df),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "緑色は各指標の6艇中最良値。"
        "平均STは小さい方を強調します。"
        "機材番号・年齢・級別には順位を付けません。"
    )

    st.subheader("📊 6艇比較")

    metric = st.selectbox(
        "比較指標",
        list(COMPARE_RULES),
        key="web9_metric"
    )

    chart = df[["艇番", metric]].copy()

    chart[metric] = pd.to_numeric(
        chart[metric], errors="coerce"
    )

    chart["艇番"] = (
        chart["艇番"].astype(str) + "号艇"
    )

    st.bar_chart(chart.set_index("艇番"))



def display_preview(race_data):
    """WEB-10: 展示・気象情報の詳細分析"""

    preview = race_data.get("preview")

    if not isinstance(preview, dict) or not preview:
        st.info(
            "展示・気象情報はまだ取得できていません。"
            "発表後に「最新情報を再取得」してください。"
        )
        return

    WIND = {
        1: "北", 2: "北北東", 3: "北東",
        4: "東北東", 5: "東", 6: "東南東",
        7: "南東", 8: "南南東", 9: "南",
        10: "南南西", 11: "南西",
        12: "西南西", 13: "西",
        14: "西北西", 15: "北西",
        16: "北北西", 17: "無風",
    }

    WEATHER = {
        1: "晴", 2: "曇り", 3: "雨",
        4: "雪", 5: "霧", 6: "台風",
        99: "その他",
    }

    PARTS = {
        1: "ピストン",
        2: "ピストンリング",
        3: "電気一式",
        4: "キャブレター",
        5: "シリンダ",
        6: "クランクシャフト",
        7: "ギヤケース",
        8: "キャリアボデー",
    }

    def numeric(value):
        if value is None or value == "":
            return None
        try:
            num = float(value)
            return num if np.isfinite(num) else None
        except (ValueError, TypeError):
            return None

    def integer(value):
        num = numeric(value)
        return int(num) if num is not None else None

    def fmt(value, unit="", digits=1):
        num = numeric(value)
        if num is None:
            return "未取得"
        return f"{num:.{digits}f}{unit}"

    def code_label(key, mapping):
        code = integer(preview.get(key))

        if code in mapping:
            return mapping[code]

        source = preview.get(key + "_source")
        if source is not None and str(source).strip():
            return str(source)

        return "未取得"

    # ==========================================
    # 1. 気象・風向
    # ==========================================

    st.subheader("🌤️ 気象・水面状況")

    wind_code = integer(
        preview.get("wind_direction_number")
    )

    wind_speed = numeric(
        preview.get("wind_speed")
    )

    direction = code_label(
        "wind_direction_number", WIND
    )

    weather = code_label(
        "weather_number", WEATHER
    )

    if wind_speed == 0 or wind_code == 17:
        wind_type = "無風"
    elif wind_code is None:
        wind_type = "判定不可"
    elif 3 <= wind_code <= 7:
        wind_type = "追い風"
    elif 11 <= wind_code <= 15:
        wind_type = "向かい風"
    elif wind_code in WIND:
        wind_type = "横風寄り"
    else:
        wind_type = "判定不可"

    c1, c2, c3 = st.columns(3)

    c1.metric("風向区分", wind_type)
    c2.metric("風向（API表記）", direction)
    c3.metric(
        "風速",
        fmt(wind_speed, " m/s", 0)
    )

    c4, c5, c6 = st.columns(3)

    c4.metric("天候", weather)
    c5.metric(
        "波高",
        fmt(preview.get("wave_height"), " cm", 0)
    )
    c6.metric(
        "気温",
        fmt(preview.get("air_temperature"), " ℃")
    )

    st.metric(
        "水温",
        fmt(preview.get("water_temperature"), " ℃")
    )

    if wind_type == "追い風":
        st.info(
            f"🌬️ 追い風｜風向表示：{direction}｜"
            f"風速：{fmt(wind_speed, ' m/s', 0)}"
        )

    elif wind_type == "向かい風":
        st.info(
            f"🌬️ 向かい風｜風向表示：{direction}｜"
            f"風速：{fmt(wind_speed, ' m/s', 0)}"
        )

    elif wind_type == "無風":
        st.success("🍃 無風")

    else:
        st.info(
            f"風向区分：{wind_type}｜"
            f"風向表示：{direction}"
        )

    st.caption(
        "風向の北・南・東・西はAPIの公式相対表記です。"
        "地図上の方位ではありません。"
    )

    with st.expander("風向区分について"):
        st.write(
            "追い風：風向コード3〜7、"
            "向かい風：11〜15、"
            "無風：17（または風速0）、"
            "その他の有効コード：横風寄り。"
        )
        st.write(
            "風の強さによってレース展開は変わりますが、"
            "風向だけで逃げ・差し・まくりを"
            "断定することはできません。"
        )

    # ==========================================
    # 2. 展示データ
    # ==========================================

    st.divider()
    st.subheader("🚤 スタート展示・展示タイム")

    racers = preview.get("racers")

    if not isinstance(racers, dict) or not racers:
        st.warning(
            "選手別の展示データはまだ取得できません。"
        )
        return

    def start_display(racer):
        source = racer.get("start_timing_source")

        if source is not None and str(source).strip():
            return str(source)

        value = numeric(racer.get("start_timing"))

        if value is None:
            return "—"

        if value < 0:
            return f"F{abs(value):.2f}"

        return f"{value:.2f}"

    def parts_display(racer):
        names = []

        if racer.get("propeller"):
            names.append("新プロペラ")

        parts = racer.get("parts")

        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue

                number = integer(part.get("number"))

                name = PARTS.get(
                    number,
                    part.get("number_source")
                )

                if not name:
                    continue

                quantity = integer(part.get("quantity"))

                if quantity is not None:
                    names.append(f"{name}×{quantity}")
                else:
                    names.append(str(name))

        if names:
            return "、".join(names)

        if "parts" not in racer and "propeller" not in racer:
            return "情報なし"

        return "交換なし"

    rows = []

    for lane in range(1, 7):
        racer = racers.get(str(lane)) or {}

        if not isinstance(racer, dict):
            racer = {}

        rows.append({
            "艇番": lane,
            "進入コース": integer(
                racer.get("course_number")
            ),
            "展示ST": start_display(racer),
            "展示タイム": numeric(
                racer.get("exhibition_time")
            ),
            "チルト": numeric(
                racer.get("tilt_adjustment")
            ),
            "体重(kg)": numeric(
                racer.get("weight")
            ),
            "部品交換": (
                parts_display(racer)
                if racer else "未取得"
            ),
        })

    df = pd.DataFrame(rows)

    # 最も速い展示タイムだけを緑にする
    def highlight_exhibition(column):
        if column.name != "展示タイム":
            return [""] * len(column)

        values = pd.to_numeric(
            column, errors="coerce"
        )

        valid = values.dropna()

        if valid.empty:
            return [""] * len(column)

        best = valid.min()

        return [
            (
                "background-color:#BBF7D0;"
                "color:#14532D;"
                "font-weight:bold;"
            )
            if pd.notna(v) and np.isclose(v, best)
            else ""
            for v in values
        ]

    styled = (
        df.style
        .apply(highlight_exhibition, axis=0)
        .format({
            "展示タイム": "{:.2f}",
            "チルト": "{:.1f}",
            "体重(kg)": "{:.1f}",
        }, na_rep="—")
    )

    st.dataframe(
        styled,
        hide_index=True,
        use_container_width=True
    )

    # ==========================================
    # 3. 展示タイムの順位
    # ==========================================

    valid = df.dropna(subset=["展示タイム"])

    if not valid.empty:
        best_time = valid["展示タイム"].min()

        best_lanes = valid.loc[
            np.isclose(
                valid["展示タイム"],
                best_time
            ),
            "艇番"
        ].tolist()

        st.success(
            "🏆 展示タイム最速："
            + "・".join(
                f"{lane}号艇"
                for lane in best_lanes
            )
            + f"（{best_time:.2f}秒）"
        )

        chart = df[
            ["艇番", "展示タイム"]
        ].copy()

        chart["艇番"] = (
            chart["艇番"].astype(str) + "号艇"
        )

        st.bar_chart(
            chart.set_index("艇番")
        )

    else:
        st.info("展示タイムはまだ未取得です。")

    # ==========================================
    # 4. 展示STと進入の注意点
    # ==========================================

    if df["展示ST"].isin(["—"]).all():
        st.info("展示STは未取得です。")

    st.caption(
        "展示STはスタート展示の値です。"
        "F表示はフライングであり、"
        "単純に『最速で優秀』とは評価しません。"
        "展示の進入コースも本番と異なる場合があります。"
    )

    st.caption(
        "この情報は表示・分析専用です。"
        "現段階ではAI予測に入力していません。"
    )


def display_race(day, venue, race_no, data):
    race = get_races(data, venue).get(
        str(race_no)
    )

    st.divider()
    st.header(
        f"{day} | {VENUES[venue]} {race_no}R"
    )

    official = official_url(
        day, venue, race_no
    )

    st.link_button(
        "公式出走表で確認",
        official
    )

    if not isinstance(race, dict):
        st.warning(
            "このレースの出走表は"
            "データ源から取得できませんでした。"
            "公式サイトを確認してください。"
        )
        return

    title = race.get("title") or "レース"

    st.subheader(str(title))

    details = {
        "締切予定": race.get("closed_at"),
        "距離": race.get("distance"),
        "開催日目": race.get("day_number"),
    }

    st.write({
        k: v for k, v in details.items()
        if v is not None
    })

    try:
        entries = build_entries(race)
    except Exception as exc:
        st.warning(f"出走表の確認に失敗: {exc}")
        return

    display_entries(entries)

    st.divider()
    display_preview(race)

    st.divider()

    st.info(
        "未来レースのAI予測は現在未実装です。"
        "AI改善チャット側で事前特徴量の生成と"
        "検証が完了してから接続します。"
    )


def main():
    st.title("📅 当日・未来レース")

    st.caption(
        "BOAT RACE AI | WEB-9 | 出走表取得"
    )

    st.info(
        "保存済みの過去レースとは別画面です。"
        "非公式データを取得して表示します。"
        "既存の学習DB・AIモデルには書き込みません。"
    )

    today = datetime.now(JST).date()

    c1, c2, c3 = st.columns(3)

    with c1:
        day = st.date_input(
            "開催日",
            value=today,
            min_value=today,
            max_value=today + timedelta(days=3),
            format="YYYY/MM/DD",
        )

    with c2:
        venue = st.selectbox(
            "競艇場",
            list(VENUES),
            format_func=lambda v:
                f"{VENUES[v]} ({v:02d})"
        )

    with c3:
        race_no = st.selectbox(
            "レース番号",
            list(range(1, 13)),
            format_func=lambda r: f"{r}R"
        )

    if st.button(
        "出走表を取得",
        type="primary",
        use_container_width=True
    ):
        st.session_state["web9_selection"] = (
            day.isoformat(), venue, race_no
        )

    selected = st.session_state.get(
        "web9_selection"
    )

    if selected is None:
        st.write(
            "日付・競艇場・レースを選択してください。"
        )
        return

    day_iso, venue, race_no = selected
    day = datetime.strptime(
        day_iso, "%Y-%m-%d"
    ).date()

    if day < today or day > today + timedelta(days=3):
        st.warning("対象期間外の日付です。")
        return
    
    if st.button(
        "🔄 最新情報を再取得",
        use_container_width=True
    ):
        download_daily.clear()
        st.rerun()   
        
    with st.spinner("出走表を取得しています"):
        try:
            data = download_daily(day_iso)
        except requests.RequestException as exc:
            st.error(f"通信エラー: {exc}")
            data = None
        except (ValueError, TypeError) as exc:
            st.error(f"データ形式エラー: {exc}")
            data = None

    if data is None:
        st.warning(
            "指定日のJSONがまだ公開されていないか、"
            "取得できませんでした。"
        )

        st.link_button(
            "BOAT RACE公式出走表を開く",
            official_url(day, venue, race_no)
        )
        return

    display_race(
        day, venue, race_no, data
    )

    st.caption(
        "データ提供：Boatrace Open API（非公式）。"
        "数分以上の更新遅延や欠損が生じる場合が"
        "あります。必ず公式出走表で最終確認してください。"
    )


if __name__ == "__main__":
    main()
