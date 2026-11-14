
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import importlib.util

import pandas as pd
import requests
import streamlit as st


# ==========================================
# WEB-12 設定
# ==========================================

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

BOAT_COLORS = {
    1: "#FFFFFF",
    2: "#111827",
    3: "#EF4444",
    4: "#2563EB",
    5: "#FACC15",
    6: "#16A34A",
}

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

RANK = {
    1: "A1", 2: "A2",
    3: "B1", 4: "B2",
}


# ==========================================
# WEB-11の既存機能を読み込む
# ==========================================

def load_web11():
    path = Path(__file__).with_name(
        "11_3連単オッズとフォーメーション.py"
    )

    if not path.exists():
        st.error(
            "WEB-11のPythonファイルが見つかりません。"
            f"確認先: {path}"
        )
        st.stop()

    spec = importlib.util.spec_from_file_location(
        "web11_shared", str(path)
    )

    if spec is None or spec.loader is None:
        st.error("WEB-11の読み込みに失敗しました。")
        st.stop()

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


web11 = load_web11()


# ==========================================
# 出走表API
# ==========================================

@st.cache_data(ttl=180, show_spinner=False)
def fetch_daily(day_iso):
    day = datetime.strptime(
        day_iso, "%Y-%m-%d"
    ).date()

    url = (
        "https://boatraceopenapi.github.io/api/v1/"
        f"{day.year}/{day:%Y%m%d}.json"
    )

    response = requests.get(
        url,
        timeout=20,
        headers={
            "User-Agent": "BOAT-RACE-AI-Research/1.0"
        },
    )

    if response.status_code == 404:
        return {}

    response.raise_for_status()
    return response.json()


def num(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


def fmt(value, digits=2, unit=""):
    value = num(value)
    if value is None:
        return "—"
    return f"{value:.{digits}f}{unit}"


def get_race(data, venue, race):
    return (
        data.get("programs", {})
        .get("stadiums", {})
        .get(str(venue), {})
        .get("races", {})
        .get(str(race), {})
    )


def wind_type(code, speed):
    if speed == 0 or code == 17:
        return "無風"
    if code is None:
        return "不明"
    if 3 <= code <= 7:
        return "追い風"
    if 11 <= code <= 15:
        return "向かい風"
    if code in WIND:
        return "横風寄り"
    return "不明"


def start_text(value, source=None):
    if source is not None and str(source).strip():
        return str(source)

    value = num(value)
    if value is None:
        return "—"
    if value < 0:
        return f"F{abs(value):.2f}"
    return f"{value:.2f}"


def highlight_best(series):
    values = pd.to_numeric(
        series, errors="coerce"
    )

    valid = values.dropna()

    if valid.empty:
        return [""] * len(series)

    # 展示タイム・平均STは小さいほど速い
    # 勝率・連対率は大きいほど高い
    if series.name in ("展示タイム", "平均ST"):
        best = valid.min()
    else:
        best = valid.max()

    return [
        (
            "background-color:#BBF7D0;"
            "color:#14532D;"
            "font-weight:bold;"
        )
        if pd.notna(value) and abs(value - best) < 1e-8
        else ""
        for value in values
    ]


# ==========================================
# 出走表表示
# ==========================================

def show_entries(race_data):
    st.subheader("🚤 出走表・選手比較")

    racers = race_data.get("racers") or {}

    if not racers:
        st.info("出走表がまだ公開されていません。")
        return

    rows = []

    for lane in range(1, 7):
        racer = racers.get(str(lane), {})

        rank = racer.get("rank_number")

        rows.append({
            "艇番": lane,
            "選手名": racer.get("name", "—"),
            "級別": RANK.get(rank, "—"),
            "全国勝率": num(
                racer.get("national_win_rate")
            ),
            "全国2連対率": num(
                racer.get("national_top_2_percent")
            ),
            "当地勝率": num(
                racer.get("local_win_rate")
            ),
            "モーター": racer.get("motor_number"),
            "モーター2連対率": num(
                racer.get("motor_top_2_percent")
            ),
            "ボート2連対率": num(
                racer.get("boat_top_2_percent")
            ),
            "平均ST": num(
                racer.get("average_start_timing")
            ),
        })

    df = pd.DataFrame(rows)

    numeric_columns = [
        "全国勝率", "全国2連対率",
        "当地勝率", "モーター2連対率",
        "ボート2連対率", "平均ST",
    ]

    styled = (
        df.style
        .apply(highlight_best, subset=numeric_columns)
        .format(
            {col: "{:.2f}" for col in numeric_columns},
            na_rep="—",
        )
    )

    st.dataframe(
        styled,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "緑色は各指標の最良値。平均STは最小値、"
        "勝率・連対率は最大値を強調しています。"
        "これはAI予測の評価ではありません。"
    )


# ==========================================
# 展示・気象表示
# ==========================================

def show_preview(race_data):
    st.subheader("🌤️ 展示・気象分析")

    preview = race_data.get("preview")

    if not isinstance(preview, dict) or not preview:
        st.info(
            "直前情報がまだありません。"
            "展示発表後に更新してください。"
        )
        return

    code = preview.get("wind_direction_number")
    speed = num(preview.get("wind_speed"))

    direction = WIND.get(code, "—")
    category = wind_type(code, speed)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("風向区分", category)
    c2.metric("風向（API表記）", direction)
    c3.metric("風速", fmt(speed, 0, "m/s"))
    c4.metric(
        "波高",
        fmt(preview.get("wave_height"), 0, "cm")
    )

    c5, c6, c7 = st.columns(3)

    c5.metric(
        "天候",
        WEATHER.get(
            preview.get("weather_number"), "—"
        ),
    )
    c6.metric(
        "気温",
        fmt(preview.get("air_temperature"), 1, "℃")
    )
    c7.metric(
        "水温",
        fmt(preview.get("water_temperature"), 1, "℃")
    )

    st.caption(
        "風向の東西南北はAPIで定義された相対表記です。"
        "実際の地理上の方位ではありません。"
    )

    racers = preview.get("racers") or {}

    if not racers:
        st.info("展示データは未取得です。")
        return

    rows = []

    for lane in range(1, 7):
        racer = racers.get(str(lane), {})

        rows.append({
            "艇番": lane,
            "進入コース": racer.get("course_number"),
            "展示タイム": num(
                racer.get("exhibition_time")
            ),
            "展示ST": start_text(
                racer.get("start_timing"),
                racer.get("start_timing_source"),
            ),
            "チルト": num(
                racer.get("tilt_adjustment")
            ),
            "体重": num(racer.get("weight")),
        })

    df = pd.DataFrame(rows)

    styled = (
        df.style
        .apply(highlight_best, subset=["展示タイム"])
        .format({
            "展示タイム": "{:.2f}",
            "チルト": "{:.1f}",
            "体重": "{:.1f}",
        }, na_rep="—")
    )

    st.dataframe(
        styled,
        hide_index=True,
        use_container_width=True,
    )

    valid = df.dropna(subset=["展示タイム"])

    if not valid.empty:
        best = valid["展示タイム"].min()

        winners = valid.loc[
            abs(valid["展示タイム"] - best) < 1e-8,
            "艇番",
        ].tolist()

        st.success(
            "展示タイム最速："
            + "・".join(
                f"{lane}号艇" for lane in winners
            )
            + f"（{best:.2f}秒）"
        )

    st.caption(
        "展示ST・進入コースはスタート展示の記録です。"
        "本番のST・進入とは異なる場合があります。"
    )


# ==========================================
# WEB-13 締切カウントダウン・自動更新
# ==========================================

@st.fragment(run_every="5s")
def show_race_clock(closed_at, race_key, auto_enabled):
    now = datetime.now(JST)
    deadline = None

    if closed_at:
        try:
            deadline = datetime.fromisoformat(
                str(closed_at).replace(" ", "T")
            )

            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=JST)
            else:
                deadline = deadline.astimezone(JST)

        except (ValueError, TypeError):
            deadline = None

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "現在時刻（日本時間）",
        now.strftime("%H:%M:%S"),
    )

    c2.metric(
        "締切予定",
        deadline.strftime("%H:%M:%S")
        if deadline else "未取得",
    )

    remaining = None

    if deadline is None:
        c3.metric("締切まで", "時刻未取得")
    else:
        remaining = (
            deadline - now
        ).total_seconds()

        if remaining <= 0:
            c3.metric("締切まで", "予定時刻経過")
        else:
            seconds = int(remaining)
            hours, rem = divmod(seconds, 3600)
            minutes, seconds = divmod(rem, 60)

            if hours:
                display = (
                    f"{hours:02d}:"
                    f"{minutes:02d}:"
                    f"{seconds:02d}"
                )
            else:
                display = (
                    f"{minutes:02d}:"
                    f"{seconds:02d}"
                )

            c3.metric("締切まで", display)

    last_refresh = st.session_state.get(
        "web13_last_refresh"
    )

    if isinstance(last_refresh, datetime):
        st.caption(
            "最終再取得操作："
            + last_refresh.astimezone(JST).strftime(
                "%Y/%m/%d %H:%M:%S"
            )
        )
    else:
        st.caption("最終再取得操作：未実施")

    st.caption(
        "締切は非公式APIの予定時刻です。"
        "変更・遅延の可能性があります。"
        "実際の締切は公式サイトで確認してください。"
    )

    # 自動更新は当日の未締切レースだけ
    if not auto_enabled:
        return

    if race_key[0] != now.date().isoformat():
        return

    if remaining is not None and remaining <= 0:
        return

    if last_refresh is None:
        st.session_state["web13_last_refresh"] = now
        return

    elapsed = (
        now - last_refresh
    ).total_seconds()

    # 180秒未満では再取得しない
    if elapsed < 180:
        return

    # 更新時刻を先に記録して二重アクセスを防ぐ
    st.session_state["web13_last_refresh"] = now

    fetch_daily.clear()
    web11.download_odds.clear()

    st.session_state["web12_odds_ready"] = race_key

    st.rerun()


# ==========================================
# 統合画面
# ==========================================

def main():
    st.title("🚤 当日レース統合分析")
    st.caption(
        "BOAT RACE AI | WEB-12 "
        "出走表・展示・気象・公式オッズ・舟券シミュレーター"
    )

    today = datetime.now(JST).date()

    c1, c2, c3 = st.columns(3)

    with c1:
        day = st.date_input(
            "開催日",
            value=today,
            min_value=max(
                today - timedelta(days=7),
                datetime(2026, 1, 1).date(),
            ),
            max_value=today + timedelta(days=3),
            format="YYYY/MM/DD",
            key="web12_day",
        )

    with c2:
        venue = st.selectbox(
            "競艇場",
            list(VENUES),
            format_func=lambda x: VENUES[x],
            key="web12_venue",
        )

    with c3:
        race = st.selectbox(
            "レース",
            list(range(1, 13)),
            format_func=lambda x: f"{x}R",
            key="web12_race",
        )

    race_key = (day.isoformat(), venue, race)
    if st.session_state.get("web13_race_key") != race_key:
        st.session_state["web13_race_key"] = race_key
        st.session_state["web13_last_refresh"] = (
            datetime.now(JST)
        )
        
    # WEB-11の金額設定をレース別に維持するため、
    # 既存の選択レース情報と連携する
    st.session_state["odds_selection"] = race_key

    st.markdown(
        f"### {day:%Y/%m/%d}｜"
        f"{VENUES[venue]} {race}R"
    )

    info_col, odds_col = st.columns(2)

    with info_col:
        st.link_button(
            "公式出走表",
            (
                "https://www.boatrace.jp/owpc/pc/race/racelist"
                f"?hd={day:%Y%m%d}&jcd={venue:02d}&rno={race}"
            ),
            use_container_width=True,
        )

    with odds_col:
        st.link_button(
            "公式3連単オッズ",
            web11.official_url(day, venue, race),
            use_container_width=True,
        )

    # 一度の操作で両ソースのキャッシュを更新
    if st.button(
        "🔄 出走表・展示・オッズを再取得",
        use_container_width=True,
    ):
        fetch_daily.clear()
        web11.download_odds.clear()

        st.session_state["web12_odds_ready"] = race_key

        st.session_state["web13_last_refresh"] = (
            datetime.now(JST)
        )

        st.rerun()
        fetch_daily.clear()
        web11.download_odds.clear()
        st.session_state["web12_odds_ready"] = race_key
        st.rerun()

    data = {}

    try:
        data = fetch_daily(day.isoformat())
    except requests.RequestException as exc:
        st.warning(
            f"出走表APIへの接続に失敗しました: {exc}"
        )

    race_data = get_race(data, venue, race)

    if race_data:
        st.success("出走表データを取得しました。")
    else:
        st.info(
            "この日付・競艇場・レースの出走表データは"
            "未取得です。公式出走表を確認してください。"
        )

    # 締切予定時刻
    closed_at = (
        race_data.get("closed_at")
        if race_data else None
    )

    st.markdown("### ⏱️ レース締切・更新管理")

    auto_enabled = st.toggle(
        "180秒間隔でデータを自動更新する",
        value=False,
        key="web13_auto_refresh",
    )

    show_race_clock(
        closed_at=closed_at,
        race_key=race_key,
        auto_enabled=auto_enabled,
    )

    # 公式オッズを選択レースごとに取得
    odds = {}

    if st.session_state.get(
        "web12_odds_ready"
    ) != race_key:
        if st.button(
            "📥 このレースの公式3連単オッズを取得",
            type="primary",
        ):
            st.session_state["web12_odds_ready"] = race_key
            st.rerun()

    if st.session_state.get(
        "web12_odds_ready"
    ) == race_key:
        try:
            odds = web11.download_odds(*race_key)
        except Exception as exc:
            st.warning(
                f"公式オッズ取得に失敗しました: {exc}"
            )

    st.caption(
        f"公式オッズ取得件数：{len(odds)}/120通り。"
        "発売前・更新遅延・発売終了などでは"
        "取得できない場合があります。"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚤 出走表",
        "🌤️ 展示・気象",
        "📊 3連単オッズ",
        "🎯 組み合わせ指定",
        "🎫 フォーメーション",
    ])

    with tab1:
        if race_data:
            show_entries(race_data)
        else:
            st.info("出走表データなし")

    with tab2:
        if race_data:
            show_preview(race_data)
        else:
            st.info("展示・気象データなし")

    with tab3:
        web11.show_odds(odds)

    with tab4:
        web11.show_single_combo(odds)

    with tab5:
        web11.show_formation(odds)

    st.divider()
    st.caption(
        "出走表・展示は非公式Boatrace Open API、"
        "オッズはBOAT RACE公式ページから取得。"
        "購入予定金額はシミュレーション専用です。"
        "実際の投票処理は行いません。"
    )


if __name__ == "__main__":
    main()
