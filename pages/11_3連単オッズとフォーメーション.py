
from datetime import datetime, timedelta
from itertools import permutations
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
import re

import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="3連単オッズ・フォーメーション",
    page_icon="🎫",
    layout="wide",
)

JST = ZoneInfo("Asia/Tokyo")
ALL_COMBOS = [
    "-".join(map(str, p))
    for p in permutations(range(1, 7), 3)
]

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


def official_url(day, venue, race):
    return (
        "https://www.boatrace.jp/owpc/pc/race/odds3t"
        f"?hd={day:%Y%m%d}&jcd={venue:02d}&rno={race}"
    )


def odds_api_url(day):
    return (
        "https://lamrongol.github.io/BoatraceOdds"
        f"/v3/{day.year}/{day:%Y%m%d}.json"
    )


def valid_combo(text):
    text = str(text).strip()

    if not re.fullmatch(r"[1-6]-[1-6]-[1-6]", text):
        return False

    return len(set(text.split("-"))) == 3


def numeric_odds(value):
    if isinstance(value, dict):
        for key in ("odds", "value", "rate"):
            if key in value:
                return numeric_odds(value[key])
        return None

    try:
        result = float(str(value).replace(",", ""))
        if 0.0 <= result < 1000000:
            return result
    except (TypeError, ValueError):
        pass

    return None


def parse_trifecta(raw):
    """3段階キー形式やフラット形式を解析する"""

    result = {}

    if isinstance(raw, dict):
        for first, second_layer in raw.items():

            if valid_combo(first):
                value = numeric_odds(second_layer)
                if value is not None:
                    result[first] = value
                continue

            if str(first) not in "123456":
                continue

            if not isinstance(second_layer, dict):
                continue

            for second, third_layer in second_layer.items():
                if str(second) not in "123456":
                    continue

                if not isinstance(third_layer, dict):
                    continue

                for third, value in third_layer.items():
                    combo = f"{first}-{second}-{third}"

                    if not valid_combo(combo):
                        continue

                    odds = numeric_odds(value)

                    if odds is not None:
                        result[combo] = odds

    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue

            combo = (
                item.get("combination")
                or item.get("number")
            )

            if not valid_combo(combo):
                continue

            odds = numeric_odds(item)

            if odds is not None:
                result[str(combo)] = odds

    return result


def find_race_odds(data, venue, race):
    """既知の候補階層から対象レースのオッズを探す"""

    if not isinstance(data, dict):
        return {}

    venue_key = str(venue)
    race_key = str(race)

    roots = [
        data.get("programs", {}),
        data.get("odds", {}),
        data,
    ]

    candidates = []

    for root in roots:
        if not isinstance(root, dict):
            continue

        stadiums = root.get("stadiums", root)

        if not isinstance(stadiums, dict):
            continue

        stadium = (
            stadiums.get(venue_key)
            or stadiums.get(f"{venue:02d}")
        )

        if not isinstance(stadium, dict):
            continue

        races = stadium.get("races", stadium)

        if not isinstance(races, dict):
            continue

        race_data = races.get(race_key)

        if isinstance(race_data, dict):
            candidates.append(race_data)

    for race_data in candidates:
        odds_data = race_data.get("odds", race_data)

        if isinstance(odds_data, dict):
            trifecta = odds_data.get("trifecta")

            if trifecta is not None:
                parsed = parse_trifecta(trifecta)
                if parsed:
                    return parsed

    return {}




@st.cache_data(ttl=180, show_spinner=False)
def download_odds(day_iso, venue, race):
    """
    BOAT RACE公式サイトから3連単オッズを取得する。
    既存DBやAIモデルへの書き込みは行わない。
    """
    day = datetime.strptime(
        day_iso, "%Y-%m-%d"
    ).date()

    url = official_url(day, venue, race)

    response = requests.get(
        url,
        timeout=20,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html",
        },
    )

    response.raise_for_status()
    response.encoding = response.apparent_encoding

    soup = BeautifulSoup(
        response.text, "html.parser"
    )

    # 3連単のオッズ表を探す
    target = None

    table_info = []

    for i, table in enumerate(soup.select("table")):
        points = table.select("td.oddsPoint")

        table_info.append(
            f"table{i}: oddsPoint={len(points)}"
        )

        if len(points) >= 100:
            target = table
            break

    if target is None:
        raise RuntimeError(
            "公式HTMLからオッズ表を検出できません。"
            " / ".join(table_info)
        )


    if target is None:
        raise RuntimeError(
            "公式ページに3連単オッズ表が見つかりません。"
            "未発売・発売終了・HTML変更の可能性があります。"
        )

    # 1行につき1着艇別のオッズが6列並ぶ
    rows = []

    for tr in target.select("tr"):
        cells = tr.select("td.oddsPoint")

        if not cells:
            continue

        if len(cells) != 6:
            raise RuntimeError(
                "オッズ表の列数が想定と異なります。"
                f"取得列数: {len(cells)}"
            )

        rows.append([
            cell.get_text(" ", strip=True)
            for cell in cells
        ])

    if len(rows) != 20:
        raise RuntimeError(
            "オッズ表の行数が想定と異なります。"
            f"取得行数: {len(rows)}"
        )

    # 転置することで、1着艇ごとに
    # 20通りずつの並びに変換する
    odds_by_first = list(zip(*rows))

    result = {}

    for first in range(1, 7):
        values = odds_by_first[first - 1]
        index = 0

        for second in range(1, 7):
            if second == first:
                continue

            for third in range(1, 7):
                if third in (first, second):
                    continue

                raw = values[index]
                index += 1

                odds = numeric_odds(raw)

                if odds is not None:
                    combination = (
                        f"{first}-{second}-{third}"
                    )
                    result[combination] = odds

    # 120通りを超えることはない
    if len(result) > 120:
        raise RuntimeError(
            "取得件数が120通りを超えました"
        )
        
    if len(result) != 120:
        raise RuntimeError(
            f"3連単オッズ取得数が不完全です: "
            f"{len(result)}/120件。"
            " HTML表の構造を確認してください。"
        )


    return result



def parse_manual(text):
    """
    1-2-3,12.5
    1-3-2,18.9
    の形式で入力
    """
    result = {}

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = re.split(r"[,\\t ]+", line)

        if len(parts) != 2:
            continue

        combo, raw_odds = parts

        if not valid_combo(combo):
            continue

        odds = numeric_odds(raw_odds)

        if odds is not None:
            result[combo] = odds

    return result


def formation(first, second, third):
    combos = []

    for a in first:
        for b in second:
            for c in third:
                if len({a, b, c}) == 3:
                    combos.append(f"{a}-{b}-{c}")

    return sorted(set(combos))


def show_odds(odds):
    st.subheader("3連単オッズ一覧")

    df = pd.DataFrame({
        "組み合わせ": ALL_COMBOS,
    })

    df["オッズ"] = df["組み合わせ"].map(odds)

    valid = df.dropna(subset=["オッズ"]).copy()

    if valid.empty:
        st.info(
            "現在、表示できるオッズがありません。"
            "公式オッズを確認するか、"
            "手動入力を使用してください。"
        )
        return

    order = st.radio(
        "並び替え",
        ["人気順（低オッズ順）", "高配当順", "組み合わせ順"],
        horizontal=True,
    )

    if order.startswith("人気"):
        df = df.sort_values(
            "オッズ", ascending=True,
            na_position="last"
        )
    elif order.startswith("高配当"):
        df = df.sort_values(
            "オッズ", ascending=False,
            na_position="last"
        )

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        f"取得済みオッズ：{len(valid)} / 120通り。"
        "空欄は未取得であり、オッズ0倍ではありません。"
    )


def show_single_combo(odds):
    st.subheader("指定した組み合わせのオッズ")

    c1, c2, c3 = st.columns(3)

    with c1:
        a = st.selectbox("1着", range(1, 7), key="single1")
    with c2:
        b = st.selectbox("2着", range(1, 7), index=1, key="single2")
    with c3:
        c = st.selectbox("3着", range(1, 7), index=2, key="single3")

    if len({a, b, c}) != 3:
        st.warning("1着・2着・3着には異なる艇を選んでください。")
        return

    combo = f"{a}-{b}-{c}"
    value = odds.get(combo)

    st.metric("組み合わせ", combo)

    if value is None:
        st.info("この組み合わせのオッズは未取得です。")
    else:
        st.metric("現在取得できているオッズ", f"{value:.1f}倍")



def show_formation(odds):
    st.subheader("🎫 3連単フォーメーション作成")

    c1, c2, c3 = st.columns(3)

    with c1:
        first = st.multiselect(
            "1着候補",
            range(1, 7),
            default=[1],
            key="form_first",
        )

    with c2:
        second = st.multiselect(
            "2着候補",
            range(1, 7),
            default=[2, 3],
            key="form_second",
        )

    with c3:
        third = st.multiselect(
            "3着候補",
            range(1, 7),
            default=[2, 3, 4],
            key="form_third",
        )

    combos = formation(first, second, third)

    if not combos:
        st.warning("有効な買い目がありません。")
        return

    # 全点一律の初期金額
    default_stake = st.number_input(
        "初期購入金額（1点あたり）",
        min_value=0,
        max_value=100000,
        value=100,
        step=100,
        key="web11_default_stake",
    )

    # 組み合わせを変更しても金額を保持
    if "web11_stakes" not in st.session_state:
        st.session_state["web11_stakes"] = {}

    stake_map = st.session_state["web11_stakes"]

    # レース別に金額を管理
    race_key = st.session_state.get("odds_selection")

    if race_key is None:
        race_key = ("未指定", 0, 0)

    race_id = "|".join(map(str, race_key))

    rows = []

    for combo in combos:
        key = (race_id, combo)

        rows.append({
            "買い目": combo,
            "オッズ": odds.get(combo),
            "購入金額（円）": stake_map.get(
                key, int(default_stake)
            ),
        })

    table = pd.DataFrame(rows)
    def odds_status(value):
        if value is None or pd.isna(value):
            return "未取得"

        if float(value) == 0:
            return "要確認（0.0倍）"

        if float(value) > 0:
            return "有効"

        return "要確認"

    table["状態"] = table["オッズ"].apply(
        odds_status
    )
    
    st.markdown("#### 買い目ごとの購入金額")

    st.caption(
        "購入金額のセルをクリックして編集できます。"
        "0円にすると購入対象から除外します。"
        "100円単位で入力してください。"
    )

    # 組み合わせごとに編集状態を分離
    editor_key = (
        f"web11_editor_{race_id}_"
        + "_".join(combos)
    )

    edited = st.data_editor(
        table,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=["買い目", "オッズ","状態"],
        column_config={
            "買い目": st.column_config.TextColumn(
                "3連単",
                width="small",
            ),
            "オッズ": st.column_config.NumberColumn(
                "オッズ（倍）",
                format="%.1f",
            ),
            "購入金額（円）": st.column_config.NumberColumn(
                "購入金額（円）",
                min_value=0,
                max_value=100000,
                step=100,
                format="%d",
                required=True,
            ),
        },
        key=editor_key,
    )

    # 金額を数値として検証
    amounts = pd.to_numeric(
        edited["購入金額（円）"],
        errors="coerce",
    )

    if amounts.isna().any():
        st.error("購入金額に数値以外が含まれています。")
        return

    if (
        (amounts < 0).any()
        or (amounts % 100 != 0).any()
    ):
        st.error(
            "購入金額は0円または100円単位で"
            "設定してください。"
        )
        return

    edited["購入金額（円）"] = amounts.astype(int)

    # 設定を記録
    for _, row in edited.iterrows():
        stake_map[
            (race_id, row["買い目"])
        ] = int(row["購入金額（円）"])

    st.session_state["web11_stakes"] = stake_map

    # 0円の買い目は購入予定から除外
    active = edited[
        edited["購入金額（円）"] > 0
    ].copy()

    total = int(
        active["購入金額（円）"].sum()
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "購入予定点数",
        f"{len(active)}点",
    )

    c2.metric(
        "購入予定金額",
        f"{total:,}円",
    )

    c3.metric(
        "除外した点数",
        f"{len(edited) - len(active)}点",
    )

    if active.empty:
        st.info("購入予定の買い目がありません。")
        return

    # 0.0倍・未取得のオッズは
    # 払戻額を計算しない
    valid_odds = pd.to_numeric(
        active["オッズ"],
        errors="coerce"
    )

    valid_odds = valid_odds.where(
        valid_odds > 0
    )

    active["仮払戻額（円）"] = (
        valid_odds
        * active["購入金額（円）"]
    ).round()

    invalid_count = (
        active["仮払戻額（円）"].isna().sum()
    )

    if invalid_count > 0:
        st.warning(
            f"{invalid_count}点の買い目は"
            "オッズが0.0倍または未取得のため、"
            "仮払戻額を計算していません。"
            "購入可否・払戻条件を公式サイトで"
            "確認してください。"
        )

    active["仮収支（円）"] = (
        active["仮払戻額（円）"] - total
    )

    st.markdown("#### 購入予定一覧")

    st.dataframe(
        active,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "仮払戻額は、その組み合わせが的中し、"
        "表示中のオッズで払戻されたと仮定した金額。"
        "仮収支は他の買い目を含む総購入金額を"
        "差し引いた値です。"
    )

    # 最も払戻額が大きいケースではなく、
    # 全買い目の合計金額を基準に試算する
    known = active.dropna(
        subset=["仮払戻額（円）"]
    )

    if not known.empty:
        best = known.loc[
            known["仮払戻額（円）"].idxmax()
        ]

        st.info(
            f"現在取得できているオッズのうち、"
            f"最大仮払戻額は"
            f"{best['買い目']} の"
            f"{best['仮払戻額（円）']:,.0f}円です。"
            "これは的中の可能性を表すものではありません。"
        )

    csv_data = active.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "購入予定一覧をCSV保存",
        data=csv_data,
        file_name="boat_formation.csv",
        mime="text/csv",
    )

    st.warning(
        "この画面は購入予定の試算専用です。"
        "実際の投票は行いません。"
        "オッズと締切時刻は公式サイトで確認してください。"
    )


def main():
    st.title("🎫 3連単オッズ・フォーメーション")
    st.caption("BOAT RACE AI | WEB-11")

    st.info(
        "この画面ではオッズ確認と買い目の試算のみ行います。"
        "実際の舟券購入や投票処理は実行しません。"
    )

    today = datetime.now(JST).date()

    c1, c2, c3 = st.columns(3)

    with c1:
        day = st.date_input(
            "日付",
            value=today,
            min_value=today - timedelta(days=7),
            max_value=today + timedelta(days=3),
            format="YYYY/MM/DD",
        )

    with c2:
        venue = st.selectbox(
            "競艇場",
            list(VENUES),
            format_func=lambda v: VENUES[v],
        )

    with c3:
        race = st.selectbox(
            "レース番号",
            range(1, 13),
            format_func=lambda r: f"{r}R",
        )

    st.link_button(
        "BOAT RACE公式3連単オッズ",
        official_url(day, venue, race),
    )


    c1, c2 = st.columns(2)

    with c1:
        get_clicked = st.button(
            "📥 オッズを取得",
            type="primary",
            use_container_width=True,
        )

    with c2:
        refresh_clicked = st.button(
            "🔄 最新オッズを再取得",
            use_container_width=True,
        )

    if get_clicked or refresh_clicked:
        st.session_state["odds_selection"] = (
            day.isoformat(),
            venue,
            race,
        )

        if refresh_clicked:
            download_odds.clear()

    selected = st.session_state.get(
        "odds_selection"
    )

    odds = {}
    source = "未取得"

    if selected:
        if selected != (day.isoformat(), venue, race):
            st.warning(
                "選択中のレースが前回の取得対象と異なります。"
                "「オッズを取得」を押してください。"
            )
        else:
            try:
                odds = download_odds(*selected)
                if odds:
                    source = "BOAT RACE公式サイト"
                else:
                    st.warning(
                        "オッズが未公開、または"
                        "取得形式を認識できませんでした。"
                    )
            except Exception as exc:
                st.warning(f"オッズ取得に失敗しました: {exc}")

    with st.expander("オッズを手動入力する"):
        manual = st.text_area(
            "組み合わせ,オッズ（1行に1点）",
            placeholder="1-2-3,12.5\n1-3-2,18.9",
            height=120,
        )

        manual_odds = parse_manual(manual)

        if manual_odds:
            odds.update(manual_odds)
            source = (
                "公式サイト＋手動入力"
                if source != "未取得"
                else "手動入力"
            )

    st.caption(
    f"オッズ取得元：{source}。"
    "公式ページの表示値を取得していますが、"
    "最終オッズとは異なる場合があります。"
    "実際の投票前には公式ページで再確認してください。"
    )

    tabs = st.tabs([
        "120通りオッズ",
        "指定組み合わせ",
        "フォーメーション",
    ])

    with tabs[0]:
        show_odds(odds)

    with tabs[1]:
        show_single_combo(odds)

    with tabs[2]:
        show_formation(odds)


if __name__ == "__main__":
    main()
