
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import importlib.util
import html

import pandas as pd
import requests
import streamlit as st


# ==========================================
# WEB-14 実戦用統合ダッシュボード
# ==========================================

JST = ZoneInfo("Asia/Tokyo")

BOAT_COLORS = {
    1: ("#FFFFFF", "#111827"),
    2: ("#111827", "#FFFFFF"),
    3: ("#EF4444", "#FFFFFF"),
    4: ("#2563EB", "#FFFFFF"),
    5: ("#FACC15", "#111827"),
    6: ("#16A34A", "#FFFFFF"),
}


# ==========================================
# WEB-12 / WEB-11 の既存処理を再利用
# ==========================================

def load_web12():
    path = Path(__file__).with_name(
        "12_当日レース統合分析.py"
    )

    if not path.exists():
        st.error(
            "WEB-12のファイルが見つかりません。\n"
            f"{path}"
        )
        st.stop()

    spec = importlib.util.spec_from_file_location(
        "web12_for_web14",
        str(path),
    )

    if spec is None or spec.loader is None:
        st.error("WEB-12を読み込めませんでした。")
        st.stop()

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


web12 = load_web12()
web11 = web12.web11

VENUES = web12.VENUES


# ==========================================
# レース選択
# ==========================================

def race_selector():
    today = datetime.now(JST).date()

    c1, c2, c3 = st.columns(3)

    with c1:
        day = st.date_input(
            "開催日",
            value=today,
            min_value=today - timedelta(days=7),
            max_value=today + timedelta(days=3),
            format="YYYY/MM/DD",
            key="web14_day",
        )

    with c2:
        venue = st.selectbox(
            "競艇場",
            list(VENUES),
            format_func=lambda x: VENUES[x],
            key="web14_venue",
        )

    with c3:
        race = st.selectbox(
            "レース",
            list(range(1, 13)),
            format_func=lambda x: f"{x}R",
            key="web14_race",
        )

    race_key = (day.isoformat(), venue, race)

    # WEB-11と同じレース別資金管理を利用
    st.session_state["odds_selection"] = race_key

    # WEB-13のカウントダウン・自動更新と連携
    if st.session_state.get("web13_race_key") != race_key:
        st.session_state["web13_race_key"] = race_key
        st.session_state["web13_last_refresh"] = (
            datetime.now(JST)
        )

    return day, venue, race, race_key


# ==========================================
# 公式リンク
# ==========================================

def show_links(day, venue, race):
    racelist_url = (
        "https://www.boatrace.jp/owpc/pc/race/racelist"
        f"?hd={day:%Y%m%d}&jcd={venue:02d}&rno={race}"
    )

    c1, c2 = st.columns(2)

    with c1:
        st.link_button(
            "公式出走表を開く",
            racelist_url,
            use_container_width=True,
        )

    with c2:
        st.link_button(
            "公式3連単オッズを開く",
            web11.official_url(day, venue, race),
            use_container_width=True,
        )


# ==========================================
# 手動更新
# ==========================================

def refresh_controls(race_key):
    c1, c2 = st.columns([2, 1])

    with c1:
        refresh_clicked = st.button(
            "🔄 出走表・展示・オッズを再取得",
            use_container_width=True,
            key="web14_refresh",
        )

    with c2:
        auto_enabled = st.toggle(
            "180秒自動更新",
            value=False,
            key="web13_auto_refresh",
        )

    if refresh_clicked:
        web12.fetch_daily.clear()
        web11.download_odds.clear()

        st.session_state["web12_odds_ready"] = race_key
        st.session_state["web13_last_refresh"] = (
            datetime.now(JST)
        )

        st.rerun()

    return auto_enabled


# ==========================================
# データ取得
# ==========================================

def fetch_selected_race(day, venue, race):
    try:
        data = web12.fetch_daily(day.isoformat())

        if not isinstance(data, dict):
            return {}, "出走表APIの応答形式が不正です。"

        race_data = web12.get_race(
            data, venue, race
        )

        return race_data or {}, None

    except requests.RequestException as exc:
        return {}, f"出走表API接続エラー: {exc}"

    except Exception as exc:
        return {}, f"出走表データ解析エラー: {exc}"


def fetch_selected_odds(race_key):
    if st.session_state.get(
        "web12_odds_ready"
    ) != race_key:
        return {}, None

    try:
        odds = web11.download_odds(*race_key)
        return odds, None

    except Exception as exc:
        return {}, f"公式オッズ取得エラー: {exc}"


# ==========================================
# 艇番カラー
# ==========================================

def boat_chip(number):
    number = int(number)

    bg, fg = BOAT_COLORS[number]

    return (
        "<span style='"
        "display:inline-block;"
        "min-width:27px;"
        "padding:3px 7px;"
        "margin-right:3px;"
        "border-radius:5px;"
        f"background:{bg};"
        f"color:{fg};"
        "border:1px solid #94A3B8;"
        "font-weight:700;"
        "text-align:center;'>"
        f"{number}</span>"
    )


def combo_html(combo):
    parts = str(combo).split("-")

    if len(parts) != 3:
        return html.escape(str(combo))

    return (
        boat_chip(parts[0])
        + "<span style='margin:0 3px;'>-</span>"
        + boat_chip(parts[1])
        + "<span style='margin:0 3px;'>-</span>"
        + boat_chip(parts[2])
    )



# ==========================================
# WEB-14.1 オッズから個別買い目を選択
# ==========================================

def race_id_text(race_key):
    return "|".join(map(str, race_key))


def show_odds_panel(odds, selected_combos, race_key):
    st.subheader("📊 公式3連単オッズ")

    race_id = race_id_text(race_key)

    if "web14_candidates" not in st.session_state:
        st.session_state["web14_candidates"] = {}

    candidates_by_race = st.session_state["web14_candidates"]
    selected = set(candidates_by_race.get(race_id, []))

    c1, c2 = st.columns(2)

    c1.metric(
        "有効オッズ",
        sum(1 for x in odds.values() if x is not None and x > 0),
    )
    c2.metric(
        "0.0倍・要確認",
        sum(1 for x in odds.values() if x == 0),
    )

    sort_mode = st.selectbox(
        "並び替え",
        ["人気順", "高配当順", "組み合わせ順"],
        key="web14_sort",
    )

    only_selected = st.checkbox(
        "チェックした買い目だけ表示",
        key="web14_only_selected",
    )

    show_zero = st.checkbox(
        "0.0倍も表示する",
        value=True,
        key="web14_show_zero",
    )

    rows = []

    for combo in web11.ALL_COMBOS:
        odd = odds.get(combo)

        if odd is None:
            status = "未取得"
        elif odd <= 0:
            status = "要確認"
        else:
            status = "有効"

        rows.append({
            "候補": combo in selected,
            "買い目": combo,
            "オッズ": odd,
            "状態": status,
            "フォーメーション": (
                "●" if combo in selected_combos else ""
            ),
        })

    df = pd.DataFrame(rows)

    if only_selected:
        df = df[df["候補"]].copy()

    if not show_zero:
        df = df[df["オッズ"] != 0].copy()

    if sort_mode == "人気順":
        df = df.sort_values(
            "オッズ", ascending=True, na_position="last"
        )
    elif sort_mode == "高配当順":
        df = df.sort_values(
            "オッズ", ascending=False, na_position="last"
        )
    else:
        df = df.sort_values("買い目")

    df = df.reset_index(drop=True)

    st.caption(
        "左端のチェックボックスを押すと、"
        "右側の『個別選択した買い目』へ追加されます。"
        "●はフォーメーションに含まれる買い目です。"
    )

    # 並び替え・絞り込みごとに編集状態を区別
    # 候補そのものは別のセッション変数に保存
    editor_key = (
        f"web14_odds_editor_{race_id}_"
        f"{sort_mode}_{int(only_selected)}_"
        f"{int(show_zero)}"
    )

    edited = st.data_editor(
        df,
        hide_index=True,
        num_rows="fixed",
        use_container_width=True,
        height=540,
        disabled=[
            "買い目",
            "オッズ",
            "状態",
            "フォーメーション",
        ],
        column_config={
            "候補": st.column_config.CheckboxColumn(
                "候補",
                help="購入候補に追加・解除",
                required=True,
                width="small",
            ),
            "買い目": st.column_config.TextColumn(
                "3連単",
                width="small",
            ),
            "オッズ": st.column_config.NumberColumn(
                "オッズ",
                format="%.1f",
            ),
            "状態": st.column_config.TextColumn(
                "状態",
                width="small",
            ),
            "フォーメーション":
                st.column_config.TextColumn(
                    "組込",
                    width="small",
                ),
        },
        key=editor_key,
    )

    # 編集された「表示中の行」だけを反映。
    # フィルター外の選択はそのまま保持する。
    for _, row in edited.iterrows():
        combo = row["買い目"]

        if bool(row["候補"]):
            selected.add(combo)
        else:
            selected.discard(combo)

    candidates_by_race[race_id] = sorted(selected)
    st.session_state["web14_candidates"] = candidates_by_race

    st.caption(
        f"個別選択した買い目：{len(selected)}点。"
        "0.0倍・未取得も候補にできますが、"
        "仮払戻額は計算しません。"
    )

    return sorted(selected)



# ==========================================
# WEB-14.2 個別選択した買い目（表示専用）
# ==========================================

def show_candidate_editor(odds, race_key):
    st.subheader("✅ 個別選択した買い目")

    race_id = race_id_text(race_key)

    candidates = st.session_state.get(
        "web14_candidates", {}
    ).get(race_id, [])

    if not candidates:
        st.info(
            "左側の公式オッズ一覧で、"
            "購入候補にチェックを入れてください。"
        )
        return

    rows = []

    for combo in candidates:
        odd = odds.get(combo)

        if odd is None:
            status = "未取得"
        elif odd <= 0:
            status = "要確認"
        else:
            status = "有効"

        rows.append({
            "3連単": combo,
            "オッズ": odd,
            "状態": status,
        })

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            "オッズ": st.column_config.NumberColumn(
                "オッズ",
                format="%.1f",
            ),
        },
    )

    st.caption(
        f"選択中：{len(candidates)}点。"
        "金額は下の購入予定サマリーで設定します。"
    )


# ==========================================
# WEB-14.2 フォーメーション（表示専用）
# ==========================================

def show_formation_preview(odds):
    st.subheader("🎫 3連単フォーメーション")

    c1, c2, c3 = st.columns(3)

    with c1:
        first = st.multiselect(
            "1着",
            range(1, 7),
            default=[1],
            key="form_first",
        )

    with c2:
        second = st.multiselect(
            "2着",
            range(1, 7),
            default=[2, 3],
            key="form_second",
        )

    with c3:
        third = st.multiselect(
            "3着",
            range(1, 7),
            default=[2, 3, 4],
            key="form_third",
        )

    combos = web11.formation(
        first, second, third
    )

    if not combos:
        st.warning("有効な組み合わせがありません。")
        return

    rows = []

    for combo in combos:
        odd = odds.get(combo)

        if odd is None:
            status = "未取得"
        elif odd <= 0:
            status = "要確認"
        else:
            status = "有効"

        rows.append({
            "3連単": combo,
            "オッズ": odd,
            "状態": status,
        })

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            "オッズ": st.column_config.NumberColumn(
                "オッズ",
                format="%.1f",
            ),
        },
    )

    st.caption(
        f"フォーメーション：{len(combos)}点。"
        "購入金額は下のサマリーで変更できます。"
    )



# ==========================================
# 購入予定サマリー（下側）
# ==========================================

def get_selected_combos():
    first = st.session_state.get(
        "form_first", [1]
    )

    second = st.session_state.get(
        "form_second", [2, 3]
    )

    third = st.session_state.get(
        "form_third", [2, 3, 4]
    )

    return web11.formation(
        first, second, third
    )




# ==========================================
# WEB-14.2 購入予定サマリー（金額編集）
# ==========================================

def show_purchase_summary(odds, race_key):
    st.subheader("💰 購入予定サマリー")

    race_id = race_id_text(race_key)

    formation_combos = set(
        get_selected_combos()
    )

    candidates = set(
        st.session_state.get(
            "web14_candidates", {}
        ).get(race_id, [])
    )

    # フォーメーションと個別選択を統合
    # 同じ買い目は一度だけ表示する
    all_combos = sorted(
        formation_combos | candidates,
        key=lambda x: tuple(
            map(int, x.split("-"))
        ),
    )

    if not all_combos:
        st.info(
            "購入候補がありません。"
            "オッズ一覧かフォーメーションで"
            "買い目を選択してください。"
        )
        return

    # 金額の唯一の編集先
    if "web14_summary_stakes" not in st.session_state:
        st.session_state["web14_summary_stakes"] = {}

    stake_map = st.session_state[
        "web14_summary_stakes"
    ]

    # 以前の入力金額があれば初期値として引き継ぐ
    old_formation = st.session_state.get(
        "web11_stakes", {}
    )
    old_candidates = st.session_state.get(
        "web14_candidate_stakes", {}
    )

    rows = []

    for combo in all_combos:
        key = (race_id, combo)

        if combo in candidates:
            origin = (
                "両方"
                if combo in formation_combos
                else "個別選択"
            )

            previous_amount = old_candidates.get(
                key,
                old_formation.get(key, 100),
            )
        else:
            origin = "フォーメーション"
            previous_amount = old_formation.get(
                key, 100
            )

        amount = stake_map.get(
            key,
            previous_amount,
        )

        odd = odds.get(combo)

        if odd is None:
            status = "未取得"
        elif odd <= 0:
            status = "要確認"
        else:
            status = "有効"

        rows.append({
            "買い目": combo,
            "選択方法": origin,
            "オッズ": odd,
            "状態": status,
            "購入金額（円）": int(amount),
        })

    df = pd.DataFrame(rows)

    st.caption(
        "購入金額を直接クリックして編集できます。"
        "100円単位で入力してください。"
        "0円にすると購入予定から除外します。"
    )

    # レース・買い目構成ごとに編集状態を管理
    editor_key = (
        f"web14_summary_editor_{race_id}_"
        + "_".join(all_combos)
    )

    edited = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=[
            "買い目",
            "選択方法",
            "オッズ",
            "状態",
        ],
        column_config={
            "買い目": st.column_config.TextColumn(
                "3連単",
                width="small",
            ),
            "選択方法": st.column_config.TextColumn(
                "選択方法",
            ),
            "オッズ": st.column_config.NumberColumn(
                "オッズ（倍）",
                format="%.1f",
            ),
            "状態": st.column_config.TextColumn(
                "状態",
            ),
            "購入金額（円）":
                st.column_config.NumberColumn(
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

    amounts = pd.to_numeric(
        edited["購入金額（円）"],
        errors="coerce",
    )

    if (
        amounts.isna().any()
        or (amounts < 0).any()
        or (amounts > 100000).any()
        or (amounts % 100 != 0).any()
    ):
        st.error(
            "購入金額は0～100,000円の範囲で"
            "100円単位で入力してください。"
        )
        return

    edited["購入金額（円）"] = (
        amounts.astype(int)
    )

    # 編集した金額をレース別に保存
    for _, row in edited.iterrows():
        key = (race_id, row["買い目"])
        stake_map[key] = int(
            row["購入金額（円）"]
        )

    st.session_state[
        "web14_summary_stakes"
    ] = stake_map

    # 0円の買い目は購入予定から除外
    active = edited[
        edited["購入金額（円）"] > 0
    ].copy()

    total = int(
        active["購入金額（円）"].sum()
    )

    # 有効なオッズだけ仮払戻額を計算
    valid_odds = pd.to_numeric(
        active["オッズ"],
        errors="coerce",
    ).where(lambda x: x > 0)

    active["仮払戻額（円）"] = (
        valid_odds
        * active["購入金額（円）"]
    ).round()

    active["仮収支（円）"] = (
        active["仮払戻額（円）"] - total
    )

    valid_count = int(
        active["仮払戻額（円）"].notna().sum()
    )

    invalid_count = len(active) - valid_count

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "購入予定点数",
        f"{len(active)}点",
    )

    c2.metric(
        "購入予定金額",
        f"{total:,}円",
    )

    c3.metric(
        "有効オッズ",
        f"{valid_count}点",
    )

    c4.metric(
        "要確認・未取得",
        f"{invalid_count}点",
    )

    if active.empty:
        st.info(
            "購入金額が設定された買い目はありません。"
        )
        return

    if invalid_count:
        st.warning(
            f"{invalid_count}点は0.0倍または未取得です。"
            "仮払戻額は計算していません。"
        )

    st.markdown("#### 📋 購入予定の買い目一覧")

    st.dataframe(
        active,
        hide_index=True,
        use_container_width=True,
        column_config={
            "オッズ": st.column_config.NumberColumn(
                "オッズ",
                format="%.1f",
            ),
            "購入金額（円）":
                st.column_config.NumberColumn(
                    "購入金額（円）",
                    format="%d",
                ),
            "仮払戻額（円）":
                st.column_config.NumberColumn(
                    "仮払戻額（円）",
                    format="%.0f",
                ),
            "仮収支（円）":
                st.column_config.NumberColumn(
                    "仮収支（円）",
                    format="%.0f",
                ),
        },
    )

    known = active.dropna(
        subset=["仮払戻額（円）"]
    )

    if not known.empty:
        best = known.loc[
            known["仮払戻額（円）"].idxmax()
        ]

        st.info(
            "仮払戻額が最大の組み合わせ："
            f"{best['買い目']} "
            f"→ {best['仮払戻額（円）']:,.0f}円"
        )

    csv_data = active.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "購入予定一覧をCSVで保存",
        data=csv_data,
        file_name=(
            f"formation_{race_key[0]}_"
            f"{race_key[1]:02d}_"
            f"{race_key[2]:02d}.csv"
        ),
        mime="text/csv",
        key="web14_csv",
    )

    st.caption(
        "購入金額の入力・変更はこのサマリーのみで"
        "行います。買い目の重複は除外します。"
        "仮払戻額は取得時点のオッズによる試算です。"
        "実際の投票処理は行いません。"
    )

# ==========================================
# メイン画面
# ==========================================

def main():
    st.title("🚤 実戦用統合ダッシュボード")

    st.caption(
        "BOAT RACE AI | WEB-14 "
        "公式オッズ × フォーメーション × 資金管理"
    )

    day, venue, race, race_key = race_selector()

    st.markdown(
        f"### {day:%Y/%m/%d}｜"
        f"{VENUES[venue]} {race}R"
    )

    show_links(day, venue, race)

    auto_enabled = refresh_controls(race_key)

    race_data, race_error = fetch_selected_race(
        day, venue, race
    )

    if race_error:
        st.warning(race_error)

    closed_at = (
        race_data.get("closed_at")
        if race_data else None
    )

    st.markdown("### ⏱️ 締切・更新管理")

    web12.show_race_clock(
        closed_at=closed_at,
        race_key=race_key,
        auto_enabled=auto_enabled,
    )

    # ======================================
    # 出走表・展示は折りたたみ表示
    # ======================================

    with st.expander(
        "🚤 出走表・選手・モーター成績",
        expanded=False,
    ):
        if race_data:
            web12.show_entries(race_data)
        else:
            st.info("出走表データがありません。")

    with st.expander(
        "🌤️ 展示・ST・気象情報",
        expanded=False,
    ):
        if race_data:
            web12.show_preview(race_data)
        else:
            st.info("展示・気象情報がありません。")

    # ======================================
    # 公式オッズ
    # ======================================

    if (
        st.session_state.get("web12_odds_ready")
        != race_key
    ):
        if st.button(
            "📥 公式3連単オッズを取得",
            type="primary",
            use_container_width=True,
            key="web14_get_odds",
        ):
            st.session_state["web12_odds_ready"] = (
                race_key
            )
            st.rerun()

    odds, odds_error = fetch_selected_odds(
        race_key
    )

    if odds_error:
        st.warning(odds_error)

    st.caption(
        f"取得済みオッズ：{len(odds)}/120通り。"
        "公式サイトの掲載値であり、"
        "最終オッズとは異なる場合があります。"
    )

    # ======================================
    # メイン：左右2列
    # ======================================

    st.divider()

    left, right = st.columns(
        [1, 1.25],
        gap="large",
    )


    with left:
        selected_combos = get_selected_combos()

        show_odds_panel(
            odds,
            selected_combos,
            race_key,
        )

    with right:
        show_candidate_editor(
            odds,
            race_key,
        )

        st.divider()

        show_formation_preview(odds)

    # 左右の下に従来の組み合わせ指定を配置
    with st.expander("🎯 組み合わせを個別指定して確認"):
        web11.show_single_combo(odds)


    # ======================================
    # 下部：購入予定まとめ
    # ======================================

    st.divider()

    show_purchase_summary(
        odds,
        race_key,
    )

    st.divider()

    st.caption(
        "出走表・展示はBoatrace Open API、"
        "オッズはBOAT RACE公式ページを使用。"
        "AIの学習・推論・DB更新は行いません。"
        "締切・オッズ・購入可否は公式サイトで"
        "最終確認してください。"
    )


if __name__ == "__main__":
    main()
