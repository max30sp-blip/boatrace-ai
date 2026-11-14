
from pathlib import Path

import streamlit as st


# ====================================================
# BOAT RACE AI
# WEB-7 : Main Navigation
# ====================================================

ROOT = Path(__file__).resolve().parent
PAGES = ROOT / "pages"


def page(filename, title, icon, default=False):
    path = PAGES / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"ページが見つかりません: {path}"
        )

    return st.Page(
        str(path),
        title=title,
        icon=icon,
        default=default,
    )


def main():

    home = page(
        "6_統合AIダッシュボード.py",
        "総合ダッシュボード",
        "🏠",
        default=True,
    )

    entries = page(
        "2_出走表と選手比較.py",
        "出走表・選手比較",
        "🚤",
    )

    prediction = page(
        "3_出走表とAI予測.py",
        "3連単AI予測",
        "🎯",
    )

    details = page(
        "4_選手モーター詳細分析.py",
        "選手・モーター詳細",
        "📊",
    )

    previous = page(
        "5_新AI分析ダッシュボード.py",
        "旧ダッシュボード（WEB-5）",
        "🧪",
    )
    
    upcoming = page(
        "9_当日未来レース出走表.py",
        "当日・未来レース",
        "📅",
    )
    
    odds_page = page(
        "11_3連単オッズとフォーメーション.py",
        "3連単オッズ・フォーメーション",
        "🎫",
    )
    
    integrated = page(
    "12_当日レース統合分析.py",
    "当日レース統合分析",
    "🚤",
)
    
    web14 = page(
        "14_実戦用統合ダッシュボード.py",
        "実戦用統合ダッシュボード",
        "🎯",
    )
    
    # ページのグループ分け

    navigation = {
        "メイン": [
            home,
            web14,
            integrated,
            upcoming,
            odds_page,
        ],
        "分析ツール": [
            entries,
            prediction,
            details,
        ],
        "開発・比較用": [
            previous,
        ],
    }

    # サイドバーの共通ナビゲーション
    selected_page = st.navigation(
        navigation,
        position="sidebar",
        expanded=True,
    )

    # 選択したページを実行
    selected_page.run()


if __name__ == "__main__":
    main()
