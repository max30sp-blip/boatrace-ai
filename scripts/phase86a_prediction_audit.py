
import json
import re
import sqlite3
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"C:\boatrace_ai")

MODEL_REPORT = (
    ROOT / "models/phase08_1_b_report.json"
)
DB_PATH = ROOT / "data/boatrace.db"

if not MODEL_REPORT.exists():
    raise FileNotFoundError(MODEL_REPORT)

with MODEL_REPORT.open(encoding="utf-8-sig") as f:
    report = json.load(f)

features = report["feature_names"]

if len(features) != 618:
    raise ValueError("Unexpected feature count")

groups = defaultdict(list)

for feature in features:
    match = re.match(r"^b[1-6]_(p|f[0-9]+)_", feature)

    if not match:
        groups["other"].append(feature)
    else:
        groups[match.group(1)].append(feature)

descriptions = {
    "p": "出走表・選手基本情報",
    "f0": "過去30/90/180日成績",
    "f1": "コース・競艇場別成績",
    "f2": "枠番・コース適性",
    "f3": "モーター・ボート成績",
    "f4": "競艇場・コース傾向",
    "f5": "選手の直近の調子",
    "f6": "今節成績・スタート情報",
    "other": "その他",
}

print("=== PHASE 8.6-A PREDICTION AUDIT ===")
print("Model features:", len(features))

print("\n=== FEATURE GROUPS ===")

for group, columns in sorted(groups.items()):
    print(
        f"{group:6s}",
        f"columns={len(columns):3d}",
        descriptions.get(group, "")
    )

print("\n=== DATABASE TABLES ===")

if not DB_PATH.exists():
    raise FileNotFoundError(DB_PATH)

con = sqlite3.connect(
    DB_PATH.as_uri() + "?mode=ro",
    uri=True
)
con.execute("PRAGMA query_only=ON")

tables = con.execute("""
    SELECT name FROM sqlite_master
    WHERE type='table'
    ORDER BY name
""").fetchall()

for (table,) in tables:
    name = table.lower()

    if any(word in name for word in (
        "program",
        "entry",
        "racer",
        "motor",
        "feature",
        "result",
        "weather"
    )):
        safe = table.replace('"', '""')

        cols = [
            item[1]
            for item in con.execute(
                f'PRAGMA table_info("{safe}")'
            )
        ]

        print(f"\n{table}: {len(cols)} columns")
        print("  ", ", ".join(cols[:25]))

con.close()

print("\n=== PRODUCTION REQUIREMENTS ===")
print("1. 当日出走表の取得")
print("2. 選手・モーター番号の照合")
print("3. 予測時点以前の履歴から618特徴量を生成")
print("4. 学習時と同じ列順に整列")
print("5. 欠損・未公開情報の検証")
print("6. 3連単120通りの確率を出力")

print("\n=== PHASE 8.6-A AUDIT COMPLETE ===")
