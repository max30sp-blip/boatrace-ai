
import json
import sqlite3
from pathlib import Path

ROOT = Path(r"C:\boatrace_ai")

TRAIN_DB = ROOT / "data/train_dataset_v1.db"
RESULT_DB = ROOT / "data/boatrace.db"
REPORT = ROOT / "models/phase08_1_b_report.json"

with REPORT.open(encoding="utf-8") as f:
    report = json.load(f)

print("=== PHASE 8.3-D DATASET ALIGNMENT ===")
print("Train period:", report["train_period"])
print("Validation period:", report["valid_period"])
print("Train rows:", report["train_rows"])
print("Validation rows:", report["valid_rows"])
print("Features:", report["features"])
print("Best iteration:", report["best_iteration"])

conn = sqlite3.connect(
    TRAIN_DB.as_uri() + "?mode=ro",
    uri=True
)
conn.execute("PRAGMA query_only=ON")

# 別DBは一時的な読み取り専用接続として参照
result_uri = RESULT_DB.as_uri() + "?mode=ro"
conn.execute("ATTACH DATABASE ? AS results", (result_uri,))

print("\n=== 1. YEARLY COVERAGE ===")

rows = conn.execute("""
    SELECT
        SUBSTR(t.race_date, 1, 4) AS year,
        COUNT(*) AS train_races,
        SUM(
            CASE WHEN EXISTS (
                SELECT 1
                FROM results.race_boat_results_v2 r
                WHERE r.race_date = t.race_date
                  AND r.venue_code = t.venue_code
                  AND r.race_no = t.race_no
            )
            THEN 1 ELSE 0 END
        ) AS matching_races
    FROM train_races t
    GROUP BY SUBSTR(t.race_date, 1, 4)
    ORDER BY year
""").fetchall()

for year, total, matched in rows:
    print(
        year,
        "train=", total,
        "matched=", matched,
        "missing=", total - matched,
        "coverage=", f"{matched/total:.2%}"
    )

print("\n=== 2. VALIDATION DATA ===")

valid_start, valid_end = report["valid_period"]

valid_count = conn.execute("""
    SELECT COUNT(*)
    FROM train_races
    WHERE race_date BETWEEN ? AND ?
""", (valid_start, valid_end)).fetchone()[0]

print("Validation DB rows:", valid_count)
print("Baseline selected rows:", report["valid_rows"])

print("\n=== 3. LABEL CHECK ===")

for label, count in conn.execute("""
    SELECT label, COUNT(*)
    FROM train_races
    GROUP BY label
    ORDER BY COUNT(*) DESC
    LIMIT 10
"""):
    print(label, count)

print("\n=== 4. DUPLICATE RACES ===")

duplicate = conn.execute("""
    SELECT COUNT(*)
    FROM (
        SELECT race_date, venue_code, race_no
        FROM train_races
        GROUP BY race_date, venue_code, race_no
        HAVING COUNT(*) > 1
    )
""").fetchone()[0]

print("Duplicate race keys:", duplicate)

conn.close()

print("\n=== PHASE 8.3-D ALIGNMENT COMPLETE ===")
