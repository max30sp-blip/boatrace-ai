
import sqlite3
from pathlib import Path

DB_PATH = Path(r"C:\boatrace_ai\data\boatrace.db")

conn = sqlite3.connect(
    DB_PATH.as_uri() + "?mode=ro",
    uri=True
)

conn.execute("PRAGMA query_only=ON")

def run(title, sql):
    print(f"\n=== {title} ===")
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]

        for row in cur.fetchall():
            print(dict(zip(cols, row)))

    except sqlite3.Error as e:
        print("ERROR:", e)


# 1. 全テーブルを確認
run("ALL TABLES", """
SELECT name
FROM sqlite_master
WHERE type = 'table'
ORDER BY name
""")


# 2. 結果データの収録期間
for table in [
    "race_boat_results",
    "race_boat_results_v2",
    "race_boat_results_supplement"
]:
    run(f"DATA COVERAGE: {table}", f"""
    SELECT
        COUNT(*) AS total_rows,
        MIN(race_date) AS first_date,
        MAX(race_date) AS last_date,
        COUNT(DISTINCT race_date) AS unique_dates,
        COUNT(DISTINCT racer_id) AS unique_racers
    FROM {table}
    """)


# 3. 年ごとのv2履歴件数
run("V2 YEARLY COVERAGE", """
SELECT
    SUBSTR(race_date, 1, 4) AS year,
    COUNT(*) AS rows,
    SUM(
        CASE WHEN motor_no IS NOT NULL
        THEN 1 ELSE 0 END
    ) AS motor_available,
    SUM(
        CASE WHEN boat_no IS NOT NULL
        THEN 1 ELSE 0 END
    ) AS boat_available
FROM race_boat_results_v2
GROUP BY SUBSTR(race_date, 1, 4)
ORDER BY year
""")


# 4. モーター・ボートの利用可能性
for table in [
    "race_boat_results",
    "race_boat_results_v2"
]:
    run(f"MOTOR / BOAT QUALITY: {table}", f"""
    SELECT
        COUNT(*) AS total_rows,
        SUM(CASE
            WHEN motor_no BETWEEN 1 AND 100
            THEN 1 ELSE 0 END
        ) AS valid_motor_rows,
        SUM(CASE
            WHEN boat_no BETWEEN 1 AND 100
            THEN 1 ELSE 0 END
        ) AS valid_boat_rows,
        SUM(CASE
            WHEN finish_position BETWEEN 1 AND 6
            THEN 1 ELSE 0 END
        ) AS valid_finish_rows,
        COUNT(DISTINCT
            venue_code || '-' || motor_no
        ) AS venue_motor_pairs,
        COUNT(DISTINCT
            venue_code || '-' || boat_no
        ) AS venue_boat_pairs
    FROM {table}
    """)


# 5. 既存選手特徴量の有効行数
run("RACER FEATURE QUALITY", """
SELECT
    COUNT(*) AS total_rows,
    SUM(CASE
        WHEN starts_30 > 0
        THEN 1 ELSE 0 END
    ) AS valid_30,
    SUM(CASE
        WHEN starts_90 > 0
        THEN 1 ELSE 0 END
    ) AS valid_90,
    SUM(CASE
        WHEN starts_180 > 0
        THEN 1 ELSE 0 END
    ) AS valid_180,
    SUM(CASE
        WHEN win_rate_180 IS NOT NULL
        THEN 1 ELSE 0 END
    ) AS valid_win_rate,
    MIN(race_date) AS first_date,
    MAX(race_date) AS last_date
FROM racer_features
""")


# 6. コース特徴量の有効行数
run("COURSE FEATURE QUALITY", """
SELECT
    COUNT(*) AS total_rows,
    SUM(CASE
        WHEN lane_starts_365 > 0
        THEN 1 ELSE 0 END
    ) AS valid_lane,
    SUM(CASE
        WHEN venue_starts_365 > 0
        THEN 1 ELSE 0 END
    ) AS valid_venue,
    SUM(CASE
        WHEN actual_course_starts_365 > 0
        THEN 1 ELSE 0 END
    ) AS valid_actual_course,
    MIN(race_date) AS first_date,
    MAX(race_date) AS last_date
FROM racer_course_features
""")


# 7. 既存の結果データに重複があるか
run("V2 DUPLICATE KEYS", """
SELECT
    COUNT(*) AS duplicate_groups,
    COALESCE(SUM(cnt - 1), 0) AS extra_rows
FROM (
    SELECT COUNT(*) AS cnt
    FROM race_boat_results_v2
    GROUP BY
        race_date,
        venue_code,
        race_no,
        lane
    HAVING COUNT(*) > 1
)
""")


# 8. 特徴量の生成対象の日付を確認
run("FEATURE DATE COVERAGE", """
SELECT
    'racer_features' AS source,
    MIN(race_date) AS first_date,
    MAX(race_date) AS last_date,
    COUNT(DISTINCT race_date) AS dates
FROM racer_features

UNION ALL

SELECT
    'racer_course_features',
    MIN(race_date),
    MAX(race_date),
    COUNT(DISTINCT race_date)
FROM racer_course_features
""")


conn.close()

print("\n=== PHASE 8.3-B AUDIT COMPLETE ===")
