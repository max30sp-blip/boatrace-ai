
import csv
import sqlite3
from pathlib import Path
from collections import Counter

ROOT = Path(r"C:\boatrace_ai")
DB = ROOT / "data" / "boatrace.db"
INPUT = ROOT / "outputs" / "phase83b_f3_preview.csv"
OUTPUT = ROOT / "outputs" / "phase83b_f3_ability_preview.csv"

KEYS = ("race_date", "venue_code", "race_no", "lane")

RACER_COLS = [
    "starts_30", "win_rate_30", "top3_rate_30", "avg_st_30",
    "starts_90", "win_rate_90", "top3_rate_90", "avg_st_90",
    "starts_180", "win_rate_180", "top3_rate_180",
    "avg_st_180", "same_lane_starts_180",
    "same_lane_win_rate_180", "same_venue_starts_180",
    "same_venue_win_rate_180"
]

COURSE_COLS = [
    "lane_starts_365", "lane_win_rate_365",
    "lane_top3_rate_365", "venue_starts_365",
    "venue_win_rate_365", "venue_lane_starts_365",
    "venue_lane_win_rate_365", "smoothed_lane_win_rate",
    "actual_course_starts_365",
    "actual_course_win_rate_365", "course_change_rate_365"
]

def make_key(row):
    return tuple(str(row[c]) for c in KEYS)

def as_float(value):
    if value is None or value == "":
        return None
    return float(value)

def difference(a, b):
    x, y = as_float(a), as_float(b)
    return "" if x is None or y is None else x - y

def load_features(conn, table, columns):
    query = f"""
        SELECT {", ".join(KEYS)}, racer_id,
               {", ".join(columns)}
        FROM {table}
        WHERE race_date BETWEEN '2026-10-23'
                            AND '2026-10-29'
    """

    cursor = conn.execute(query)
    names = [d[0] for d in cursor.description]
    result = {}

    for values in cursor:
        row = dict(zip(names, values))
        key = make_key(row)

        if key in result:
            raise ValueError(f"Duplicate key: {table} {key}")

        result[key] = row

    return result

conn = sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)
conn.execute("PRAGMA query_only=ON")

racer = load_features(conn, "racer_features", RACER_COLS)
course = load_features(conn, "racer_course_features", COURSE_COLS)

with INPUT.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    f3 = list(reader)
    original_columns = reader.fieldnames

if not original_columns:
    raise ValueError("Empty F3 CSV")

f3_keys = {make_key(r) for r in f3}

print("=== 1. MISSING 54 ROWS ===")
missing = sorted(set(racer) - f3_keys)

race_groups = sorted({
    (d, v, n) for d, v, n, lane in missing
})

print("Missing rows:", len(missing))
print("Missing races:", len(race_groups))

for race in race_groups:
    print("Missing race:", race)

print("\n=== 2. JOIN FEATURE TABLES ===")

extra_columns = (
    ["racer_" + c for c in RACER_COLS]
    + ["course_" + c for c in COURSE_COLS]
    + [
        "motor_vs_racer_win_180",
        "motor_vs_racer_top3_180",
        "boat_vs_racer_top3_180",
        "motor_vs_lane_win_365",
        "motor_vs_venue_win_365",
        "motor_vs_boat_top2_180"
    ]
)

output_columns = original_columns + extra_columns

stats = Counter()

with OUTPUT.open(
    "w", encoding="utf-8-sig", newline=""
) as f:

    writer = csv.DictWriter(f, fieldnames=output_columns)
    writer.writeheader()

    for row in f3:
        key = make_key(row)
        r = racer.get(key)
        c = course.get(key)

        if r is None:
            stats["missing_racer"] += 1
        elif str(r["racer_id"]) != row["racer_id"]:
            stats["racer_id_mismatch"] += 1
            r = None
        else:
            stats["matched_racer"] += 1

        if c is None:
            stats["missing_course"] += 1
        elif str(c["racer_id"]) != row["racer_id"]:
            stats["course_id_mismatch"] += 1
            c = None
        else:
            stats["matched_course"] += 1

        for col in RACER_COLS:
            row["racer_" + col] = (
                r[col] if r is not None and r[col] is not None
                else ""
            )

        for col in COURSE_COLS:
            row["course_" + col] = (
                c[col] if c is not None and c[col] is not None
                else ""
            )

        row["motor_vs_racer_win_180"] = difference(
            row["motor_180_win_rate"],
            row["racer_win_rate_180"]
        )

        row["motor_vs_racer_top3_180"] = difference(
            row["motor_180_top3_rate"],
            row["racer_top3_rate_180"]
        )

        row["boat_vs_racer_top3_180"] = difference(
            row["boat_180_top3_rate"],
            row["racer_top3_rate_180"]
        )

        row["motor_vs_lane_win_365"] = difference(
            row["motor_180_win_rate"],
            row["course_lane_win_rate_365"]
        )

        row["motor_vs_venue_win_365"] = difference(
            row["motor_180_win_rate"],
            row["course_venue_win_rate_365"]
        )

        row["motor_vs_boat_top2_180"] = difference(
            row["motor_180_top2_smoothed"],
            row["boat_180_top2_smoothed"]
        )

        writer.writerow(row)
        stats["output_rows"] += 1

conn.close()

print("\n=== 3. MERGE RESULTS ===")
for name, value in sorted(stats.items()):
    print(f"{name}: {value}")

print("Output columns:", len(output_columns))
print("Output:", OUTPUT)

print("\n=== PHASE 8.3-B MERGE COMPLETE ===")
