
import csv
import sqlite3
from pathlib import Path
from collections import Counter

ROOT = Path(r"C:\boatrace_ai")
DB = ROOT / "data" / "boatrace.db"
CSV = ROOT / "outputs" / "phase83b_f3_preview.csv"

KEY_COLS = ("race_date", "venue_code", "race_no", "lane")

def key(row):
    return tuple(str(row[c]) for c in KEY_COLS)

with CSV.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

conn = sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)

def get_keys(table):
    sql = f"""
        SELECT race_date, venue_code, race_no, lane
        FROM {table}
        WHERE race_date BETWEEN '2026-10-23' AND '2026-10-29'
    """
    return {tuple(map(str, r)) for r in conn.execute(sql)}

csv_keys = {key(r) for r in rows}
v2_keys = get_keys("race_boat_results_v2")
racer_keys = get_keys("racer_features")
course_keys = get_keys("racer_course_features")

print("=== 1. ROW COVERAGE ===")
print("CSV rows:", len(rows))
print("CSV unique keys:", len(csv_keys))
print("V2 keys:", len(v2_keys))
print("Racer feature keys:", len(racer_keys))
print("Course feature keys:", len(course_keys))

for name, keys in [
    ("V2", v2_keys),
    ("RACER", racer_keys),
    ("COURSE", course_keys)
]:
    missing = sorted(keys - csv_keys)
    extra = sorted(csv_keys - keys)

    print(f"\n{name} vs F3")
    print("Missing from F3:", len(missing))
    print("Only in F3:", len(extra))
    print("Missing examples:", missing[:5])
    print("Extra examples:", extra[:5])

print("\n=== 2. FEATURE VALIDATION ===")

errors = Counter()

for row in rows:
    for kind in ("motor", "boat"):
        prev_n = None

        for days in (30, 90, 180):
            prefix = f"{kind}_{days}"

            n = int(row[f"{prefix}_starts"])
            win = row[f"{prefix}_win_rate"]
            top2 = row[f"{prefix}_top2_rate"]
            top3 = row[f"{prefix}_top3_rate"]
            smooth = float(row[f"{prefix}_top2_smoothed"])

            if n < 0:
                errors["negative_starts"] += 1

            if prev_n is not None and n < prev_n:
                errors["nonmonotonic_starts"] += 1

            prev_n = n

            if not 0 <= smooth <= 1:
                errors["invalid_smoothed_rate"] += 1

            if n == 0:
                if any(x != "" for x in (win, top2, top3)):
                    errors["zero_starts_with_rates"] += 1
                continue

            if any(x == "" for x in (win, top2, top3)):
                errors["missing_rates"] += 1
                continue

            w, t2, t3 = map(float, (win, top2, top3))

            if not (0 <= w <= t2 <= t3 <= 1):
                errors["invalid_rate_order"] += 1

            expected = (t2 * n + 20 / 3) / (n + 20)
            if abs(expected - smooth) > 1e-9:
                errors["invalid_smoothing"] += 1

print("Validation errors:", dict(errors))

print("\n=== 3. HISTORY COVERAGE ===")

for kind in ("motor", "boat"):
    for days in (30, 90, 180):
        name = f"{kind}_{days}_starts"
        counts = [int(r[name]) for r in rows]

        print(
            name,
            "min=", min(counts),
            "max=", max(counts),
            "avg=", round(sum(counts) / len(counts), 2),
            "zero=", sum(n == 0 for n in counts)
        )

print("\n=== 4. EQUIPMENT NUMBERS ===")

for kind in ("motor", "boat"):
    col = f"{kind}_no"
    values = [int(r[col]) for r in rows if r[col] != ""]

    print(
        col,
        "min=", min(values),
        "max=", max(values),
        "over100=", sum(v > 100 for v in values)
    )

print("\n=== 5. DAILY ROW COUNTS ===")

daily = Counter(r["race_date"] for r in rows)

for day, count in sorted(daily.items()):
    print(day, count)

print("\n=== 6. INVALID RESULTS IN TARGET PERIOD ===")

result = conn.execute("""
    SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN finish_position BETWEEN 1 AND 6
            THEN 1 ELSE 0 END) AS valid,
        SUM(CASE WHEN finish_position IS NULL
                  OR finish_position NOT BETWEEN 1 AND 6
            THEN 1 ELSE 0 END) AS invalid
    FROM race_boat_results_v2
    WHERE race_date BETWEEN '2026-10-23' AND '2026-10-29'
""").fetchone()

print("Total / valid / invalid:", result)

conn.close()

print("\n=== PHASE 8.3-B VALIDATION COMPLETE ===")
