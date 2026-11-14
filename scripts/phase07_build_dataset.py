
import csv
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "boatrace.db"
OUTPUT = ROOT / "data" / "training"
OUTPUT.mkdir(parents=True, exist_ok=True)

BASE_FEATURES = [
    "age", "weight_kg", "national_win_rate",
    "national_2rate", "local_win_rate",
    "local_2rate", "motor_2rate", "boat_2rate"
]

HISTORY_FEATURES = [
    "starts_30", "win_rate_30", "top3_rate_30",
    "avg_st_30", "starts_90", "win_rate_90",
    "top3_rate_90", "avg_st_90", "starts_180",
    "win_rate_180", "top3_rate_180", "avg_st_180",
    "same_lane_starts_180", "same_lane_win_rate_180",
    "same_venue_starts_180", "same_venue_win_rate_180"
]

COURSE_FEATURES = [
    "lane_starts_365", "lane_win_rate_365",
    "lane_top3_rate_365", "venue_starts_365",
    "venue_win_rate_365", "venue_lane_starts_365",
    "venue_lane_win_rate_365",
    "smoothed_lane_win_rate",
    "actual_course_starts_365",
    "actual_course_win_rate_365",
    "course_change_rate_365"
]

FEATURES = (
    ["grade"] + BASE_FEATURES
    + HISTORY_FEATURES + COURSE_FEATURES
)

GRADES = {"B2": 0, "B1": 1, "A2": 2, "A1": 3}


def get_columns(conn, table):
    return {
        row[1] for row in
        conn.execute(f"PRAGMA table_info({table})")
    }


def validate_schema(conn):
    tables = [
        ("program_entries", ["grade"] + BASE_FEATURES),
        ("racer_features", HISTORY_FEATURES),
        ("racer_course_features", COURSE_FEATURES),
        ("trifecta_results", ["combination"]),
        ("race_boat_results_v2", ["finish_position"])
    ]

    for table, cols in tables:
        actual = get_columns(conn, table)
        missing = set(cols) - actual

        if missing:
            raise RuntimeError(
                f"{table} missing columns: {sorted(missing)}"
            )


def load_data(conn):
    columns = (
        [f"p.{x}" for x in ["grade"] + BASE_FEATURES]
        + [f"h.{x}" for x in HISTORY_FEATURES]
        + [f"c.{x}" for x in COURSE_FEATURES]
    )

    sql = f"""
        SELECT
            p.race_date, p.venue_code, p.race_no,
            p.lane, p.racer_id,
            {", ".join(columns)}
        FROM program_entries p
        LEFT JOIN racer_features h
          ON p.race_date=h.race_date
         AND p.venue_code=h.venue_code
         AND p.race_no=h.race_no
         AND p.lane=h.lane
        LEFT JOIN racer_course_features c
          ON p.race_date=c.race_date
         AND p.venue_code=c.venue_code
         AND p.race_no=c.race_no
         AND p.lane=c.lane
        ORDER BY
            p.race_date, p.venue_code, p.race_no, p.lane
    """

    entries = defaultdict(dict)

    for row in conn.execute(sql):
        key = tuple(row[:3])
        lane = row[3]
        values = dict(zip(FEATURES, row[5:]))

        values["grade"] = GRADES.get(values["grade"])

        entries[key][lane] = values

    return entries


def main():
    if not DB.exists():
        raise FileNotFoundError(DB)

    with sqlite3.connect(DB) as conn:
        validate_schema(conn)
        entries = load_data(conn)

        payouts = {
            tuple(row[:3]): row[3]
            for row in conn.execute("""
                SELECT race_date, venue_code, race_no,
                       combination
                FROM trifecta_results
            """)
        }

        actual = defaultdict(list)

        for day, venue, race, lane, finish in conn.execute("""
            SELECT race_date, venue_code, race_no,
                   lane, finish_position
            FROM race_boat_results_v2
        """):
            if finish in (1, 2, 3):
                actual[(day, venue, race)].append(
                    (finish, lane)
                )

    header = ["race_date", "venue_code", "race_no"]

    for lane in range(1, 7):
        header.extend(
            f"lane{lane}_{feature}" for feature in FEATURES
        )

    header.append("target_combo")

    records = []
    skipped = Counter()

    for key, lanes in sorted(entries.items()):
        if set(lanes) != set(range(1, 7)):
            skipped["incomplete_entries"] += 1
            continue

        if key not in payouts:
            skipped["no_trifecta"] += 1
            continue

        ranks = sorted(actual.get(key, []))

        if len(ranks) != 3:
            skipped["incomplete_top3"] += 1
            continue

        combo = "-".join(str(lane) for _, lane in ranks)

        if combo != payouts[key]:
            skipped["target_mismatch"] += 1
            continue

        if len(set(lane for _, lane in ranks)) != 3:
            skipped["duplicate_target"] += 1
            continue

        row = list(key)

        for lane in range(1, 7):
            row.extend(lanes[lane].get(f) for f in FEATURES)

        row.append(combo)
        records.append(row)

    output_path = OUTPUT / "trifecta_dataset.csv"

    with output_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(records)

    print("\n========== PHASE 7.0 SUMMARY ==========")
    print("Dataset:", output_path)
    print("Race rows:", f"{len(records):,}")
    print("Feature columns:", 6 * len(FEATURES) + 1)
    print("Skipped:", dict(skipped))

    if records:
        print("First date:", records[0][0])
        print("Last date:", records[-1][0])

        combos = Counter(row[-1] for row in records)
        print("Unique trifecta labels:", len(combos))
        print("Most common:", combos.most_common(5))

    print("Status: dataset creation completed")


if __name__ == "__main__":
    main()
