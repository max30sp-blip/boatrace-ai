
import csv
import math
import sqlite3
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict, Counter

ROOT = Path(r"C:\boatrace_ai")
DB = ROOT / "data/boatrace.db"
TRAIN_DB = ROOT / "data/train_dataset_v1.db"

OUT_DIR = ROOT / "outputs/phase86c"
TARGET = "2026-10-29"
WINDOWS = (30, 90, 180)
FIELDS = ("starts", "win_rate", "top3_rate", "avg_st")


def readonly(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    con = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True
    )
    con.execute("PRAGMA query_only=ON")
    return con


def num(value):
    if value is None or value == "":
        return None
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def normalize_finish(value):
    if value is None:
        return None
    try:
        n = int(value)
        return n if 1 <= n <= 6 else None
    except (ValueError, TypeError):
        return None


def stats(history, target_day, window):
    # 元の summarize() と同じ期間境界
    recent = [
        (finish, st)
        for day, finish, st in history
        if 0 < (target_day - day).days <= window
    ]

    n = len(recent)

    if n == 0:
        return {
            "starts": 0,
            "win_rate": None,
            "top3_rate": None,
            "avg_st": None,
        }

    wins = sum(
        finish == 1 for finish, _ in recent
    )

    top3 = sum(
        finish is not None and 1 <= finish <= 3
        for finish, _ in recent
    )

    # 元の実装では ST の値域による除外をしない
    st_values = [
        st for _, st in recent
        if st is not None
    ]

    return {
        "starts": n,
        "win_rate": wins / n,
        "top3_rate": top3 / n,
        "avg_st": (
            sum(st_values) / len(st_values)
            if st_values else None
        )
    }


def equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= 1e-6


def save_csv(path, rows, columns):
    if path.exists():
        raise FileExistsError(
            f"Output already exists: {path}"
        )

    with path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    target_day = date.fromisoformat(TARGET)
    earliest = (
        target_day - timedelta(days=180)
    ).isoformat()

    con = readonly(DB)
    train = readonly(TRAIN_DB)

    programs = con.execute("""
        SELECT venue_code, race_no, lane,
               racer_id, grade
        FROM program_entries
        WHERE race_date = ?
        ORDER BY venue_code, race_no, lane
    """, (TARGET,)).fetchall()

    if not programs:
        raise ValueError("No program entries")

    racers = {
        int(r[3]) for r in programs if r[3] is not None
    }

    histories = defaultdict(list)

    for d, rid, finish, st in con.execute("""
        SELECT race_date, racer_id,
               finish_position, start_timing
        FROM race_boat_results_v2
        WHERE race_date >= ?
          AND race_date < ?
        ORDER BY race_date
    """, (earliest, TARGET)):

        if rid is None or int(rid) not in racers:
            continue

        histories[int(rid)].append((
            date.fromisoformat(d),
            normalize_finish(finish),
            num(st),
        ))

    cur = train.execute("""
        SELECT *
        FROM train_races
        WHERE race_date = ?
    """, (TARGET,))

    columns = [d[0] for d in cur.description]

    references = {}

    for values in cur:
        item = dict(zip(columns, values))
        key = (
            int(item["venue_code"]),
            int(item["race_no"])
        )
        if key in references:
            raise ValueError(f"Duplicate race: {key}")
        references[key] = item

    con.close()
    train.close()

    counts = defaultdict(Counter)
    differences = []
    features = []
    racers_missing = defaultdict(
        lambda: {"appearances": 0, "max_gap": 0}
    )

    for venue, race, lane, rid, grade in programs:
        venue = int(venue)
        race = int(race)
        lane = int(lane)
        rid = int(rid)

        ref = references.get((venue, race))

        if ref is None:
            raise ValueError(
                f"Missing reference: {(venue, race)}"
            )

        record = {
            "race_date": TARGET,
            "venue_code": venue,
            "race_no": race,
            "lane": lane,
            "racer_id": rid,
        }

        for window in WINDOWS:
            computed = stats(
                histories[rid], target_day, window
            )

            for field in FIELDS:
                name = (
                    f"b{lane}_f0_{field}_{window}d"
                )

                actual = computed[field]
                expected = num(ref.get(name))

                record[name] = actual

                if equal(actual, expected):
                    counts[(window, field)]["matched"] += 1
                else:
                    counts[(window, field)]["mismatch"] += 1

                    differences.append({
                        "venue_code": venue,
                        "race_no": race,
                        "lane": lane,
                        "racer_id": rid,
                        "window": window,
                        "field": field,
                        "computed": actual,
                        "reference": expected,
                        "difference": (
                            actual - expected
                            if actual is not None
                            and expected is not None
                            else ""
                        )
                    })

                    if field == "starts":
                        gap = int(expected - actual)
                        item = racers_missing[rid]
                        item["appearances"] += 1
                        item["max_gap"] = max(
                            item["max_gap"], gap
                        )

        features.append(record)

    OUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    feature_columns = [
        "race_date", "venue_code",
        "race_no", "lane", "racer_id"
    ] + [
        f"b{lane}_f0_{field}_{window}d"
        for lane in range(1, 7)
        for window in WINDOWS
        for field in FIELDS
    ]

    # 1行は1艇なので、F0はその艇の12項目だけを保持
    # 他艇の列はCSV上では空欄になる
    save_csv(
        OUT_DIR / f"f0_candidate_{TARGET}.csv",
        features,
        feature_columns
    )

    difference_columns = [
        "venue_code", "race_no", "lane",
        "racer_id", "window", "field",
        "computed", "reference", "difference"
    ]

    save_csv(
        OUT_DIR / f"f0_remaining_differences_{TARGET}.csv",
        differences,
        difference_columns
    )

    missing_rows = [
        {
            "racer_id": rid,
            "appearances": item["appearances"],
            "max_gap": item["max_gap"]
        }
        for rid, item in sorted(
            racers_missing.items(),
            key=lambda kv: (
                -kv[1]["max_gap"],
                -kv[1]["appearances"]
            )
        )
    ]

    save_csv(
        OUT_DIR / f"f0_missing_racers_{TARGET}.csv",
        missing_rows,
        ["racer_id", "appearances", "max_gap"]
    )

    print("=== PHASE 8.6-C FINAL F0 CHECK ===")
    print("Program boats:", len(programs))
    print("Racers with history:", len(histories))

    total_match = 0
    total_check = 0

    for window in WINDOWS:
        print(f"\n=== {window} DAYS ===")

        for field in FIELDS:
            item = counts[(window, field)]
            ok = item["matched"]
            ng = item["mismatch"]
            total = ok + ng

            total_match += ok
            total_check += total

            print(
                f"{field:12s}",
                f"matched={ok:3d}",
                f"mismatch={ng:3d}",
                f"rate={100*ok/total:.2f}%"
            )

    print("\n=== SUMMARY ===")
    print("Matching values:", total_match)
    print("Compared values:", total_check)
    print(
        "Overall match rate:",
        f"{100*total_match/total_check:.2f}%"
    )
    print(
        "Racers with start count differences:",
        len(racers_missing)
    )

    print("\n=== PHASE 8.6-C FINAL CHECK COMPLETE ===")
    print("Output directory:", OUT_DIR)


if __name__ == "__main__":
    main()
