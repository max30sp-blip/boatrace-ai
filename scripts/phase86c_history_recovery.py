
import csv
import math
import sqlite3
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

ROOT = Path(r"C:\boatrace_ai")
DB = ROOT / "data/boatrace.db"
TRAIN_DB = ROOT / "data/train_dataset_v1.db"
OUT = ROOT / "outputs/phase86c/history_recovery_report.csv"

TARGET = "2026-10-29"
WINDOWS = (30, 90, 180)
FIELDS = ("starts", "win_rate", "top3_rate", "avg_st")

SOURCES = (
    "race_boat_results_v2",
    "race_boat_results",
    "race_boat_results_supplement",
)


def readonly(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    con = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True
    )
    con.execute("PRAGMA query_only=ON")
    return con


def number(value):
    if value is None or str(value).strip() == "":
        return None

    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def finish_number(value):
    n = number(value)
    if n is None:
        return None

    if n.is_integer() and 1 <= n <= 6:
        return int(n)

    return None


def calculate(history, target_day, window):
    recent = [
        (finish, st)
        for day, finish, st in history
        if 0 < (target_day - day).days <= window
    ]

    n = len(recent)

    if n == 0:
        return (0, None, None, None)

    wins = sum(f == 1 for f, _ in recent)
    top3 = sum(
        f is not None and f <= 3
        for f, _ in recent
    )

    starts = [
        st for _, st in recent
        if st is not None
    ]

    return (
        n,
        wins / n,
        top3 / n,
        sum(starts) / len(starts) if starts else None,
    )


def equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= 1e-6


def load_source(con, table, racers, earliest):
    exists = con.execute("""
        SELECT 1 FROM sqlite_master
        WHERE type='table' AND name=?
    """, (table,)).fetchone()

    if not exists:
        return {}

    rows = con.execute(f"""
        SELECT
            race_date, venue_code, race_no,
            racer_id, finish_position, start_timing
        FROM "{table}"
        WHERE race_date >= ?
          AND race_date < ?
    """, (earliest, TARGET))

    found = {}

    for d, venue, race, racer, finish, st in rows:
        if None in (d, venue, race, racer):
            continue

        racer = int(racer)

        if racer not in racers:
            continue

        key = (d, int(venue), int(race), racer)

        if key in found:
            raise ValueError(
                f"Duplicate source key: {table} {key}"
            )

        found[key] = (
            date.fromisoformat(d),
            finish_number(finish),
            number(st),
        )

    return found


def main():
    target_day = date.fromisoformat(TARGET)
    earliest = (
        target_day - timedelta(days=180)
    ).isoformat()

    con = readonly(DB)
    train = readonly(TRAIN_DB)

    programs = con.execute("""
        SELECT venue_code, race_no, lane, racer_id
        FROM program_entries
        WHERE race_date = ?
        ORDER BY venue_code, race_no, lane
    """, (TARGET,)).fetchall()

    if not programs:
        raise ValueError("No programs")

    racers = {
        int(row[3]) for row in programs
        if row[3] is not None
    }

    cursor = train.execute("""
        SELECT * FROM train_races
        WHERE race_date = ?
    """, (TARGET,))

    names = [d[0] for d in cursor.description]

    reference = {}

    for values in cursor:
        row = dict(zip(names, values))
        key = (
            int(row["venue_code"]),
            int(row["race_no"])
        )
        reference[key] = row

    sources = {
        name: load_source(con, name, racers, earliest)
        for name in SOURCES
    }

    con.close()
    train.close()

    print("=== PHASE 8.6-C HISTORY RECOVERY ===")

    for name in SOURCES:
        print(name, "records:", len(sources[name]))

    base = sources["race_boat_results_v2"]
    original = sources["race_boat_results"]
    supplement = sources["race_boat_results_supplement"]

    scenarios = {
        "v2_only": dict(base),
        "v2_plus_results": {**original, **base},
        "v2_plus_supplement": {**supplement, **base},
        "all_sources": {
            **supplement, **original, **base
        }
    }

    report = []

    for name, records in scenarios.items():
        added = len(records) - len(base)

        histories = defaultdict(list)

        for key, data in records.items():
            histories[key[3]].append(data)

        matches = defaultdict(int)
        totals = defaultdict(int)

        for venue, race, lane, racer in programs:
            if racer is None:
                continue

            ref = reference.get(
                (int(venue), int(race))
            )

            if ref is None:
                continue

            history = histories[int(racer)]

            for window in WINDOWS:
                values = calculate(
                    history, target_day, window
                )

                for field, actual in zip(FIELDS, values):
                    col = (
                        f"b{int(lane)}_f0_{field}_{window}d"
                    )

                    expected = number(ref.get(col))
                    key = (window, field)

                    totals[key] += 1

                    if equal(actual, expected):
                        matches[key] += 1

        total_ok = sum(matches.values())
        total_n = sum(totals.values())

        print(f"\n=== {name.upper()} ===")
        print("New unique history records:", added)

        for window in WINDOWS:
            ok = sum(
                matches[(window, field)]
                for field in FIELDS
            )

            n = sum(
                totals[(window, field)]
                for field in FIELDS
            )

            print(
                f"{window}d:",
                f"{ok}/{n}",
                f"{100 * ok / n:.2f}%"
            )

        print(
            "Overall:",
            f"{100 * total_ok / total_n:.2f}%"
        )

        report.append({
            "scenario": name,
            "added_records": added,
            "matches": total_ok,
            "total": total_n,
            "match_rate": total_ok / total_n
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)

    if OUT.exists():
        raise FileExistsError(
            f"Report already exists: {OUT}"
        )

    with OUT.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(report[0])
        )
        writer.writeheader()
        writer.writerows(report)

    print("\n=== PHASE 8.6-C RECOVERY COMPLETE ===")
    print("Report:", OUT)


if __name__ == "__main__":
    main()
