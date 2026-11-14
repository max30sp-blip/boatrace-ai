
import csv
import math
import sqlite3
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict, Counter

ROOT = Path(r"C:\boatrace_ai")

DB = ROOT / "data/boatrace.db"
TRAIN_DB = ROOT / "data/train_dataset_v1.db"
OUT = ROOT / "outputs/phase86c"

TARGET = "2026-10-29"
WINDOWS = (30, 90, 180)

F0_FIELDS = (
    "starts",
    "win_rate",
    "top3_rate",
    "avg_st"
)


def open_readonly(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    con = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True
    )
    con.execute("PRAGMA query_only=ON")
    return con


def norm_id(value):
    if value is None:
        return None
    return str(value).strip()


def number(value):
    if value is None or value == "":
        return None

    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def same_value(a, b):
    if a is None or b is None:
        return a is None

    return abs(a - b) <= 1e-6


def build_stats(events, target_day, window):
    selected = [
        (day, finish, st)
        for day, finish, st in events
        if 0 < (target_day - day).days < window
    ]

    valid = [
        (finish, st)
        for _, finish, st in selected
        if finish is not None and 1 <= finish <= 6
    ]

    starts = len(valid)

    wins = sum(
        finish == 1 for finish, st in valid
    )

    top3 = sum(
        finish <= 3 for finish, st in valid
    )

    st_values = [
        st for finish, st in valid
        if st is not None and 0 <= st <= 1
    ]

    return {
        "starts": starts,
        "win_rate": wins / starts if starts else None,
        "top3_rate": top3 / starts if starts else None,
        "avg_st": (
            sum(st_values) / len(st_values)
            if st_values else None
        )
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    output_file = OUT / f"f0_comparison_{TARGET}.csv"

    if output_file.exists():
        raise FileExistsError(
            f"Comparison already exists: {output_file}"
        )

    target_day = date.fromisoformat(TARGET)
    history_start = (
        target_day - timedelta(days=180)
    ).isoformat()

    con = open_readonly(DB)
    train = open_readonly(TRAIN_DB)

    programs = con.execute("""
        SELECT race_date, venue_code, race_no,
               lane, racer_id, grade
        FROM program_entries
        WHERE race_date = ?
        ORDER BY venue_code, race_no, lane
    """, (TARGET,)).fetchall()

    if not programs:
        raise ValueError("No program entries")

    racer_ids = {
        norm_id(row[4]) for row in programs
    }

    # 予測対象日より前のレース結果のみ取得
    history_rows = con.execute("""
        SELECT race_date, racer_id,
               finish_position, start_timing
        FROM race_boat_results_v2
        WHERE race_date >= ?
          AND race_date < ?
        ORDER BY race_date
    """, (history_start, TARGET))

    histories = defaultdict(list)
    history_count = 0

    for d, racer, finish, st in history_rows:
        rid = norm_id(racer)

        if rid not in racer_ids:
            continue

        try:
            finish = (
                int(finish)
                if finish is not None else None
            )
        except (ValueError, TypeError):
            finish = None

        histories[rid].append((
            date.fromisoformat(d),
            finish,
            number(st)
        ))

        history_count += 1

    # 2026年10月29日の既存学習データ
    cur = train.execute("""
        SELECT *
        FROM train_races
        WHERE race_date = ?
    """, (TARGET,))

    names = [d[0] for d in cur.description]

    reference = {}

    for raw in cur:
        row = dict(zip(names, raw))

        k = (
            int(row["venue_code"]),
            int(row["race_no"])
        )

        if k in reference:
            raise ValueError(f"Duplicate reference: {k}")

        reference[k] = row

    print("=== PHASE 8.6-C F0 VALIDATION ===")
    print("Target date:", TARGET)
    print("Program entries:", len(programs))
    print("Unique racers:", len(racer_ids))
    print("Historical entries:", history_count)
    print("Reference races:", len(reference))

    comparisons = []
    matched = Counter()
    mismatched = Counter()
    absent = Counter()
    grade_pairs = defaultdict(Counter)

    for d, venue, race, lane, racer, grade in programs:
        venue = int(venue)
        race = int(race)
        lane = int(lane)

        expected = reference.get((venue, race))
        if expected is None:
            absent["reference_race"] += 1
            continue

        racer = norm_id(racer)

        output = {
            "race_date": d,
            "venue_code": venue,
            "race_no": race,
            "lane": lane,
            "racer_id": racer,
            "grade": grade,
            "reference_grade_rank": expected.get(
                f"b{lane}_p_grade_rank"
            )
        }

        if grade is not None:
            grade_pairs[str(grade)][
                str(output["reference_grade_rank"])
            ] += 1

        for window in WINDOWS:
            values = build_stats(
                histories.get(racer, []),
                target_day,
                window
            )

            for field in F0_FIELDS:
                col = f"b{lane}_f0_{field}_{window}d"

                actual = values[field]
                reference_value = number(
                    expected.get(col)
                )

                prefix = f"{field}_{window}d"

                output[f"computed_{prefix}"] = actual
                output[f"reference_{prefix}"] = (
                    reference_value
                )

                if same_value(actual, reference_value):
                    matched[prefix] += 1
                else:
                    mismatched[prefix] += 1

        comparisons.append(output)

    con.close()
    train.close()

    if not comparisons:
        raise ValueError(
            "No program entries matched training DB"
        )

    columns = list(comparisons[0])

    with output_file.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(comparisons)

    print("\n=== GRADE MAPPING OBSERVATIONS ===")

    for grade, ranks in sorted(grade_pairs.items()):
        print(
            f"{grade}:",
            dict(ranks)
        )

    print("\n=== F0 MATCH RESULTS ===")

    for window in WINDOWS:
        print(f"\nWindow: {window} days")

        for field in F0_FIELDS:
            k = f"{field}_{window}d"

            ok = matched[k]
            ng = mismatched[k]
            total = ok + ng

            percentage = (
                ok / total * 100 if total else 0
            )

            print(
                f"{field:12s}",
                f"matched={ok:4d}",
                f"mismatched={ng:4d}",
                f"rate={percentage:.2f}%"
            )

    print("\n=== VALIDATION SUMMARY ===")
    print("Compared boat entries:", len(comparisons))
    print("Missing reference entries:", sum(absent.values()))
    print("Total matching values:", sum(matched.values()))
    print("Total mismatching values:", sum(mismatched.values()))

    print("\n=== PHASE 8.6-C F0 VALIDATION COMPLETE ===")
    print("Output:", output_file)


if __name__ == "__main__":
    main()
