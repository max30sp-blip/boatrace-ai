
import sqlite3
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict, Counter

ROOT = Path(r"C:\boatrace_ai")
DB = ROOT / "data/boatrace.db"
TRAIN_DB = ROOT / "data/train_dataset_v1.db"
SOURCE = ROOT / "scripts/phase06_fast_features.py"

TARGET = "2026-10-29"
WINDOWS = (30, 90, 180)


def readonly(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    con = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True
    )
    con.execute("PRAGMA query_only=ON")
    return con


def valid_finish(value):
    try:
        return 1 <= int(value) <= 6
    except (TypeError, ValueError):
        return False


def as_int(value):
    return int(value) if value is not None else None


print("=== PHASE 8.6-C F0 RULE TEST ===")

# 元の高速F0実装も表示
if SOURCE.is_file():
    lines = SOURCE.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    print("\n=== ORIGINAL SUMMARIZE FUNCTION ===")

    for i in range(136, min(178, len(lines))):
        print(f"L{i+1:04d}: {lines[i]}")

con = readonly(DB)
train = readonly(TRAIN_DB)

programs = con.execute("""
    SELECT venue_code, race_no, lane, racer_id
    FROM program_entries
    WHERE race_date = ?
""", (TARGET,)).fetchall()

references = {}

for row in train.execute("""
    SELECT *
    FROM train_races
    WHERE race_date = ?
""", (TARGET,)):
    # SELECT * の列順はPRAGMAと一致
    if not references:
        names = [
            x[1] for x in train.execute(
                "PRAGMA table_info(train_races)"
            )
        ]

    item = dict(zip(names, row))
    references[
        (int(item["venue_code"]), int(item["race_no"]))
    ] = item

target_day = date.fromisoformat(TARGET)
history_start = (
    target_day - timedelta(days=180)
).isoformat()

racer_ids = {
    int(p[3]) for p in programs if p[3] is not None
}

histories = defaultdict(list)

for d, racer, finish in con.execute("""
    SELECT race_date, racer_id, finish_position
    FROM race_boat_results_v2
    WHERE race_date >= ?
      AND race_date < ?
""", (history_start, TARGET)):

    if racer is None or int(racer) not in racer_ids:
        continue

    histories[int(racer)].append(
        (date.fromisoformat(d), finish)
    )

con.close()
train.close()

# 境界条件と出走数の定義を組み合わせて比較
RULES = [
    ("exclusive_valid", False, False),
    ("inclusive_valid", True, False),
    ("exclusive_all", False, True),
    ("inclusive_all", True, True),
]

print("\n=== START COUNT RULE COMPARISON ===")

for window in WINDOWS:
    print(f"\n--- {window} DAYS ---")

    outcomes = {}

    for name, inclusive, count_all in RULES:
        matched = 0
        compared = 0
        differences = Counter()
        examples = []

        for venue, race, lane, racer in programs:
            ref = references.get(
                (int(venue), int(race))
            )

            if ref is None or racer is None:
                continue

            field = f"b{int(lane)}_f0_starts_{window}d"
            expected = ref.get(field)

            if expected is None:
                continue

            actual = 0

            for d, finish in histories[int(racer)]:
                delta = (target_day - d).days

                inside = (
                    0 < delta <= window
                    if inclusive
                    else 0 < delta < window
                )

                if not inside:
                    continue

                if count_all or valid_finish(finish):
                    actual += 1

            expected = int(expected)
            diff = actual - expected

            compared += 1
            differences[diff] += 1

            if diff == 0:
                matched += 1
            elif len(examples) < 3:
                examples.append((
                    int(venue), int(race),
                    int(lane), int(racer),
                    actual, expected
                ))

        accuracy = (
            100 * matched / compared
            if compared else 0
        )

        outcomes[name] = (
            accuracy, matched, compared,
            differences, examples
        )

        print(
            f"{name:18s}",
            f"matched={matched:3d}/{compared:3d}",
            f"rate={accuracy:.2f}%"
        )

    best_name = max(
        outcomes, key=lambda k: outcomes[k][0]
    )

    best = outcomes[best_name]

    print("Best candidate:", best_name)
    print(
        "Difference distribution:",
        best[3].most_common(8)
    )
    print("Mismatch examples:", best[4])

print("\n=== PHASE 8.6-C RULE TEST COMPLETE ===")
