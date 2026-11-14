
import re
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "boatrace.db"
RESULTS = ROOT / "data" / "extracted" / "results"

VENUE_RE = re.compile(r"^\s*(\d{2})KBGN\s*$")
RACE_RE = re.compile(r"^\s*(\d{1,2})R\s+.*H\s*\d+\s*m")
FINISH_RE = re.compile(
    r"^\s*(\d{2}|S[012]|F|K[01]|L[01])\s+([1-6])\s+\d{4}\s+"
)


def get_missing():
    with sqlite3.connect(DB) as conn:
        return conn.execute("""
            SELECT k.race_date, k.venue_code,
                   k.race_no, k.combination
            FROM trifecta_results k
            WHERE NOT EXISTS (
                SELECT 1
                FROM race_boat_results_v2 r
                WHERE r.race_date = k.race_date
                  AND r.venue_code = k.venue_code
                  AND r.race_no = k.race_no
            )
            ORDER BY k.race_date, k.venue_code, k.race_no
        """).fetchall()


def scan_file(path):
    lines = path.read_text(
        encoding="cp932", errors="replace"
    ).splitlines()

    races = {}
    venue = None
    race = None
    entries = []

    def commit():
        if venue is not None and race is not None:
            races[(venue, race)] = entries.copy()

    for line in lines:
        vm = VENUE_RE.match(line)
        if vm:
            commit()
            venue = int(vm.group(1))
            race = None
            entries = []
            continue

        rm = RACE_RE.match(line)
        if rm:
            commit()
            race = int(rm.group(1))
            entries = []
            continue

        if race is None:
            continue

        fm = FINISH_RE.match(line)
        if fm:
            entries.append((fm.group(1), int(fm.group(2))))

    commit()
    return races


def classify(entries):
    if not entries:
        return "no_rows"

    positions = [
        int(rank) for rank, _ in entries
        if rank.isdigit()
    ]

    duplicate = any(
        count > 1 for count in Counter(positions).values()
    )

    special = [
        rank for rank, _ in entries
        if not rank.isdigit()
    ]

    if duplicate:
        return "dead_heat"

    if special:
        return "special_result"

    if len(entries) != 6:
        return "incomplete_rows"

    if sorted(positions) != [1, 2, 3, 4, 5, 6]:
        return "irregular_finish"

    return "other_parser_issue"


def main():
    missing = get_missing()
    grouped = defaultdict(list)

    for day, venue, race, combo in missing:
        grouped[day].append((venue, race, combo))

    stats = Counter()
    examples = defaultdict(list)

    print("========== PHASE 5.4 ==========")
    print("Missing races:", len(missing))

    for day, targets in sorted(grouped.items()):
        filename = "k" + day[2:].replace("-", "") + ".txt"
        path = RESULTS / day[:4] / filename

        if not path.exists():
            for venue, race, combo in targets:
                stats["missing_file"] += 1
            continue

        parsed = scan_file(path)

        for venue, race, combo in targets:
            entries = parsed.get((venue, race), [])
            kind = classify(entries)
            stats[kind] += 1

            if len(examples[kind]) < 5:
                examples[kind].append(
                    (day, venue, race, combo, entries)
                )

    print("\n=== CLASSIFICATION ===")
    for kind, count in stats.most_common():
        print(f"{kind}: {count}")

    print("\n=== EXAMPLES ===")
    for kind, rows in examples.items():
        print(f"\n[{kind}]")
        for row in rows:
            print(row)

    print("\nPHASE 5.4 COMPLETE")


if __name__ == "__main__":
    main()
