
import argparse
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone, date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROGRAM_DIR = ROOT / "data" / "extracted" / "programs"
DB_PATH = ROOT / "data" / "boatrace.db"

DATE_PATTERN = re.compile(r"b(\d{2})(\d{2})(\d{2})\.txt$", re.I)
VENUE_PATTERN = re.compile(r"^\s*(\d{2})BBGN")
RACE_PATTERN = re.compile(r"^\s*(\d{1,2})R\s+")
ENTRY_START = re.compile(r"^\s*([1-6])\s+(\d{4})")


ENTRY_PATTERN = re.compile(
    r"^\s*(?P<lane>[1-6])\s+"
    r"(?P<racer_id>\d{4})"
    r"(?P<name>.+?)"
    r"(?P<age>\d{2})"
    r"(?P<branch>[^\d\s]{2})"
    r"(?P<weight>\d{2,3})"
    r"(?P<grade>[AB][12])\s*"
    r"(?P<national_win>\d+\.\d{2})\s*"
    r"(?P<national_2rate>\d+\.\d{2})\s*"
    r"(?P<local_win>\d+\.\d{2})\s*"
    r"(?P<local_2rate>\d+\.\d{2})\s*"
    r"(?P<motor_no>\d{1,3})\s*"
    r"(?P<motor_2rate>\d+\.\d{2})\s*"
    r"(?P<boat_no>\d{1,3})\s*"
    r"(?P<boat_2rate>\d+\.\d{2})"
    r"(?P<tail>.*)$"
)



def normalize(text):
    return unicodedata.normalize("NFKC", text)


def get_date(path):
    match = DATE_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(f"Invalid filename: {path.name}")

    yy, mm, dd = map(int, match.groups())
    year = 2000 + yy if yy <= 69 else 1900 + yy

    return date(year, mm, dd).isoformat()


def init_db(conn):
    conn.execute("PRAGMA busy_timeout=30000")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS program_entries (
            race_date TEXT NOT NULL,
            venue_code INTEGER NOT NULL,
            race_no INTEGER NOT NULL,
            lane INTEGER NOT NULL,
            racer_id INTEGER NOT NULL,
            racer_name TEXT NOT NULL,
            age INTEGER,
            branch TEXT,
            weight_kg INTEGER,
            grade TEXT,
            national_win_rate REAL,
            national_2rate REAL,
            local_win_rate REAL,
            local_2rate REAL,
            motor_no INTEGER,
            motor_2rate REAL,
            boat_no INTEGER,
            boat_2rate REAL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (
                race_date, venue_code, race_no, lane
            )
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entries_racer
        ON program_entries(racer_id, race_date)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entries_race
        ON program_entries(
            race_date, venue_code, race_no
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS program_import_log (
            source_file TEXT PRIMARY KEY,
            file_size INTEGER NOT NULL,
            file_mtime_ns INTEGER NOT NULL,
            entries INTEGER NOT NULL,
            warnings INTEGER NOT NULL,
            imported_at TEXT NOT NULL
        )
    """)

    conn.commit()


def parse_entry(line, day, venue, race_no, source):
    match = ENTRY_PATTERN.match(line)

    if not match:
        raise ValueError("Unrecognized entry: " + line[:120])

    g = match.groupdict()

    values = [
        float(g[key])
        for key in (
            "national_2rate", "local_2rate",
            "motor_2rate", "boat_2rate"
        )
    ]

    if any(not 0 <= v <= 100 for v in values):
        raise ValueError("Invalid 2-rate: " + line[:120])

    return (
        day,
        venue,
        race_no,
        int(g["lane"]),
        int(g["racer_id"]),
        " ".join(g["name"].split()),
        int(g["age"]),
        g["branch"],
        int(g["weight"]),
        g["grade"],
        float(g["national_win"]),
        float(g["national_2rate"]),
        float(g["local_win"]),
        float(g["local_2rate"]),
        int(g["motor_no"]),
        float(g["motor_2rate"]),
        int(g["boat_no"]),
        float(g["boat_2rate"]),
        source
    )


def parse_file(path):
    day = get_date(path)
    source = str(path.relative_to(ROOT))
    text = path.read_text(encoding="cp932")

    records = []
    warnings = []

    venue = None
    race_no = None
    seen = set()

    for line_number, raw_line in enumerate(
        text.splitlines(), start=1
    ):
        line = normalize(raw_line)

        venue_match = VENUE_PATTERN.match(line)

        if venue_match:
            venue = int(venue_match.group(1))
            if not 1 <= venue <= 24:
                raise ValueError(f"Invalid venue: {venue}")
            race_no = None
            continue

        race_match = RACE_PATTERN.match(line)

        if race_match and venue is not None:
            candidate = int(race_match.group(1))

            if 1 <= candidate <= 12:
                race_no = candidate
                continue

        if venue is None or race_no is None:
            continue

        if not ENTRY_START.match(line):
            continue

        try:
            record = parse_entry(
                line, day, venue, race_no, source
            )

            key = (
                record[0], record[1],
                record[2], record[3]
            )

            if key in seen:
                warnings.append(
                    f"line {line_number}: duplicate {key}"
                )
                continue

            seen.add(key)
            records.append(record)

        except ValueError as exc:
            warnings.append(
                f"line {line_number}: {exc}"
            )

    # 正常なレースは通常6艇分存在する
    race_lanes = {}

    for record in records:
        race_key = record[:3]
        race_lanes.setdefault(
            race_key, set()
        ).add(record[3])

    incomplete = [
        key for key, lanes in race_lanes.items()
        if lanes != {1, 2, 3, 4, 5, 6}
    ]

    for key in incomplete:
        warnings.append(
            f"Incomplete race {key}: "
            f"lanes={sorted(race_lanes[key])}"
        )

    return records, warnings


def import_file(conn, path):
    relative = str(path.relative_to(ROOT))
    stat = path.stat()

    previous = conn.execute("""
        SELECT file_size, file_mtime_ns
        FROM program_import_log
        WHERE source_file = ?
    """, (relative,)).fetchone()

    if previous == (stat.st_size, stat.st_mtime_ns):
        return "cached", 0, []

    records, warnings = parse_file(path)

    if not records:
        return "empty", 0, warnings

    # データの破損や予期しない形式を見逃さない
    # 警告があるファイルは登録せず、原因を確認
    if warnings:
        return "review", len(records), warnings

    with conn:
        conn.execute(
            "DELETE FROM program_entries WHERE source_file = ?",
            (relative,)
        )

        conn.executemany("""
            INSERT INTO program_entries VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, records)

        conn.execute("""
            INSERT INTO program_import_log VALUES (
                ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(source_file) DO UPDATE SET
                file_size=excluded.file_size,
                file_mtime_ns=excluded.file_mtime_ns,
                entries=excluded.entries,
                warnings=excluded.warnings,
                imported_at=excluded.imported_at
        """, (
            relative,
            stat.st_size,
            stat.st_mtime_ns,
            len(records),
            0,
            datetime.now(timezone.utc).isoformat()
        ))

    return "imported", len(records), []


def summary(conn):
    print("\n========== DATABASE SUMMARY ==========")

    total = conn.execute(
        "SELECT COUNT(*) FROM program_entries"
    ).fetchone()[0]

    races = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT race_date, venue_code, race_no
            FROM program_entries
            GROUP BY race_date, venue_code, race_no
        )
    """).fetchone()[0]

    venues = conn.execute("""
        SELECT COUNT(DISTINCT venue_code)
        FROM program_entries
    """).fetchone()[0]

    print("Entries:", f"{total:,}")
    print("Races:", f"{races:,}")
    print("Venues:", venues)

    if races:
        print("Average entries per race:", round(total/races, 3))

    print("\nSAMPLE ENTRIES")

    rows = conn.execute("""
        SELECT race_date, venue_code, race_no,
               lane, racer_id, racer_name, grade,
               national_win_rate, motor_2rate
        FROM program_entries
        ORDER BY race_date DESC, venue_code, race_no, lane
        LIMIT 12
    """).fetchall()

    for row in rows:
        print(row)

    print("\nRESULT MATCH CHECK")

    has_results = conn.execute("""
        SELECT COUNT(*) FROM sqlite_master
        WHERE type='table' AND name='trifecta_results'
    """).fetchone()[0]

    if has_results:
        matched = conn.execute("""
            SELECT COUNT(*) FROM trifecta_results r
            WHERE EXISTS (
                SELECT 1 FROM program_entries e
                WHERE e.race_date = r.race_date
                AND e.venue_code = r.venue_code
                AND e.race_no = r.race_no
            )
        """).fetchone()[0]

        result_count = conn.execute(
            "SELECT COUNT(*) FROM trifecta_results"
        ).fetchone()[0]

        print("Matched races:", matched)
        print("Result races:", result_count)
        print(
            "Not matched yet:",
            result_count - matched
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=7
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be >= 0")

    files = sorted(
        PROGRAM_DIR.rglob("b*.txt"),
        reverse=True
    )

    if args.limit:
        files = files[:args.limit]

    if not files:
        print("No program files found.")
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        init_db(conn)

        stats = {
            "imported": 0,
            "cached": 0,
            "empty": 0,
            "review": 0,
            "failed": 0
        }

        for path in files:
            try:
                status, count, warnings = import_file(
                    conn, path
                )

                stats[status] += 1

                print(
                    f"[{status.upper()}] "
                    f"{path.name}: {count} entries"
                )

                for warning in warnings[:5]:
                    print("  [WARN]", warning)

                if len(warnings) > 5:
                    print(
                        f"  ...and {len(warnings)-5} "
                        f"more warnings"
                    )

            except Exception as exc:
                stats["failed"] += 1
                print(f"[ERROR] {path.name}: {exc}")

        print("\n========== IMPORT SUMMARY ==========")

        for key, value in stats.items():
            print(f"{key}: {value}")

        summary(conn)


if __name__ == "__main__":
    main()
