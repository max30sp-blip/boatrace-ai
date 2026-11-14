
import argparse
import re
import sqlite3
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone, date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "boatrace.db"
RESULTS_DIR = ROOT / "data" / "extracted" / "results"

DATE_RE = re.compile(r"k(\d{2})(\d{2})(\d{2})\.txt$", re.I)
VENUE_RE = re.compile(r"^\s*(\d{2})KBGN")
RACE_RE = re.compile(
    r"^\s*(\d{1,2})R\s+.*?\bH\s*\d{3,4}\s*m\b",
    re.I
)

ROW_RE = re.compile(
    r"^\s*(?P<rank>0?[1-6])\s+"
    r"(?P<lane>[1-6])\s+"
    r"(?P<racer_id>\d{4})\s+"
    r"(?P<name>.+?)\s+"
    r"(?P<motor>\d{1,3})\s+"
    r"(?P<boat>\d{1,3})\s+"
    r"(?P<exhibition>\d+\.\d{2})\s+"
    r"(?P<course>[1-6])\s+"
    r"(?P<st>[+-]?\d+\.\d{2})(?=\s|$)"
)


def normalize(value):
    return unicodedata.normalize("NFKC", value)


def file_date(path):
    match = DATE_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Invalid filename: {path.name}")

    yy, mm, dd = map(int, match.groups())
    year = 2000 + yy if yy <= 69 else 1900 + yy
    return date(year, mm, dd).isoformat()


def initialize(conn):
    conn.execute("PRAGMA busy_timeout=30000")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS race_boat_results (
            race_date TEXT NOT NULL,
            venue_code INTEGER NOT NULL,
            race_no INTEGER NOT NULL,
            lane INTEGER NOT NULL,
            racer_id INTEGER NOT NULL,
            finish_position INTEGER NOT NULL,
            entry_course INTEGER NOT NULL,
            start_timing REAL NOT NULL,
            exhibition_time REAL NOT NULL,
            motor_no INTEGER NOT NULL,
            boat_no INTEGER NOT NULL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (
                race_date, venue_code, race_no, lane
            )
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_boat_results_racer
        ON race_boat_results(racer_id, race_date)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_boat_results_course
        ON race_boat_results(venue_code, entry_course)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS race_results_import_log (
            source_file TEXT PRIMARY KEY,
            file_size INTEGER NOT NULL,
            file_mtime_ns INTEGER NOT NULL,
            imported_races INTEGER NOT NULL,
            skipped_races INTEGER NOT NULL,
            imported_at TEXT NOT NULL
        )
    """)

    conn.commit()


def parse_file(path):
    day = file_date(path)
    source = str(path.relative_to(ROOT))
    content = path.read_text(encoding="cp932")

    races = defaultdict(list)
    rejected = defaultdict(list)

    venue = None
    race_no = None
    in_race_table = False

    for line_number, raw in enumerate(
        content.splitlines(), start=1
    ):
        line = normalize(raw)

        venue_match = VENUE_RE.match(line)

        if venue_match:
            venue = int(venue_match.group(1))
            if not 1 <= venue <= 24:
                raise ValueError(
                    f"Invalid venue: {venue}"
                )
            race_no = None
            in_race_table = False
            continue

        if venue is None:
            continue

        race_match = RACE_RE.match(line)

        if race_match:
            candidate = int(race_match.group(1))

            race_no = candidate if 1 <= candidate <= 12 else None
            in_race_table = race_no is not None
            continue

        if not in_race_table or race_no is None:
            continue

        key = (day, venue, race_no)
        match = ROW_RE.match(line)

        if match:
            g = match.groupdict()

            record = (
                day,
                venue,
                race_no,
                int(g["lane"]),
                int(g["racer_id"]),
                int(g["rank"]),
                int(g["course"]),
                float(g["st"]),
                float(g["exhibition"]),
                int(g["motor"]),
                int(g["boat"]),
                source
            )

            races[key].append(record)

        elif re.match(r"^\s*(?:0?[1-6]|F|L|S|欠|失|転|落|妨)\s+[1-6]\s+", line):
            rejected[key].append(
                f"line {line_number}: {line.strip()[:100]}"
            )

        # 個別レースの払戻欄に移動したら終了
        if "単勝" in line or "3連単" in line:
            if races.get(key) or rejected.get(key):
                in_race_table = False

    return races, rejected


def validate_race(conn, key, rows, rejected):
    if rejected:
        return "Unsupported/special finish row"

    if len(rows) != 6:
        return f"Expected 6 finish rows, got {len(rows)}"

    lanes = [r[3] for r in rows]
    positions = [r[5] for r in rows]
    courses = [r[6] for r in rows]

    expected = set(range(1, 7))

    if set(lanes) != expected or len(set(lanes)) != 6:
        return "Invalid or duplicate boat lanes"

    if set(positions) != expected or len(set(positions)) != 6:
        return "Invalid or duplicate finish positions"

    if set(courses) != expected or len(set(courses)) != 6:
        return "Invalid or duplicate entry courses"

    for row in rows:
        st = row[7]
        exhibition = row[8]

        if not -1 < st < 2:
            return f"Invalid ST: {st}"

        if not 5 < exhibition < 10:
            return f"Invalid exhibition time: {exhibition}"

    # 番組表と選手登録番号を照合
    entries = conn.execute("""
        SELECT lane, racer_id
        FROM program_entries
        WHERE race_date = ?
          AND venue_code = ?
          AND race_no = ?
    """, key).fetchall()

    if len(entries) != 6:
        return f"Program entries not complete: {len(entries)}"

    program_map = dict(entries)
    result_map = {r[3]: r[4] for r in rows}

    if program_map != result_map:
        return "Racer IDs differ from program"

    # 3連単の結果と上位3艇が一致すること
    payout = conn.execute("""
        SELECT combination
        FROM trifecta_results
        WHERE race_date = ?
          AND venue_code = ?
          AND race_no = ?
    """, key).fetchone()

    if payout is None:
        return "Trifecta result missing"

    top_three = "-".join(
        str(r[3]) for r in sorted(
            rows, key=lambda r: r[5]
        )[:3]
    )

    if top_three != payout[0]:
        return (
            f"Trifecta mismatch: "
            f"finish={top_three}, payout={payout[0]}"
        )

    return None


def import_file(conn, path, force=False):
    stat = path.stat()
    source = str(path.relative_to(ROOT))

    previous = conn.execute("""
        SELECT file_size, file_mtime_ns
        FROM race_results_import_log
        WHERE source_file = ?
    """, (source,)).fetchone()

    if not force and previous == (
        stat.st_size, stat.st_mtime_ns
    ):
        return "cached", 0, 0, []

    races, rejected = parse_file(path)

    all_keys = sorted(set(races) | set(rejected))
    valid_rows = []
    skipped = []

    for key in all_keys:
        error = validate_race(
            conn,
            key,
            races.get(key, []),
            rejected.get(key, [])
        )

        if error:
            skipped.append((key, error))
        else:
            valid_rows.extend(races[key])

    imported_races = len(valid_rows) // 6

    with conn:
        # 再処理時はこのファイル由来のデータだけ更新
        conn.execute("""
            DELETE FROM race_boat_results
            WHERE source_file = ?
        """, (source,))

        conn.executemany("""
            INSERT INTO race_boat_results (
                race_date, venue_code, race_no,
                lane, racer_id, finish_position,
                entry_course, start_timing,
                exhibition_time, motor_no,
                boat_no, source_file
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, valid_rows)

        conn.execute("""
            INSERT INTO race_results_import_log
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_file)
            DO UPDATE SET
                file_size=excluded.file_size,
                file_mtime_ns=excluded.file_mtime_ns,
                imported_races=excluded.imported_races,
                skipped_races=excluded.skipped_races,
                imported_at=excluded.imported_at
        """, (
            source,
            stat.st_size,
            stat.st_mtime_ns,
            imported_races,
            len(skipped),
            datetime.now(timezone.utc).isoformat()
        ))

    return "imported", imported_races, len(skipped), skipped


def summary(conn):
    print("\n========== DATABASE SUMMARY ==========")

    total = conn.execute(
        "SELECT COUNT(*) FROM race_boat_results"
    ).fetchone()[0]

    races = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT race_date, venue_code, race_no
            FROM race_boat_results
            GROUP BY race_date, venue_code, race_no
        )
    """).fetchone()[0]

    print("Boat results:", f"{total:,}")
    print("Complete races:", f"{races:,}")

    print("\nCOURSE WIN STATISTICS")

    rows = conn.execute("""
        SELECT
            entry_course,
            COUNT(*) AS starts,
            SUM(
                CASE WHEN finish_position = 1
                THEN 1 ELSE 0 END
            ) AS wins,
            ROUND(
                100.0 * SUM(
                    CASE WHEN finish_position = 1
                    THEN 1 ELSE 0 END
                ) / COUNT(*), 2
            ) AS win_pct
        FROM race_boat_results
        GROUP BY entry_course
        ORDER BY entry_course
    """).fetchall()

    for course, starts, wins, win_pct in rows:
        print(
            f"Course {course}: "
            f"starts={starts:,} "
            f"wins={wins:,} "
            f"win_rate={win_pct}%"
        )

    print("\nSAMPLE RACE RESULTS")

    sample = conn.execute("""
        SELECT race_date, venue_code, race_no,
               lane, racer_id, finish_position,
               entry_course, start_timing
        FROM race_boat_results
        ORDER BY race_date DESC,
                 venue_code DESC,
                 race_no, finish_position
        LIMIT 12
    """).fetchall()

    for row in sample:
        print(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--limit", type=int, default=7)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be >= 0")

    files = sorted(
        RESULTS_DIR.rglob("k*.txt"),
        reverse=True
    )

    if args.limit:
        files = files[:args.limit]

    if not files:
        print("No result files found.")
        return

    stats = defaultdict(int)

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        initialize(conn)

        for path in files:
            try:
                status, count, skipped_count, warnings = (
                    import_file(conn, path, args.force)
                )

                stats[status] += 1
                stats["races_imported"] += count
                stats["races_skipped"] += skipped_count

                print(
                    f"[{status.upper()}] {path.name}: "
                    f"{count} races, {skipped_count} skipped"
                )

                for key, reason in warnings[:10]:
                    print(f"  [SKIP] {key}: {reason}")

                if len(warnings) > 10:
                    print(
                        f"  ... and {len(warnings)-10} more"
                    )

            except Exception as exc:
                stats["failed"] += 1
                print(f"[ERROR] {path.name}: {exc}")

        print("\n========== IMPORT SUMMARY ==========")

        for key in (
            "imported", "cached", "failed",
            "races_imported", "races_skipped"
        ):
            print(f"{key}: {stats[key]}")

        summary(conn)


if __name__ == "__main__":
    main()
