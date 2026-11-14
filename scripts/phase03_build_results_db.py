
import argparse
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "extracted" / "results"
DB_PATH = ROOT / "data" / "boatrace.db"

# 競走成績の「3連単」列（先頭の組番と払戻金）
PAYOUT_RE = re.compile(
    r"^\s*(\d{1,2})R\s+"
    r"([1-6]-[1-6]-[1-6])\s+"
    r"([\d,]+)(?=\s|$)"
)

VENUE_RE = re.compile(r"^(\d{2})KBGN")
DATE_RE = re.compile(r"^k(\d{2})(\d{2})(\d{2})\.txt$", re.I)


def normalize(text):
    return unicodedata.normalize("NFKC", text)


def init_db():
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trifecta_results (
                race_date TEXT NOT NULL,
                venue_code INTEGER NOT NULL,
                race_no INTEGER NOT NULL,
                combination TEXT NOT NULL,
                payout_yen INTEGER NOT NULL,
                source_file TEXT NOT NULL,
                PRIMARY KEY (
                    race_date, venue_code, race_no
                )
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trifecta_combo
            ON trifecta_results(combination)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS trifecta_import_log (
                source_file TEXT PRIMARY KEY,
                file_size INTEGER NOT NULL,
                file_mtime_ns INTEGER NOT NULL,
                parsed_count INTEGER NOT NULL,
                imported_at TEXT NOT NULL
            )
        """)


def get_race_date(path):
    match = DATE_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Invalid filename: {path.name}")

    yy, mm, dd = map(int, match.groups())
    year = 2000 + yy if yy <= 69 else 1900 + yy

    return f"{year:04d}-{mm:02d}-{dd:02d}"


def parse_result_file(path):
    race_date = get_race_date(path)
    text = path.read_bytes().decode("cp932")

    records = []
    venue = None
    in_payout = False
    seen_races = set()

    for original_line in text.splitlines():
        line = normalize(original_line).strip()

        # 競艇場ごとの成績ブロック
        venue_match = VENUE_RE.match(line)
        if venue_match:
            venue = int(venue_match.group(1))
            if not 1 <= venue <= 24:
                raise ValueError(
                    f"Invalid venue code: {venue}"
                )
            in_payout = False
            seen_races = set()
            continue

        if venue is None:
            continue

        if "[払戻金]" in line:
            in_payout = True
            continue

        if not in_payout:
            continue

        match = PAYOUT_RE.match(line)

        if match:
            race_no = int(match.group(1))
            combination = match.group(2)
            payout = int(match.group(3).replace(",", ""))

            boats = combination.split("-")

            if not 1 <= race_no <= 12:
                raise ValueError(
                    f"Invalid race number: {race_no}"
                )

            if len(set(boats)) != 3:
                raise ValueError(
                    f"Invalid combination: {combination}"
                )

            if payout <= 0:
                raise ValueError(
                    f"Invalid payout: {payout}"
                )

            if race_no in seen_races:
                raise ValueError(
                    f"Duplicate payout: "
                    f"{race_date} venue={venue} race={race_no}"
                )

            seen_races.add(race_no)

            records.append((
                race_date,
                venue,
                race_no,
                combination,
                payout,
                str(path.relative_to(ROOT))
            ))

        elif seen_races and not re.match(r"^\d{1,2}R\b", line):
            # 通常の払戻一覧が終了したら次のセクションへ
            # 空行などは無視
            if line and not line.startswith("-"):
                in_payout = False

    return records


def import_file(conn, path):
    stat = path.stat()
    relative = str(path.relative_to(ROOT))

    previous = conn.execute("""
        SELECT file_size, file_mtime_ns
        FROM trifecta_import_log
        WHERE source_file = ?
    """, (relative,)).fetchone()

    if previous == (stat.st_size, stat.st_mtime_ns):
        return "cached", 0

    records = parse_result_file(path)

    if not records:
        return "empty", 0

    # 既存の同一日・同一場の結果は更新可能
    with conn:
        conn.executemany("""
            INSERT INTO trifecta_results (
                race_date, venue_code, race_no,
                combination, payout_yen, source_file
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                race_date, venue_code, race_no
            )
            DO UPDATE SET
                combination=excluded.combination,
                payout_yen=excluded.payout_yen,
                source_file=excluded.source_file
        """, records)

        conn.execute("""
            INSERT INTO trifecta_import_log (
                source_file, file_size,
                file_mtime_ns, parsed_count, imported_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_file)
            DO UPDATE SET
                file_size=excluded.file_size,
                file_mtime_ns=excluded.file_mtime_ns,
                parsed_count=excluded.parsed_count,
                imported_at=excluded.imported_at
        """, (
            relative,
            stat.st_size,
            stat.st_mtime_ns,
            len(records),
            datetime.now(timezone.utc).isoformat()
        ))

    return "imported", len(records)


def show_summary(conn):
    print("\n========== DATABASE SUMMARY ==========")

    total = conn.execute(
        "SELECT COUNT(*) FROM trifecta_results"
    ).fetchone()[0]

    print("Total races:", f"{total:,}")

    rows = conn.execute("""
        SELECT combination, COUNT(*) AS hits,
               ROUND(AVG(payout_yen), 0) AS avg_payout
        FROM trifecta_results
        GROUP BY combination
        ORDER BY hits DESC
        LIMIT 10
    """).fetchall()

    print("\nTOP 10 TRIFECTA COMBINATIONS")
    print("Combination | Count | Avg payout (yen/100yen)")

    for combo, count, avg_payout in rows:
        print(
            f"{combo:12} | {count:6} | {avg_payout:10.0f}"
        )

    venues = conn.execute("""
        SELECT COUNT(DISTINCT venue_code)
        FROM trifecta_results
    """).fetchone()[0]

    print("\nVenues:", venues)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=14)
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be >= 0")

    paths = sorted(
        RESULTS_DIR.rglob("k*.txt"),
        reverse=True
    )

    if args.limit:
        paths = paths[:args.limit]

    if not paths:
        print("No extracted results found.")
        return

    init_db()

    stats = {
        "imported": 0,
        "cached": 0,
        "empty": 0,
        "failed": 0
    }

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        for path in paths:
            try:
                status, count = import_file(conn, path)
                stats[status] += 1

                print(
                    f"[{status.upper()}] "
                    f"{path.name}: {count} races"
                )

            except Exception as exc:
                stats["failed"] += 1
                print(f"[ERROR] {path.name}: {exc}")

        print("\n========== IMPORT SUMMARY ==========")

        for key, value in stats.items():
            print(f"{key}: {value}")

        show_summary(conn)


if __name__ == "__main__":
    main()
