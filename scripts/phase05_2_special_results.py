
import argparse
import re
import sqlite3
import unicodedata
from collections import defaultdict, Counter
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "boatrace.db"
SOURCE = ROOT / "data" / "extracted" / "results"

DATE_RE = re.compile(r"k(\d{2})(\d{2})(\d{2})\.txt$", re.I)
VENUE_RE = re.compile(r"^\s*(\d{2})KBGN")
RACE_RE = re.compile(
    r"^\s*(\d{1,2})R\b.*?\bH\s*\d{3,4}\s*m\b",
    re.I
)

# 先頭：結果コード、艇番、選手登録番号
ROW_START = re.compile(
    r"^\s*(?P<status>0?[1-6]|F|L\d?|K\d?|S\d?)"
    r"\s+(?P<lane>[1-6])\s+(?P<racer>\d{4})\s+"
    r"(?P<rest>.*)$"
)

# 後半：モーター、ボート、展示、進入、ST
# 結果コードによって欠測項目が存在する
NUM = r"\d+\.\d{2}"
MISSING = r"[A-Z]\s*\."
TAIL_RE = re.compile(
    r"(?P<motor>\d{1,3})\s+"
    r"(?P<boat>\d{1,3})\s+"
    r"(?P<exhibit>" + NUM + r"|" + MISSING + r")\s+"
    r"(?P<course>[1-6]|" + MISSING + r")\s+"
    r"(?P<st>[FL]?" + NUM + r"|" + MISSING + r")"
    r"(?=\s|$)"
)


# 欠場等で展示・進入・STが欠測している特殊行
K_TAIL_RE = re.compile(
    r"(?P<motor>\d{1,3})\s+"
    r"(?P<boat>\d{1,3})\s+"
    r"K\s*\.\s+K\s*\.\s+\.\s*\."
    r"\s*$"
)


def norm(s):
    return unicodedata.normalize("NFKC", s)


def date_from_name(path):
    m = DATE_RE.fullmatch(path.name)
    if not m:
        raise ValueError(f"Bad filename: {path.name}")
    yy, mm, dd = map(int, m.groups())
    year = 2000 + yy if yy <= 69 else 1900 + yy
    return date(year, mm, dd).isoformat()


def numeric_or_none(value):
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return None



def parse_row(line, key, source):
    m = ROW_START.match(line)
    if not m:
        return None

    g = m.groupdict()
    status = g["status"]
    rest = g["rest"]

    # K0/K1などの特殊欠場行
    if status.startswith("K"):
        special = K_TAIL_RE.search(rest)

        if special:
            return (
                key[0], key[1], key[2],
                int(g["lane"]),
                int(g["racer"]),
                status,
                None,  # 着順なし
                None,  # 進入コース不明
                None,  # STなし
                None,  # 展示タイムなし
                int(special.group("motor")),
                int(special.group("boat")),
                source
            )

    # 通常行・その他の特殊着順行
    tail = TAIL_RE.search(rest)

    if not tail:
        raise ValueError(
            "Unrecognized boat row: " + line[:140]
        )

    values = tail.groupdict()

    finish = int(status) if status.isdigit() else None

    if finish is not None and not 1 <= finish <= 6:
        finish = None

    st_raw = values["st"].replace(" ", "")
    st = numeric_or_none(st_raw)

    if st is None and st_raw.startswith("F"):
        fraction = numeric_or_none(st_raw[1:])
        st = -fraction if fraction is not None else None

    if st is None and st_raw.startswith("L"):
        st = numeric_or_none(st_raw[1:])

    course_raw = values["course"].strip()

    course = (
        int(course_raw)
        if course_raw.isdigit()
        else None
    )

    return (
        key[0], key[1], key[2],
        int(g["lane"]),
        int(g["racer"]),
        status,
        finish,
        course,
        st,
        numeric_or_none(values["exhibit"]),
        int(values["motor"]),
        int(values["boat"]),
        source
    )



def parse_file(path):
    day = date_from_name(path)
    source = str(path.relative_to(ROOT))
    text = path.read_text(encoding="cp932")

    races = defaultdict(list)
    problems = defaultdict(list)
    venue = None
    race = None
    active = False

    for number, raw in enumerate(text.splitlines(), 1):
        line = norm(raw)

        vm = VENUE_RE.match(line)
        if vm:
            venue = int(vm.group(1))
            if not 1 <= venue <= 24:
                raise ValueError("Invalid venue")
            race = None
            active = False
            continue

        if venue is None:
            continue

        rm = RACE_RE.match(line)
        if rm:
            race = int(rm.group(1))
            active = 1 <= race <= 12
            continue

        if not active:
            continue

        key = (day, venue, race)

        # 払戻欄に入ったら着順の読み込みを終了
        if re.match(r"^\s*(単勝|複勝|[23]連単|[23]連複)", line):
            active = False
            continue

        if not ROW_START.match(line):
            continue

        try:
            row = parse_row(line, key, source)
            if row is not None:
                races[key].append(row)
        except ValueError as exc:
            problems[key].append(f"line {number}: {exc}")

    return races, problems


def setup(conn):
    conn.execute("PRAGMA busy_timeout=30000")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS race_boat_results_v2 (
            race_date TEXT NOT NULL,
            venue_code INTEGER NOT NULL,
            race_no INTEGER NOT NULL,
            lane INTEGER NOT NULL,
            racer_id INTEGER NOT NULL,
            result_code TEXT NOT NULL,
            finish_position INTEGER,
            entry_course INTEGER,
            start_timing REAL,
            exhibition_time REAL,
            motor_no INTEGER,
            boat_no INTEGER,
            source_file TEXT NOT NULL,
            PRIMARY KEY(race_date, venue_code, race_no, lane)
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_results_v2_racer
        ON race_boat_results_v2(racer_id, race_date)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS race_results_v2_log (
            source_file TEXT PRIMARY KEY,
            file_size INTEGER NOT NULL,
            file_mtime_ns INTEGER NOT NULL,
            imported_races INTEGER NOT NULL,
            skipped_races INTEGER NOT NULL,
            imported_at TEXT NOT NULL
        )
    """)
    conn.commit()


def validate(conn, key, rows, errors):
    if errors:
        return errors[0]

    if len(rows) != 6:
        return f"Expected 6 boat rows, got {len(rows)}"

    lanes = [r[3] for r in rows]
    if set(lanes) != set(range(1, 7)):
        return "Invalid/duplicate lanes"

    rank_values = [
        r[6] for r in rows if r[6] is not None
    ]

    if len(rank_values) != len(set(rank_values)):
        return "Duplicate finish positions"

    if len(rank_values) < 3:
        return "Fewer than 3 normal finish positions"

    if sorted(rank_values)[:3] != [1, 2, 3]:
        return "Top 3 positions incomplete"

    for row in rows:
        _, _, _, lane, racer, status, finish, course, st, ex, motor, boat, _ = row

        if not 1 <= racer <= 9999:
            return f"Invalid racer number: {racer}"
        if course is not None and not 1 <= course <= 6:
            return f"Invalid course: {course}"
        if st is not None and not -1 < st < 2:
            return f"Invalid ST: {st}"
        if ex is not None and not 5 < ex < 10:
            return f"Invalid exhibition time: {ex}"

    program = conn.execute("""
        SELECT lane, racer_id
        FROM program_entries
        WHERE race_date=? AND venue_code=? AND race_no=?
    """, key).fetchall()

    if len(program) != 6:
        return "Missing complete program data"

    if dict(program) != {r[3]: r[4] for r in rows}:
        return "Program racer mismatch"

    payout = conn.execute("""
        SELECT combination
        FROM trifecta_results
        WHERE race_date=? AND venue_code=? AND race_no=?
    """, key).fetchone()

    if payout is None:
        return "Trifecta payout missing"

    first_three = sorted(
        (r for r in rows if r[6] is not None),
        key=lambda r: r[6]
    )[:3]

    combo = "-".join(str(r[3]) for r in first_three)

    if combo != payout[0]:
        return f"Trifecta mismatch: {combo} != {payout[0]}"

    return None


def process_file(conn, path, force):
    source = str(path.relative_to(ROOT))
    stat = path.stat()

    previous = conn.execute("""
        SELECT file_size, file_mtime_ns
        FROM race_results_v2_log
        WHERE source_file=?
    """, (source,)).fetchone()

    if not force and previous == (stat.st_size, stat.st_mtime_ns):
        return "cached", 0, [], Counter()

    races, problems = parse_file(path)
    keys = sorted(set(races) | set(problems))
    approved = []
    skipped = []
    status_counts = Counter()

    for key in keys:
        rows = races.get(key, [])
        error = validate(conn, key, rows, problems.get(key, []))

        if error is not None:
            skipped.append((key, error))
            continue

        approved.extend(rows)
        status_counts.update(r[5] for r in rows)

    with conn:
        conn.execute(
            "DELETE FROM race_boat_results_v2 WHERE source_file=?",
            (source,)
        )

        conn.executemany("""
            INSERT INTO race_boat_results_v2 VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, approved)

        conn.execute("""
            INSERT INTO race_results_v2_log VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_file) DO UPDATE SET
                file_size=excluded.file_size,
                file_mtime_ns=excluded.file_mtime_ns,
                imported_races=excluded.imported_races,
                skipped_races=excluded.skipped_races,
                imported_at=excluded.imported_at
        """, (
            source, stat.st_size, stat.st_mtime_ns,
            len(approved) // 6, len(skipped),
            datetime.now(timezone.utc).isoformat()
        ))

    return "imported", len(approved) // 6, skipped, status_counts


def report(conn):
    print("\n========== DATABASE SUMMARY ==========")

    boats = conn.execute(
        "SELECT COUNT(*) FROM race_boat_results_v2"
    ).fetchone()[0]

    races = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT 1 FROM race_boat_results_v2
            GROUP BY race_date, venue_code, race_no
        )
    """).fetchone()[0]

    print("Boat results:", f"{boats:,}")
    print("Complete races:", f"{races:,}")

    print("\nSPECIAL RESULT CODES")

    for code, count in conn.execute("""
        SELECT result_code, COUNT(*)
        FROM race_boat_results_v2
        WHERE finish_position IS NULL
        GROUP BY result_code
        ORDER BY COUNT(*) DESC
    """):
        print(code, count)

    print("\nCOURSE WIN STATISTICS")

    for course, starts, wins in conn.execute("""
        SELECT entry_course, COUNT(*),
               SUM(CASE WHEN finish_position=1
                   THEN 1 ELSE 0 END)
        FROM race_boat_results_v2
        WHERE entry_course BETWEEN 1 AND 6
        GROUP BY entry_course
        ORDER BY entry_course
    """):
        rate = 100 * wins / starts if starts else 0
        print(
            f"Course {course}: "
            f"starts={starts:,} wins={wins:,} "
            f"win_rate={rate:.2f}%"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be >= 0")

    files = sorted(SOURCE.rglob("k*.txt"), reverse=True)
    if args.limit:
        files = files[:args.limit]

    if not files:
        print("No result files found")
        return

    summary = Counter()

    with sqlite3.connect(DB, timeout=30) as conn:
        setup(conn)

        for path in files:
            try:
                state, count, skipped, codes = process_file(
                    conn, path, args.force
                )

                summary[state] += 1
                summary["races_imported"] += count
                summary["races_skipped"] += len(skipped)

                print(
                    f"[{state.upper()}] {path.name}: "
                    f"{count} races, {len(skipped)} skipped"
                )

                for key, reason in skipped[:8]:
                    print(f"  [SKIP] {key}: {reason}")

                if len(skipped) > 8:
                    print(
                        f"  ... and {len(skipped)-8} more"
                    )

            except Exception as exc:
                summary["failed"] += 1
                print(f"[ERROR] {path.name}: {exc}")

        print("\n========== IMPORT SUMMARY ==========")

        for item in (
            "imported", "cached", "failed",
            "races_imported", "races_skipped"
        ):
            print(f"{item}: {summary[item]}")

        report(conn)


if __name__ == "__main__":
    main()
