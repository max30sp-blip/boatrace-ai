
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "boatrace.db"
RESULT_DIR = ROOT / "data" / "extracted" / "results"

VENUE_RE = re.compile(r"^\s*(\d{2})KBGN\s*$")
END_RE = re.compile(r"^\s*\d{2}KEND\s*$")
RACE_RE = re.compile(r"^\s*(\d{1,2})R\s+.*H\s*\d+\s*m")
BOAT_RE = re.compile(
    r"^\s*(0[1-6]|S[012]|F|K[01]|L[01])\s+"
    r"([1-6])\s+(\d{4})\s+(.+?)\s*$"
)

# モーター・ボート番号、展示、進入、STを末尾から解析
DETAIL_RE = re.compile(
    r"(\d{1,3})\s+(\d{1,3})\s+"
    r"(\d{1,2}\.\d{2})\s+"
    r"([1-6])\s+"
    r"([FfLl]?\d*\.\d{2})\b"
)


def normalize(line):
    return unicodedata.normalize("NFKC", line)


def get_missing(conn):
    return conn.execute("""
        SELECT k.race_date, k.venue_code, k.race_no
        FROM trifecta_results k
        WHERE NOT EXISTS (
            SELECT 1 FROM race_boat_results_v2 r
            WHERE r.race_date = k.race_date
              AND r.venue_code = k.venue_code
              AND r.race_no = k.race_no
        )
        ORDER BY k.race_date, k.venue_code, k.race_no
    """).fetchall()


def create_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS
        race_boat_results_supplement (
            race_date TEXT NOT NULL,
            venue_code INTEGER NOT NULL,
            race_no INTEGER NOT NULL,
            boat_no INTEGER NOT NULL,
            racer_id INTEGER NOT NULL,
            finish_code TEXT NOT NULL,
            finish_position INTEGER,
            entry_course INTEGER,
            exhibition_time REAL,
            start_timing REAL,
            raw_line TEXT NOT NULL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (
                race_date, venue_code, race_no, boat_no
            )
        );

        CREATE TABLE IF NOT EXISTS
        trifecta_payouts_supplement (
            race_date TEXT NOT NULL,
            venue_code INTEGER NOT NULL,
            race_no INTEGER NOT NULL,
            combination TEXT NOT NULL,
            payout_yen INTEGER NOT NULL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (
                race_date, venue_code, race_no, combination
            )
        );
    """)


def parse_file(path):
    lines = path.read_text(
        encoding="cp932", errors="replace"
    ).splitlines()

    race_sections = {}
    venue = None
    race = None
    section = []

    def save_section():
        if venue is not None and race is not None:
            race_sections[(venue, race)] = section.copy()

    for line in lines:
        vm = VENUE_RE.match(line)
        if vm:
            save_section()
            venue = int(vm.group(1))
            race = None
            section = []
            continue

        if END_RE.match(line):
            save_section()
            venue = None
            race = None
            section = []
            continue

        rm = RACE_RE.match(line)
        if rm and venue is not None:
            save_section()
            race = int(rm.group(1))
            section = [line]
            continue

        if venue is not None and race is not None:
            section.append(line)

    save_section()
    return race_sections


def parse_boats(section):
    boats = []
    started = False

    for line in section[1:]:
        match = BOAT_RE.match(line)

        if not match:
            if started:
                break
            continue

        started = True

        code, boat, racer, tail = match.groups()

        position = int(code) if code.isdigit() else None

        course = None
        exhibition = None
        st = None

        detail = DETAIL_RE.search(tail)

        if detail:
            exhibition = float(detail.group(3))
            course = int(detail.group(4))
            st_text = detail.group(5)

            if re.fullmatch(r"\d*\.\d{2}", st_text):
                st = float(st_text)

        boats.append({
            "boat_no": int(boat),
            "racer_id": int(racer),
            "finish_code": code,
            "finish_position": position,
            "entry_course": course,
            "exhibition_time": exhibition,
            "start_timing": st,
            "raw_line": line,
        })

    return boats


def parse_payouts(section):
    payouts = []
    reading = False

    for raw_line in section:
        line = normalize(raw_line)

        if "3連単" in line:
            reading = True
            line = line.split("3連単", 1)[1]

        elif reading and "3連複" in line:
            break

        elif not reading:
            continue

        matches = re.findall(
            r"([1-6]-[1-6]-[1-6])\s+([\d,]+)",
            line
        )

        for combination, amount in matches:
            a, b, c = map(int, combination.split("-"))

            if len({a, b, c}) != 3:
                continue

            payouts.append((
                combination,
                int(amount.replace(",", ""))
            ))

    # 同じ的中組み合わせの重複を排除
    return list(dict.fromkeys(payouts))


def classify(boats):
    positions = [
        b["finish_position"]
        for b in boats
        if b["finish_position"] is not None
    ]

    counts = Counter(positions)

    dead_heat = any(v > 1 for v in counts.values())
    special = any(
        b["finish_position"] is None
        for b in boats
    )

    if dead_heat and special:
        return "dead_heat_and_special"
    if dead_heat:
        return "dead_heat"
    if special:
        return "special_result"
    return "other"


def process_one(conn, day, venue, race, section, source):
    boats = parse_boats(section)
    payouts = parse_payouts(section)

    boat_numbers = [b["boat_no"] for b in boats]

    # 不完全な解析結果は登録しない
    if (
        len(boats) != 6
        or len(set(boat_numbers)) != 6
        or not payouts
    ):
        return "review", len(boats), len(payouts)

    # 同着では複数的中を保持する
    # ここでは払戻情報を元Kファイルから取得し、
    # 勝手に着順から払戻金を推定しない
    for b in boats:
        conn.execute("""
            INSERT OR REPLACE INTO
            race_boat_results_supplement (
                race_date, venue_code, race_no,
                boat_no, racer_id,
                finish_code, finish_position,
                entry_course, exhibition_time,
                start_timing, raw_line, source_file
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            day, venue, race,
            b["boat_no"], b["racer_id"],
            b["finish_code"], b["finish_position"],
            b["entry_course"], b["exhibition_time"],
            b["start_timing"], b["raw_line"], str(source)
        ))

    for combination, payout in payouts:
        conn.execute("""
            INSERT OR REPLACE INTO
            trifecta_payouts_supplement (
                race_date, venue_code, race_no,
                combination, payout_yen, source_file
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            day, venue, race,
            combination, payout, str(source)
        ))

    return classify(boats), len(boats), len(payouts)


def main():
    if not DB.exists():
        raise FileNotFoundError(DB)

    with sqlite3.connect(DB, timeout=60) as conn:
        conn.execute("PRAGMA busy_timeout=60000")

        missing = get_missing(conn)
        print("========== PHASE 5.5 ==========")
        print("Target races:", len(missing))

        if not missing:
            print("No missing races detected")
            return

        # データベースを変更する前に対象を取得
        grouped = defaultdict(list)
        for day, venue, race in missing:
            grouped[day].append((venue, race))

        # 全ての対象を解析してから登録開始
        prepared = []
        review = []

        for day, targets in sorted(grouped.items()):
            filename = "k" + day[2:].replace("-", "") + ".txt"
            path = RESULT_DIR / day[:4] / filename

            if not path.exists():
                for venue, race in targets:
                    review.append((day, venue, race, "missing_file"))
                continue

            sections = parse_file(path)

            for venue, race in targets:
                section = sections.get((venue, race))

                if section is None:
                    review.append((day, venue, race, "missing_section"))
                    continue

                boats = parse_boats(section)
                payouts = parse_payouts(section)

                if (
                    len(boats) != 6
                    or len({b["boat_no"] for b in boats}) != 6
                    or not payouts
                ):
                    review.append((
                        day, venue, race,
                        f"boats={len(boats)},payouts={len(payouts)}"
                    ))
                    continue

                prepared.append((
                    day, venue, race, section, path
                ))

        print("Prepared races:", len(prepared))
        print("Review races:", len(review))

        if review:
            print("\n=== REVIEW EXAMPLES ===")
            for item in review[:20]:
                print(item)

        create_tables(conn)

        stats = Counter()
        boats_added = 0
        payouts_added = 0

        with conn:
            for day, venue, race, section, path in prepared:
                status, nb, np = process_one(
                    conn, day, venue, race, section, path
                )
                stats[status] += 1
                boats_added += nb
                payouts_added += np

        print("\n=== IMPORT SUMMARY ===")
        print("Stored race records:", sum(stats.values()))
        print("Boat rows processed:", boats_added)
        print("Payout rows processed:", payouts_added)

        for status, count in sorted(stats.items()):
            print(f"{status}: {count}")

        for table in (
            "race_boat_results_supplement",
            "trifecta_payouts_supplement"
        ):
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            print(f"{table}: {n:,}")

        print("\n=== VERIFICATION ===")
        for day, venue, race in (
            ("2016-01-02", 18, 12),
            ("2019-02-23", 3, 12),
        ):
            result = conn.execute("""
                SELECT combination, payout_yen
                FROM trifecta_payouts_supplement
                WHERE race_date = ?
                  AND venue_code = ?
                  AND race_no = ?
                ORDER BY combination
            """, (day, venue, race)).fetchall()

            print(day, venue, race, result)

        print("\nPHASE 5.5 COMPLETE")


if __name__ == "__main__":
    main()
