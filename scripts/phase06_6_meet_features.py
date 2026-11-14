
import argparse
import re
import sqlite3
import unicodedata
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
PROGRAM_DIR = ROOT / "data" / "extracted" / "programs"
SOURCE_DB = ROOT / "data" / "boatrace.db"
OUTPUT_DB = ROOT / "data" / "meet_features.db"

TABLE = "racer_meet_features"
BUILD = "racer_meet_features_build"
BATCH_SIZE = 10000


def normalize(s):
    return unicodedata.normalize("NFKC", s)


def parse_program(path):
    text = path.read_text(
        encoding="cp932", errors="replace"
    )

    entries = {}
    venue_days = {}

    venue = None
    race = None
    meet_day = None
    seen_lanes = set()

    for raw in text.splitlines():
        line = normalize(raw)

        header = re.match(r"^(\d{2})BBGN", line)
        if header:
            venue = int(header.group(1))
            race = None
            meet_day = None
            seen_lanes = set()
            continue

        if venue is None:
            continue

        if race is None:
            m = re.search(r"第\s*(\d+)\s*日", line)
            if m:
                meet_day = int(m.group(1))
                venue_days[venue] = meet_day

        m = re.match(r"^\s*(\d{1,2})R\s", line)
        if m:
            n = int(m.group(1))
            race = n if 1 <= n <= 12 else None
            seen_lanes = set()
            continue

        if race is None or meet_day is None:
            continue

        if len(seen_lanes) == 6:
            continue

        m = re.match(r"^([1-6])\s+(\d{4})", raw)
        if not m or len(raw) < 71:
            continue

        lane = int(m.group(1))
        racer = int(m.group(2))

        if lane in seen_lanes:
            continue

        seen_lanes.add(lane)

        # 固定長の今節成績欄
        form = raw[59:71]

        key = (venue, race, lane, racer)
        entries[key] = (meet_day, form)

    return entries, venue_days


def parse_form(meet_day, form):
    # 最大6日分・各日2文字の表示
    days = min(max(meet_day - 1, 0), 6)
    section = form[:days * 2]

    values = []
    unknown = 0

    for ch in section:
        if ch.isspace():
            continue

        if ch in "123456":
            values.append(ch)
        elif ch.upper() in ("F", "L", "S", "K"):
            values.append(ch.upper())
        else:
            unknown += 1

    finishes = [
        int(x) for x in values
        if x in "123456"
    ]

    n = len(finishes)

    return (
        len(values),
        n,
        sum(x == 1 for x in finishes) / n if n else None,
        sum(x <= 3 for x in finishes) / n if n else None,
        sum(finishes) / n if n else None,
        values.count("F"),
        values.count("L"),
        unknown
    )


def parse_st(value):
    try:
        st = float(value)
    except (ValueError, TypeError):
        return None

    if 0 <= st <= 1:
        return st

    return None


def calculate_st(values):
    n = len(values)

    if n == 0:
        return 0, None, None

    mean = sum(values) / n

    if n < 2:
        return n, mean, None

    variance = sum(
        (x - mean) ** 2 for x in values
    ) / n

    return n, mean, variance ** 0.5


def new_state(day, current_date, partial):
    return {
        "day": day,
        "date": current_date,
        "partial": int(partial),
        "st": defaultdict(list),
    }


def update_meet_state(states, venue, day, current_date):
    """
    競艇場ごとの開催節を管理する。

    day=1 は必ず新しい節。
    日数が戻る場合も新しい節として扱う。
    不連続な日程は不確実性を記録する。
    """
    previous = states.get(venue)

    if previous is None:
        states[venue] = new_state(
            day, current_date, day != 1
        )
        return

    previous_day = previous["day"]
    previous_date = previous["date"]

    if day == 1:
        states[venue] = new_state(
            day, current_date, False
        )
        return

    if day < previous_day:
        states[venue] = new_state(
            day, current_date, True
        )
        return

    if day > previous_day + 1:
        # 中間日が未観測の可能性
        states[venue] = new_state(
            day, current_date, True
        )
        return

    if day == previous_day:
        # 順延などの可能性があるため、
        # 履歴は残すが完全性は保証しない
        previous["partial"] = 1

    if (
        current_date - previous_date
        > timedelta(days=1)
    ):
        # 空白日に未収集レースがないか未確認
        previous["partial"] = 1

    previous["day"] = day
    previous["date"] = current_date


def load_results_for_date(src, rd):
    """
    対象日に確定したSTを取得する。
    当日の特徴量を保存した後にだけ利用する。
    """
    sql = """
        SELECT venue_code, racer_id, start_timing
        FROM race_boat_results_v2
        WHERE race_date = ?

        UNION ALL

        SELECT venue_code, racer_id, start_timing
        FROM race_boat_results_supplement
        WHERE race_date = ?
    """

    return src.execute(sql, (rd, rd))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    if args.end:
        date.fromisoformat(args.end)

    files = sorted(PROGRAM_DIR.glob("*/b*.txt"))

    if args.end:
        limit = args.end.replace("-", "")
        files = [
            p for p in files
            if re.fullmatch(r"b\d{6}", p.stem)
            and ("20" + p.stem[1:]) <= limit
        ]

    src = sqlite3.connect(
        SOURCE_DB.as_uri() + "?mode=ro",
        uri=True,
        timeout=60
    )
    dst = sqlite3.connect(OUTPUT_DB, timeout=60)

    try:
        dst.execute(f"DROP TABLE IF EXISTS {BUILD}")

        dst.execute(f"""
            CREATE TABLE {BUILD} (
                race_date TEXT NOT NULL,
                venue_code INTEGER NOT NULL,
                race_no INTEGER NOT NULL,
                lane INTEGER NOT NULL,
                racer_id INTEGER NOT NULL,

                meet_day INTEGER,
                meet_records INTEGER NOT NULL,
                meet_valid_finishes INTEGER NOT NULL,

                meet_win_rate REAL,
                meet_top3_rate REAL,
                meet_avg_finish REAL,

                meet_f_count INTEGER NOT NULL,
                meet_l_count INTEGER NOT NULL,
                meet_unknown_chars INTEGER NOT NULL,

                meet_st_count INTEGER NOT NULL,
                meet_avg_st REAL,
                meet_st_std REAL,
                meet_st_partial INTEGER NOT NULL,

                PRIMARY KEY (
                    race_date, venue_code, race_no, lane
                )
            )
        """)

        dst.commit()

        placeholders = ",".join(["?"] * 18)
        insert_sql = (
            f"INSERT INTO {BUILD} VALUES ({placeholders})"
        )

        states = {}
        batch = []

        total = 0
        unmatched = 0
        unknown = 0
        partial = 0
        invalid_day = 0
        st_more_than_form = 0
        first_day_invalid_st = 0
        files_done = 0

        print("========== PHASE 6.6 MEET V2 ==========")
        print("Source:", SOURCE_DB)
        print("Output:", OUTPUT_DB)
        print("Files:", len(files))
        print("Source mode: READ ONLY")
        print("Same-day results: EXCLUDED", flush=True)

        for path in files:
            if not re.fullmatch(r"b\d{6}", path.stem):
                continue

            s = path.stem[1:]
            rd = (
                f"20{s[:2]}-{s[2:4]}-{s[4:6]}"
            )
            current_date = date.fromisoformat(rd)

            program, venue_days = parse_program(path)

            for venue, day_no in venue_days.items():
                update_meet_state(
                    states, venue, day_no, current_date
                )

            actual = list(src.execute("""
                SELECT venue_code, race_no, lane, racer_id
                FROM program_entries
                WHERE race_date = ?
                ORDER BY venue_code, race_no, lane
            """, (rd,)))

            # 当日のK結果を参照するのは、
            # この日の全レースの特徴量保存後。
            for venue, race, lane, racer in actual:
                total += 1

                key = (venue, race, lane, racer)
                match = program.get(key)

                if match is None:
                    unmatched += 1
                    meet_day = venue_days.get(venue)
                    form_stats = (
                        0, 0, None, None, None, 0, 0, 0
                    )
                else:
                    meet_day, form = match
                    form_stats = parse_form(meet_day, form)

                if meet_day is None or meet_day < 1:
                    invalid_day += 1

                unknown += form_stats[-1]

                state = states.get(venue)

                if state is None:
                    is_partial = 1
                    st_values = []
                else:
                    is_partial = state["partial"]
                    st_values = state["st"].get(racer, [])

                st_count, mean_st, std_st = calculate_st(
                    st_values
                )

                # 初日は必ず今節STが存在しない
                if meet_day == 1:
                    if st_count:
                        first_day_invalid_st += 1

                    st_count = 0
                    mean_st = None
                    std_st = None

                if st_count > form_stats[0]:
                    if meet_day >= 8:
                        # Bファイルの表示範囲を超えた可能性
                        pass
                    else:
                        st_more_than_form += 1
                        is_partial = 1

                # 完全性が保証できない場合は、
                # 観測済み件数だけ保存し平均値は欠損
                if is_partial:
                    mean_st = None
                    std_st = None

                partial += int(bool(is_partial))

                batch.append((
                    rd, venue, race, lane, racer,
                    meet_day,
                    *form_stats,
                    st_count, mean_st, std_st,
                    int(bool(is_partial))
                ))

                if len(batch) >= BATCH_SIZE:
                    dst.executemany(insert_sql, batch)
                    dst.commit()
                    batch.clear()

            # この日の全レースの特徴量が確定した後、
            # この日の実際のSTを次の日以降の履歴へ追加
            today_racers = defaultdict(set)

            for venue, race, lane, racer in actual:
                today_racers[venue].add(racer)

            for venue, racer, raw_st in load_results_for_date(
                src, rd
            ):
                if (
                    venue not in states
                    or racer not in today_racers[venue]
                ):
                    continue

                st = parse_st(raw_st)

                if st is not None:
                    states[venue]["st"][racer].append(st)

            files_done += 1

            if files_done % 100 == 0:
                print(
                    f"Files: {files_done:,}/{len(files):,}"
                    f" | Entries: {total:,}"
                    f" | Partial: {partial:,}"
                    f" | Unmatched: {unmatched:,}",
                    flush=True
                )

        if batch:
            dst.executemany(insert_sql, batch)
            dst.commit()

        saved = dst.execute(
            f"SELECT COUNT(*) FROM {BUILD}"
        ).fetchone()[0]

        print("\n=== SUMMARY ===")
        print("Processed files:", files_done)
        print("Processed entries:", f"{total:,}")
        print("Stored rows:", f"{saved:,}")
        print("Unmatched B entries:", unmatched)
        print("Invalid meet days:", invalid_day)
        print("Unknown form characters:", unknown)
        print("Partial ST entries:", partial)
        print("ST more than form:", st_more_than_form)
        print("First day invalid ST:", first_day_invalid_st)

        if saved != total:
            raise RuntimeError("Stored row count mismatch")

        if unmatched or invalid_day or unknown:
            raise RuntimeError(
                "Data audit failed. "
                "Existing main table was preserved."
            )

        if st_more_than_form or first_day_invalid_st:
            raise RuntimeError(
                "ST consistency audit failed. "
                "Existing main table was preserved."
            )

        with dst:
            dst.execute(f"DROP TABLE IF EXISTS {TABLE}")
            dst.execute(
                f"ALTER TABLE {BUILD} RENAME TO {TABLE}"
            )

        print("\n=== SAMPLE ===")
        for row in dst.execute(f"""
            SELECT
                race_date, venue_code, race_no,
                lane, racer_id,
                meet_day, meet_records,
                meet_avg_finish,
                meet_st_count,
                meet_avg_st,
                meet_st_partial
            FROM {TABLE}
            ORDER BY race_date DESC,
                     venue_code, race_no, lane
            LIMIT 10
        """):
            print(row)

        print("\nPHASE 6.6 MEET V2 COMPLETE")

    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    main()
