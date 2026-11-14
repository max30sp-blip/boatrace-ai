
import argparse
import sqlite3
from pathlib import Path
from datetime import date
from collections import defaultdict, deque

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "boatrace.db"
OUTPUT = ROOT / "data" / "venue_detail_features.db"

TABLE = "venue_detail_features_fast"
BUILD = "venue_detail_features_build"

WINDOW = 365
BATCH_SIZE = 10000


def rate(a, b):
    return a / b if b else None


def empty():
    # 出走数・1着数・3着以内数
    return [0, 0, 0]


def change(stats, key, win, top3, sign):
    item = stats[key]
    item[0] += sign
    item[1] += sign * win
    item[2] += sign * top3

    if any(x < 0 for x in item):
        raise RuntimeError(f"Negative statistics: {key}")


def get_rates(stats, key):
    n, wins, top3 = stats.get(key, (0, 0, 0))
    return (n, rate(wins, n), rate(top3, n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    if args.end:
        date.fromisoformat(args.end)

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    src = sqlite3.connect(
        SOURCE.as_uri() + "?mode=ro",
        uri=True,
        timeout=60
    )
    dst = sqlite3.connect(OUTPUT, timeout=60)

    try:
        # 完成済みテーブルは最後まで保持する
        dst.execute(f"DROP TABLE IF EXISTS {BUILD}")

        dst.execute(f"""
            CREATE TABLE {BUILD} (
                race_date TEXT NOT NULL,
                venue_code INTEGER NOT NULL,
                race_no INTEGER NOT NULL,
                lane INTEGER NOT NULL,
                racer_id INTEGER NOT NULL,

                venue_lane_starts_365d INTEGER NOT NULL,
                venue_lane_win_rate_365d REAL,
                venue_lane_top3_rate_365d REAL,

                venue_course_starts_365d INTEGER NOT NULL,
                venue_course_win_rate_365d REAL,
                venue_course_top3_rate_365d REAL,

                national_course_starts_365d INTEGER NOT NULL,
                national_course_win_rate_365d REAL,

                venue_course_vs_national REAL,
                venue_races_365d INTEGER NOT NULL,

                PRIMARY KEY (
                    race_date, venue_code, race_no, lane
                )
            )
        """)
        dst.commit()

        entry_sql = """
            SELECT race_date, venue_code, race_no,
                   lane, racer_id
            FROM program_entries
        """
        params = []

        if args.end:
            entry_sql += " WHERE race_date <= ?"
            params.append(args.end)

        entry_sql += """
            ORDER BY race_date, venue_code, race_no, lane
        """

        # 通常結果のlaneは艇番。
        # 補完結果のboat_noも艇番。
        # 進入コースはentry_course。
        result_sql = """
            SELECT *
            FROM (
                SELECT
                    r.race_date AS race_date,
                    r.venue_code AS venue_code,
                    r.race_no AS race_no,
                    r.lane AS lane,
                    r.entry_course AS entry_course,
                    r.finish_position AS finish_position
                FROM race_boat_results_v2 r

                UNION ALL

                SELECT
                    s.race_date AS race_date,
                    s.venue_code AS venue_code,
                    s.race_no AS race_no,
                    s.boat_no AS lane,
                    s.entry_course AS entry_course,
                    s.finish_position AS finish_position
                FROM race_boat_results_supplement s
            )
            ORDER BY race_date, venue_code, race_no, lane
        """

        entries = src.execute(entry_sql, params)
        results = src.execute(result_sql)
        next_result = results.fetchone()

        by_venue_lane = defaultdict(empty)
        by_venue_course = defaultdict(empty)
        by_national_course = defaultdict(empty)

        # 過去365日以内のレース数
        venue_race_count = defaultdict(int)

        # 日付順の履歴
        history = deque()
        race_history = deque()

        insert_sql = f"""
            INSERT INTO {BUILD} VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?,
                ?
            )
        """

        batch = []
        total = 0
        current_day = None
        pending = []
        unknown_courses = 0
        invalid_lanes = 0
        excluded_finishes = 0

        def update(event, sign):
            (
                _, venue, lane, course,
                win, top3
            ) = event

            change(
                by_venue_lane,
                (venue, lane),
                win, top3, sign
            )

            if course is not None:
                change(
                    by_venue_course,
                    (venue, course),
                    win, top3, sign
                )

                change(
                    by_national_course,
                    course,
                    win, top3, sign
                )

        def process_day(day, day_entries):
            nonlocal next_result
            nonlocal total, batch
            nonlocal unknown_courses
            nonlocal invalid_lanes
            nonlocal excluded_finishes

            today = date.fromisoformat(day).toordinal()
            cutoff = today - WINDOW

            # 365日より前の結果を除外
            while history and history[0][0] < cutoff:
                update(history.popleft(), -1)

            while (
                race_history
                and race_history[0][0] < cutoff
            ):
                _, venue = race_history.popleft()
                venue_race_count[venue] -= 1

            # 対象日より前の結果だけを追加する
            last_race_key = None

            while (
                next_result is not None
                and next_result[0] < day
            ):
                (
                    rd, venue, race,
                    lane, course, finish
                ) = next_result

                next_result = results.fetchone()

                ordinal = date.fromisoformat(rd).toordinal()
                if ordinal < cutoff:
                    continue

                try:
                    venue = int(venue)
                    race = int(race)
                    lane = int(lane)
                except (TypeError, ValueError):
                    invalid_lanes += 1
                    continue

                if lane not in range(1, 7):
                    invalid_lanes += 1
                    continue

                race_key = (rd, venue, race)

                if race_key != last_race_key:
                    venue_race_count[venue] += 1
                    race_history.append((ordinal, venue))
                    last_race_key = race_key

                try:
                    course = int(course)
                except (TypeError, ValueError):
                    course = None

                if course not in range(1, 7):
                    course = None
                    unknown_courses += 1

                try:
                    finish = int(finish)
                except (TypeError, ValueError):
                    finish = None

                # 着順不明・失格等は勝率計算から除外
                # 1～6着がある場合だけ集計する
                if finish not in range(1, 7):
                    excluded_finishes += 1
                    continue

                event = (
                    ordinal,
                    venue,
                    lane,
                    course,
                    int(finish == 1),
                    int(finish <= 3)
                )

                history.append(event)
                update(event, +1)

            for rd, venue, race, lane, racer_id in day_entries:
                if (
                    racer_id is None
                    or lane not in range(1, 7)
                ):
                    raise ValueError(
                        f"Invalid entry: {rd} "
                        f"{venue}-{race}-{lane}"
                    )

                vl = get_rates(
                    by_venue_lane,
                    (venue, lane)
                )
                vc = get_rates(
                    by_venue_course,
                    (venue, lane)
                )
                nc = get_rates(
                    by_national_course,
                    lane
                )

                difference = (
                    vc[1] - nc[1]
                    if vc[1] is not None
                    and nc[1] is not None
                    else None
                )

                batch.append((
                    rd, venue, race, lane, racer_id,
                    *vl,
                    *vc,
                    nc[0],
                    nc[1],
                    difference,
                    venue_race_count[venue]
                ))

                total += 1

                if len(batch) >= BATCH_SIZE:
                    dst.executemany(insert_sql, batch)
                    dst.commit()
                    batch.clear()

                if total % 100000 == 0:
                    print(
                        f"Processed entries: {total:,}",
                        flush=True
                    )

        print("========== PHASE 6.4 VENUE DETAIL ==========")
        print("Source:", SOURCE)
        print("Output:", OUTPUT)
        print("Window:", WINDOW, "days")
        print("End:", args.end or "ALL")
        print("Source mode: READ ONLY", flush=True)

        for entry in entries:
            day = entry[0]

            if current_day is None:
                current_day = day

            if day != current_day:
                process_day(current_day, pending)
                pending = []
                current_day = day

            pending.append(entry)

        if pending:
            process_day(current_day, pending)

        if batch:
            dst.executemany(insert_sql, batch)
            dst.commit()

        saved = dst.execute(
            f"SELECT COUNT(*) FROM {BUILD}"
        ).fetchone()[0]

        if saved != total:
            raise RuntimeError(
                f"Count mismatch: {saved} != {total}"
            )

        # 完成後のみ本番テーブルに切り替え
        with dst:
            dst.execute(f"DROP TABLE IF EXISTS {TABLE}")
            dst.execute(
                f"ALTER TABLE {BUILD} RENAME TO {TABLE}"
            )

        print("\n=== SUMMARY ===")
        print("Processed entries:", f"{total:,}")
        print("Stored rows:", f"{saved:,}")
        print("Unknown historical courses:", unknown_courses)
        print("Invalid historical lanes:", invalid_lanes)
        print("Excluded special finishes:", excluded_finishes)

        print("\n=== SAMPLE ===")
        for row in dst.execute(f"""
            SELECT
                race_date, venue_code, race_no,
                lane, racer_id,
                venue_lane_starts_365d,
                venue_lane_win_rate_365d,
                venue_course_starts_365d,
                venue_course_win_rate_365d,
                national_course_win_rate_365d,
                venue_course_vs_national
            FROM {TABLE}
            ORDER BY race_date DESC,
                     venue_code, race_no, lane
            LIMIT 10
        """):
            print(row)

        print("\nPHASE 6.4 VENUE COMPLETE")

    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    main()
