
import argparse
import sqlite3
from pathlib import Path
from datetime import date
from collections import defaultdict, deque

ROOT = Path(__file__).resolve().parent.parent
SRC_DB = ROOT / "data" / "boatrace.db"
OUT_DB = ROOT / "data" / "lane_features.db"

TABLE = "racer_lane_features_fast"
BUILD = "racer_lane_features_build"
WINDOW = 365
BATCH_SIZE = 10000


def empty():
    # starts, wins, top3, known, same, course_sum
    return [0, 0, 0, 0, 0, 0]


def apply(stats, key, event, sign):
    _, _, _, _, _, _, finish, course = event
    s = stats[key]

    s[0] += sign
    s[1] += sign * (finish == 1)
    s[2] += sign * (1 <= finish <= 3)

    if course is not None:
        s[3] += sign
        s[4] += sign * (course == event[5])
        s[5] += sign * course

    if any(v < 0 for v in s):
        raise RuntimeError(f"Negative stats: {key} {s}")


def ratio(a, b):
    return a / b if b else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    if args.end:
        date.fromisoformat(args.end)

    if not SRC_DB.exists():
        raise FileNotFoundError(SRC_DB)

    src = sqlite3.connect(
        SRC_DB.as_uri() + "?mode=ro",
        uri=True
    )
    dst = sqlite3.connect(OUT_DB, timeout=60)

    try:
        # 既存の完成済みテーブルは保持する。
        # 再実行時は作成途中のテーブルだけ再作成する。
        dst.execute(f"DROP TABLE IF EXISTS {BUILD}")

        dst.execute(f"""
            CREATE TABLE {BUILD} (
                race_date TEXT NOT NULL,
                venue_code INTEGER NOT NULL,
                race_no INTEGER NOT NULL,
                lane INTEGER NOT NULL,
                racer_id INTEGER NOT NULL,

                lane_starts_365d INTEGER NOT NULL,
                lane_win_rate_365d REAL,
                lane_top3_rate_365d REAL,

                lane_course_known_365d INTEGER NOT NULL,
                lane_same_course_rate_365d REAL,
                lane_course_change_rate_365d REAL,
                lane_avg_course_365d REAL,

                racer_course_change_rate_365d REAL,

                venue_lane_starts_365d INTEGER NOT NULL,
                venue_lane_win_rate_365d REAL,
                venue_lane_top3_rate_365d REAL,

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

        # v2のlaneとsupplementのboat_noが艇番。
        # v2のboat_noはボート機材番号なので使用しない。
        result_sql = """
            SELECT race_date, venue_code, race_no,
                   racer_id, lane, finish_position,
                   entry_course
            FROM race_boat_results_v2

            UNION ALL

            SELECT race_date, venue_code, race_no,
                   racer_id, boat_no AS lane,
                   finish_position, entry_course
            FROM race_boat_results_supplement

            ORDER BY race_date, venue_code, race_no, lane
        """

        entries = src.execute(entry_sql, params)
        results = src.execute(result_sql)
        next_result = results.fetchone()

        by_lane = defaultdict(empty)
        by_venue_lane = defaultdict(empty)
        by_racer = defaultdict(empty)
        active = deque()

        insert_sql = f"""
            INSERT INTO {BUILD} VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?,
                ?, ?, ?
            )
        """

        total = 0
        historical_count = 0
        batch = []
        current_day = None
        pending = []
        invalid_result_lanes = 0
        unknown_courses = 0

        def update(event, sign):
            _, racer, venue, _, _, lane, _, _ = event
            apply(by_lane, (racer, lane), event, sign)
            apply(
                by_venue_lane,
                (racer, venue, lane), event, sign
            )
            apply(by_racer, racer, event, sign)

        def process_day(day, day_entries):
            nonlocal next_result, total
            nonlocal historical_count, batch
            nonlocal invalid_result_lanes, unknown_courses

            ordinal = date.fromisoformat(day).toordinal()
            earliest = ordinal - WINDOW

            # 365日より古い結果を取り除く。
            while active and active[0][0] < earliest:
                update(active.popleft(), -1)

            # 予測当日の結果を絶対に使わない。
            while next_result is not None and next_result[0] < day:
                (
                    rd, venue, race,
                    racer_id, lane, finish, course
                ) = next_result

                next_result = results.fetchone()
                result_day = date.fromisoformat(rd).toordinal()

                if result_day < earliest or racer_id is None:
                    continue

                try:
                    lane = int(lane)
                    venue = int(venue)
                except (TypeError, ValueError):
                    invalid_result_lanes += 1
                    continue

                if lane not in range(1, 7):
                    invalid_result_lanes += 1
                    continue

                try:
                    finish = int(finish)
                except (TypeError, ValueError):
                    finish = 0

                try:
                    course = int(course)
                except (TypeError, ValueError):
                    course = None

                if course not in range(1, 7):
                    course = None
                    unknown_courses += 1

                event = (
                    result_day,
                    int(racer_id),
                    venue,
                    int(race),
                    rd,
                    lane,
                    finish,
                    course
                )

                active.append(event)
                update(event, +1)

            for race_date, venue, race, lane, racer_id in day_entries:
                if racer_id is None or lane not in range(1, 7):
                    raise ValueError(
                        f"Invalid entry: {race_date} "
                        f"{venue} {race} {lane} {racer_id}"
                    )

                lane_stat = by_lane.get(
                    (racer_id, lane), empty()
                )
                venue_stat = by_venue_lane.get(
                    (racer_id, venue, lane), empty()
                )
                racer_stat = by_racer.get(
                    racer_id, empty()
                )

                n, wins, top3, known, same, course_sum = lane_stat
                vn, vw, vt, _, _, _ = venue_stat
                _, _, _, rk, rs, _ = racer_stat

                historical_count += int(n > 0)

                batch.append((
                    race_date, venue, race, lane, racer_id,

                    n,
                    ratio(wins, n),
                    ratio(top3, n),

                    known,
                    ratio(same, known),
                    ratio(known - same, known),
                    ratio(course_sum, known),

                    ratio(rk - rs, rk),

                    vn,
                    ratio(vw, vn),
                    ratio(vt, vn)
                ))

                total += 1

                if len(batch) >= BATCH_SIZE:
                    dst.executemany(insert_sql, batch)
                    dst.commit()
                    batch.clear()

                if total % 100000 == 0:
                    print(
                        f"Processed: {total:,} "
                        f"| lane history: {historical_count:,}",
                        flush=True
                    )

        print("========== PHASE 6.2 LANE FAST ==========")
        print("Source:", SRC_DB)
        print("Output:", OUT_DB)
        print("Window:", WINDOW, "days")
        print("End date:", args.end or "ALL")
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

        # 全件成功してから本番テーブルへ切り替え。
        with dst:
            dst.execute(f"DROP TABLE IF EXISTS {TABLE}")
            dst.execute(
                f"ALTER TABLE {BUILD} RENAME TO {TABLE}"
            )

        print("\n=== SUMMARY ===")
        print("Processed entries:", f"{total:,}")
        print("Stored rows:", f"{saved:,}")
        print("Entries with lane history:", f"{historical_count:,}")
        print("Unknown historical courses:", unknown_courses)
        print("Invalid historical lanes:", invalid_result_lanes)
        print(
            "Lane history coverage:",
            f"{historical_count / total * 100:.2f}%"
            if total else "N/A"
        )

        print("\n=== SAMPLE ===")
        for row in dst.execute(f"""
            SELECT race_date, venue_code, race_no,
                   lane, racer_id,
                   lane_starts_365d,
                   lane_win_rate_365d,
                   lane_course_change_rate_365d,
                   venue_lane_starts_365d
            FROM {TABLE}
            ORDER BY race_date DESC,
                     venue_code, race_no, lane
            LIMIT 10
        """):
            print(row)

        print("\nPHASE 6.2 LANE COMPLETE")

    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    main()
