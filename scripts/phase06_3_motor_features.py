
import argparse
import sqlite3
from pathlib import Path
from datetime import date
from collections import defaultdict, deque

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DB = ROOT / "data" / "boatrace.db"
OUTPUT_DB = ROOT / "data" / "motor_features.db"

TABLE = "equipment_features_fast"
BUILD = "equipment_features_build"
BATCH = 10000


def empty():
    return [0, 0, 0]


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def get_stats(stats, key):
    n, wins, top3 = stats.get(key, (0, 0, 0))
    return n, rate(wins, n), rate(top3, n)


def update(stats, key, win, top3, sign):
    values = stats[key]
    values[0] += sign
    values[1] += sign * win
    values[2] += sign * top3

    if any(value < 0 for value in values):
        raise RuntimeError(f"Negative stats: {key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    if args.end:
        date.fromisoformat(args.end)

    if not SOURCE_DB.exists():
        raise FileNotFoundError(SOURCE_DB)

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

                motor_starts_30d INTEGER NOT NULL,
                motor_win_rate_30d REAL,
                motor_top3_rate_30d REAL,

                motor_starts_90d INTEGER NOT NULL,
                motor_win_rate_90d REAL,
                motor_top3_rate_90d REAL,
                motor_win_trend REAL,

                boat_starts_30d INTEGER NOT NULL,
                boat_win_rate_30d REAL,
                boat_top3_rate_30d REAL,

                boat_starts_90d INTEGER NOT NULL,
                boat_win_rate_90d REAL,
                boat_top3_rate_90d REAL,
                boat_win_trend REAL,

                PRIMARY KEY (
                    race_date, venue_code, race_no, lane
                )
            )
        """)
        dst.commit()

        entry_sql = """
            SELECT race_date, venue_code, race_no,
                   lane, racer_id, motor_no, boat_no
            FROM program_entries
        """

        params = []
        if args.end:
            entry_sql += " WHERE race_date <= ?"
            params.append(args.end)

        entry_sql += """
            ORDER BY race_date, venue_code, race_no, lane
        """

        # 過去のレース結果と、当時の出走表の機材を結合。
        # 結果テーブルのboat_noを機材番号として扱わない。
        
        result_sql = """
            SELECT *
            FROM (
                SELECT
                    r.race_date AS race_date,
                    r.venue_code AS venue_code,
                    p.motor_no AS motor_no,
                    p.boat_no AS boat_no,
                    r.finish_position AS finish_position
                FROM race_boat_results_v2 r
                JOIN program_entries p
                  ON p.race_date = r.race_date
                 AND p.venue_code = r.venue_code
                 AND p.race_no = r.race_no
                 AND p.lane = r.lane
                 AND p.racer_id = r.racer_id

                UNION ALL

                SELECT
                    r.race_date AS race_date,
                    r.venue_code AS venue_code,
                    p.motor_no AS motor_no,
                    p.boat_no AS boat_no,
                    r.finish_position AS finish_position
                FROM race_boat_results_supplement r
                JOIN program_entries p
                  ON p.race_date = r.race_date
                 AND p.venue_code = r.venue_code
                 AND p.race_no = r.race_no
                 AND p.lane = r.boat_no
                 AND p.racer_id = r.racer_id
            )
            ORDER BY race_date, venue_code
            """


        entries = src.execute(entry_sql, params)
        results = src.execute(result_sql)
        next_result = results.fetchone()

        # 30日・90日について独立した集計を保持
        motor30 = defaultdict(empty)
        motor90 = defaultdict(empty)
        boat30 = defaultdict(empty)
        boat90 = defaultdict(empty)

        history30 = deque()
        history90 = deque()

        insert_sql = f"""
            INSERT INTO {BUILD} VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?
            )
        """

        total = 0
        batch = []
        current_day = None
        pending = []

        def valid_number(value):
            try:
                number = int(value)
                return number if number > 0 else None
            except (TypeError, ValueError):
                return None

        def apply_event(event, stats_m, stats_b, sign):
            _, venue, motor, boat, win, top3 = event

            if motor is not None:
                update(
                    stats_m,
                    (venue, motor),
                    win, top3, sign
                )

            if boat is not None:
                update(
                    stats_b,
                    (venue, boat),
                    win, top3, sign
                )

        def process_day(day, day_entries):
            nonlocal next_result, total, batch

            today = date.fromisoformat(day).toordinal()
            cutoff30 = today - 30
            cutoff90 = today - 90

            while history30 and history30[0][0] < cutoff30:
                event = history30.popleft()
                apply_event(event, motor30, boat30, -1)

            while history90 and history90[0][0] < cutoff90:
                event = history90.popleft()
                apply_event(event, motor90, boat90, -1)

            # 当日より前の結果だけを反映
            while next_result and next_result[0] < day:
                rd, venue, motor, boat, finish = next_result
                next_result = results.fetchone()

                ordinal = date.fromisoformat(rd).toordinal()

                if ordinal < cutoff90:
                    continue

                try:
                    position = int(finish)
                except (TypeError, ValueError):
                    position = 0

                event = (
                    ordinal,
                    int(venue),
                    valid_number(motor),
                    valid_number(boat),
                    int(position == 1),
                    int(1 <= position <= 3)
                )

                history90.append(event)
                apply_event(event, motor90, boat90, +1)

                if ordinal >= cutoff30:
                    history30.append(event)
                    apply_event(event, motor30, boat30, +1)

            for (
                race_date, venue, race, lane,
                racer_id, motor, boat
            ) in day_entries:

                if racer_id is None or lane not in range(1, 7):
                    raise ValueError(
                        f"Invalid entry: {race_date}, "
                        f"{venue}, {race}, {lane}"
                    )

                motor = valid_number(motor)
                boat = valid_number(boat)

                # 機材不明の場合、履歴件数0と欠損率を保存
                m30 = (
                    get_stats(motor30, (venue, motor))
                    if motor is not None
                    else (0, None, None)
                )
                m90 = (
                    get_stats(motor90, (venue, motor))
                    if motor is not None
                    else (0, None, None)
                )
                b30 = (
                    get_stats(boat30, (venue, boat))
                    if boat is not None
                    else (0, None, None)
                )
                b90 = (
                    get_stats(boat90, (venue, boat))
                    if boat is not None
                    else (0, None, None)
                )

                motor_trend = (
                    m30[1] - m90[1]
                    if m30[1] is not None
                    and m90[1] is not None
                    else None
                )

                boat_trend = (
                    b30[1] - b90[1]
                    if b30[1] is not None
                    and b90[1] is not None
                    else None
                )

                batch.append((
                    race_date, venue, race, lane, racer_id,
                    *m30, *m90, motor_trend,
                    *b30, *b90, boat_trend
                ))

                total += 1

                if len(batch) >= BATCH:
                    dst.executemany(insert_sql, batch)
                    dst.commit()
                    batch.clear()

                if total % 100000 == 0:
                    print(
                        f"Processed entries: {total:,}",
                        flush=True
                    )

        print("========== PHASE 6.3 MOTOR FAST ==========")
        print("Source:", SOURCE_DB)
        print("Output:", OUTPUT_DB)
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
                f"Row mismatch: {saved} != {total}"
            )

        with dst:
            dst.execute(f"DROP TABLE IF EXISTS {TABLE}")
            dst.execute(
                f"ALTER TABLE {BUILD} RENAME TO {TABLE}"
            )

        print("\n=== SUMMARY ===")
        print("Processed entries:", f"{total:,}")
        print("Stored rows:", f"{saved:,}")

        print("\n=== SAMPLE ===")
        for row in dst.execute(f"""
            SELECT
                race_date, venue_code, race_no,
                lane, racer_id,
                motor_starts_30d, motor_win_rate_30d,
                motor_starts_90d, motor_win_rate_90d,
                boat_starts_30d, boat_win_rate_30d
            FROM {TABLE}
            ORDER BY race_date DESC,
                     venue_code, race_no, lane
            LIMIT 10
        """):
            print(row)

        print("\nPHASE 6.3 MOTOR COMPLETE")

    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    main()
