
import argparse
import sqlite3
from pathlib import Path
from datetime import date
from collections import defaultdict, deque

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "boatrace.db"
OUTPUT = ROOT / "data" / "motor_features_v2.db"

TABLE = "equipment_features_v2"
BUILD = "equipment_features_v2_build"
BATCH = 10000


def q(s):
    return '"' + s.replace('"', '""') + '"'


def number(value):
    try:
        n = int(value)
        return n if n > 0 else None
    except (ValueError, TypeError):
        return None


def new_stats():
    return [0, 0]


def update(stats, key, top2, sign):
    row = stats[key]
    row[0] += sign
    row[1] += sign * top2

    if min(row) < 0:
        raise RuntimeError(
            f"Negative stats: {key}"
        )


def get_stats(stats, venue, equipment):
    if equipment is None:
        return 0, None

    n, top2 = stats.get(
        (venue, equipment), (0, 0)
    )

    return n, top2 / n if n else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    if args.end:
        date.fromisoformat(args.end)

    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    src = sqlite3.connect(
        SOURCE.resolve().as_uri() + "?mode=ro",
        uri=True,
        timeout=120
    )
    dst = sqlite3.connect(
        OUTPUT, timeout=120
    )

    try:
        print("=" * 65)
        print("PHASE 8.3-B MOTOR FEATURES V2")
        print("=" * 65)
        print("Source: READ ONLY")
        print("Output:", OUTPUT)
        print("Existing F3: NOT MODIFIED")
        print("Final test: NOT EVALUATED", flush=True)

        # 元DBに交換部品情報があるか確認
        print("\n=== SOURCE COLUMN AUDIT ===")

        keywords = (
            "change", "replace", "repair",
            "parts", "piston", "ring",
            "carb", "maintenance",
            "exchange", "交換", "整備"
        )

        found = []

        for (table,) in src.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
        """):
            for row in src.execute(
                f"PRAGMA table_info({q(table)})"
            ):
                col = row[1]

                if any(
                    word in col.lower()
                    for word in keywords
                ):
                    found.append((table, col))

        for table, col in found:
            print(table, col)

        if not found:
            print(
                "No obvious replacement columns found."
            )

        # 新しい特徴量テーブル
        dst.execute(f"DROP TABLE IF EXISTS {BUILD}")

        dst.execute(f"""
            CREATE TABLE {BUILD} (
                race_date TEXT NOT NULL,
                venue_code INTEGER NOT NULL,
                race_no INTEGER NOT NULL,
                lane INTEGER NOT NULL,
                racer_id INTEGER NOT NULL,

                motor_starts_30d INTEGER NOT NULL,
                motor_top2_rate_30d REAL,
                motor_starts_90d INTEGER NOT NULL,
                motor_top2_rate_90d REAL,
                motor_top2_trend REAL,

                boat_starts_30d INTEGER NOT NULL,
                boat_top2_rate_30d REAL,
                boat_starts_90d INTEGER NOT NULL,
                boat_top2_rate_90d REAL,
                boat_top2_trend REAL,

                PRIMARY KEY (
                    race_date, venue_code,
                    race_no, lane
                )
            )
        """)

        # 部品交換イベントの保存先を用意
        # 未取得の情報を「交換なし」として埋めない
        dst.execute("""
            CREATE TABLE IF NOT EXISTS
            motor_replacement_events (
                race_date TEXT NOT NULL,
                venue_code INTEGER NOT NULL,
                race_no INTEGER NOT NULL,
                lane INTEGER NOT NULL,
                motor_no INTEGER,
                component TEXT NOT NULL,
                quantity INTEGER,
                observed_at TEXT,
                source_url TEXT,
                PRIMARY KEY (
                    race_date, venue_code,
                    race_no, lane, component
                )
            )
        """)

        dst.commit()

        entry_sql = """
            SELECT
                race_date, venue_code,
                race_no, lane, racer_id,
                motor_no, boat_no
            FROM program_entries
        """

        params = []

        if args.end:
            entry_sql += " WHERE race_date <= ?"
            params.append(args.end)

        entry_sql += """
            ORDER BY
                race_date, venue_code,
                race_no, lane
        """

        # 通常結果を優先し、補完結果は
        # 通常結果がない場合に限り使用する
        result_sql = """
            SELECT race_date, venue_code,
                   motor_no, boat_no,
                   finish_position
            FROM (
                SELECT
                    r.race_date,
                    r.venue_code,
                    p.motor_no,
                    p.boat_no,
                    r.finish_position
                FROM race_boat_results_v2 r
                JOIN program_entries p
                  ON p.race_date = r.race_date
                 AND p.venue_code = r.venue_code
                 AND p.race_no = r.race_no
                 AND p.lane = r.lane
                 AND p.racer_id = r.racer_id

                UNION ALL

                SELECT
                    r.race_date,
                    r.venue_code,
                    p.motor_no,
                    p.boat_no,
                    r.finish_position
                FROM race_boat_results_supplement r
                JOIN program_entries p
                  ON p.race_date = r.race_date
                 AND p.venue_code = r.venue_code
                 AND p.race_no = r.race_no
                 AND p.lane = r.boat_no
                 AND p.racer_id = r.racer_id
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM race_boat_results_v2 x
                    WHERE x.race_date = r.race_date
                      AND x.venue_code = r.venue_code
                      AND x.race_no = r.race_no
                      AND x.lane = r.boat_no
                )
            )
            ORDER BY race_date, venue_code
        """

        entries = src.execute(entry_sql, params)
        results = src.execute(result_sql)
        next_result = results.fetchone()

        m30 = defaultdict(new_stats)
        m90 = defaultdict(new_stats)
        b30 = defaultdict(new_stats)
        b90 = defaultdict(new_stats)

        h30 = deque()
        h90 = deque()

        insert_sql = f"""
            INSERT INTO {BUILD}
            VALUES ({",".join(["?"] * 15)})
        """

        batch = []
        pending = []
        current_day = None

        total = 0
        valid_events = 0
        skipped_invalid = 0

        def apply_event(event, sign, days):
            _, venue, motor, boat, top2 = event

            ms = m30 if days == 30 else m90
            bs = b30 if days == 30 else b90

            if motor is not None:
                update(
                    ms, (venue, motor), top2, sign
                )

            if boat is not None:
                update(
                    bs, (venue, boat), top2, sign
                )

        def process_day(day, day_entries):
            nonlocal next_result, total
            nonlocal valid_events, skipped_invalid

            today = date.fromisoformat(day).toordinal()

            # 厳密に過去30日・90日だけ
            cutoff30 = today - 30
            cutoff90 = today - 90

            while h30 and h30[0][0] < cutoff30:
                apply_event(h30.popleft(), -1, 30)

            while h90 and h90[0][0] < cutoff90:
                apply_event(h90.popleft(), -1, 90)

            # 当日結果を特徴量に入れない
            while (
                next_result is not None
                and next_result[0] < day
            ):
                rd, venue, motor, boat, finish = (
                    next_result
                )
                next_result = results.fetchone()

                ordinal = date.fromisoformat(
                    rd
                ).toordinal()

                if ordinal < cutoff90:
                    continue

                pos = number(finish)

                # 着順不明は分母に入れない
                if pos is None or pos > 6:
                    skipped_invalid += 1
                    continue

                event = (
                    ordinal,
                    venue,
                    number(motor),
                    number(boat),
                    int(pos <= 2)
                )

                h90.append(event)
                apply_event(event, +1, 90)

                if ordinal >= cutoff30:
                    h30.append(event)
                    apply_event(event, +1, 30)

                valid_events += 1

            for (
                rd, venue, race, lane,
                racer, motor, boat
            ) in day_entries:

                motor = number(motor)
                boat = number(boat)

                mn30, mr30 = get_stats(
                    m30, venue, motor
                )
                mn90, mr90 = get_stats(
                    m90, venue, motor
                )

                bn30, br30 = get_stats(
                    b30, venue, boat
                )
                bn90, br90 = get_stats(
                    b90, venue, boat
                )

                mt = (
                    mr30 - mr90
                    if mr30 is not None
                    and mr90 is not None
                    else None
                )

                bt = (
                    br30 - br90
                    if br30 is not None
                    and br90 is not None
                    else None
                )

                batch.append((
                    rd, venue, race, lane, racer,
                    mn30, mr30, mn90, mr90, mt,
                    bn30, br30, bn90, br90, bt
                ))

                total += 1

                if len(batch) >= BATCH:
                    dst.executemany(
                        insert_sql, batch
                    )
                    dst.commit()
                    batch.clear()

                if total % 100000 == 0:
                    print(
                        f"Processed: {total:,}",
                        flush=True
                    )

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
                f"Row mismatch {saved} vs {total}"
            )

        with dst:
            dst.execute(
                f"DROP TABLE IF EXISTS {TABLE}"
            )
            dst.execute(
                f"ALTER TABLE {BUILD} RENAME TO {TABLE}"
            )

        print("\n=== SUMMARY ===")
        print("Stored rows:", f"{saved:,}")
        print("Valid historical events:",
              f"{valid_events:,}")
        print("Skipped invalid finishes:",
              f"{skipped_invalid:,}")

        for row in dst.execute(f"""
            SELECT race_date, venue_code, race_no,
                   lane, motor_top2_rate_90d,
                   boat_top2_rate_90d
            FROM {TABLE}
            ORDER BY race_date DESC
            LIMIT 10
        """):
            print(row)

        print("\nPHASE 8.3-B COMPLETE")

    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    main()
