
import argparse
import sqlite3
import math
from pathlib import Path
from datetime import date
from collections import defaultdict, deque

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "boatrace.db"
OUTPUT = ROOT / "data" / "recent_form_features.db"

TABLE = "racer_recent_form_fast"
BUILD = "racer_recent_form_build"
BATCH_SIZE = 10000
WINDOWS = (5, 10, 20)


def number(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def timing(value):
    try:
        x = float(value)
    except (ValueError, TypeError):
        return None

    # Fなどの負値や不自然な値は集計しない
    return x if 0 <= x <= 1.0 else None


def summarize(history, size):
    recent = list(history)[-size:]

    valid = [
        finish for _, finish, _ in recent
        if finish is not None and 1 <= finish <= 6
    ]

    st_values = [
        st for _, _, st in recent if st is not None
    ]

    n = len(valid)
    st_n = len(st_values)

    wins = sum(x == 1 for x in valid)
    top3 = sum(x <= 3 for x in valid)

    mean_st = sum(st_values) / st_n if st_n else None

    std_st = None
    if st_n >= 2:
        std_st = math.sqrt(
            sum((x - mean_st) ** 2 for x in st_values) / st_n
        )

    return (
        len(recent),
        n,
        wins / n if n else None,
        top3 / n if n else None,
        sum(valid) / n if n else None,
        st_n,
        mean_st,
        std_st
    )


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
        required = {
            "program_entries": {"race_date", "venue_code",
                                "race_no", "lane", "racer_id"},
            "race_boat_results_v2": {
                "race_date", "venue_code", "race_no", "lane",
                "racer_id", "finish_position", "start_timing"
            },
            "race_boat_results_supplement": {
                "race_date", "venue_code", "race_no", "boat_no",
                "racer_id", "finish_position", "start_timing"
            }
        }

        for table, cols in required.items():
            actual = {
                row[1]
                for row in src.execute(f"PRAGMA table_info({table})")
            }
            if not cols.issubset(actual):
                raise RuntimeError(
                    f"{table}: missing {sorted(cols - actual)}"
                )

        dst.execute(f"DROP TABLE IF EXISTS {BUILD}")

        fields = [
            "race_date TEXT NOT NULL",
            "venue_code INTEGER NOT NULL",
            "race_no INTEGER NOT NULL",
            "lane INTEGER NOT NULL",
            "racer_id INTEGER NOT NULL",
            "days_since_last_race INTEGER"
        ]

        for w in WINDOWS:
            fields += [
                f"recent_{w}_records INTEGER NOT NULL",
                f"recent_{w}_valid_finishes INTEGER NOT NULL",
                f"recent_{w}_win_rate REAL",
                f"recent_{w}_top3_rate REAL",
                f"recent_{w}_avg_finish REAL",
                f"recent_{w}_valid_st INTEGER NOT NULL",
                f"recent_{w}_avg_st REAL",
                f"recent_{w}_st_std REAL"
            ]

        fields.append(
            "PRIMARY KEY (race_date, venue_code, race_no, lane)"
        )

        dst.execute(
            f"CREATE TABLE {BUILD} ({', '.join(fields)})"
        )
        dst.commit()

        entry_sql = """
            SELECT race_date, venue_code, race_no, lane, racer_id
            FROM program_entries
        """
        params = []

        if args.end:
            entry_sql += " WHERE race_date <= ?"
            params.append(args.end)

        entry_sql += """
            ORDER BY race_date, venue_code, race_no, lane
        """

        result_sql = """
            SELECT *
            FROM (
                SELECT
                    race_date,
                    venue_code,
                    race_no,
                    lane,
                    racer_id,
                    finish_position,
                    start_timing
                FROM race_boat_results_v2

                UNION ALL

                SELECT
                    race_date,
                    venue_code,
                    race_no,
                    boat_no AS lane,
                    racer_id,
                    finish_position,
                    start_timing
                FROM race_boat_results_supplement
            )
            ORDER BY race_date, venue_code, race_no, lane
        """

        entries = src.execute(entry_sql, params)
        results = src.execute(result_sql)
        next_result = results.fetchone()

        # 各選手の直近20走だけを保持
        history = defaultdict(lambda: deque(maxlen=20))
        last_date = {}

        column_count = 6 + len(WINDOWS) * 8
        placeholders = ", ".join(["?"] * column_count)

        insert_sql = (
            f"INSERT INTO {BUILD} VALUES ({placeholders})"
        )

        batch = []
        total = 0
        with_history = 0
        current_day = None
        pending = []

        def process_day(day, day_entries):
            nonlocal next_result, total, with_history, batch

            today = date.fromisoformat(day).toordinal()

            # 必ず前日以前の結果だけ反映
            while next_result is not None and next_result[0] < day:
                rd, venue, race, lane, racer, finish, st = next_result
                next_result = results.fetchone()

                if racer is None:
                    continue

                racer = int(racer)
                ordinal = date.fromisoformat(rd).toordinal()

                history[racer].append((
                    ordinal,
                    number(finish),
                    timing(st)
                ))

                last_date[racer] = ordinal

            for rd, venue, race, lane, racer in day_entries:
                if racer is None or lane not in (1, 2, 3, 4, 5, 6):
                    raise ValueError(
                        f"Invalid entry: {rd} {venue} "
                        f"{race} {lane} {racer}"
                    )

                racer = int(racer)
                h = history.get(racer, ())

                days = (
                    today - last_date[racer]
                    if racer in last_date else None
                )

                row = [
                    rd, venue, race, lane, racer, days
                ]

                for w in WINDOWS:
                    row.extend(summarize(h, w))

                batch.append(tuple(row))
                total += 1
                with_history += int(bool(h))

                if len(batch) >= BATCH_SIZE:
                    dst.executemany(insert_sql, batch)
                    dst.commit()
                    batch.clear()

                if total % 100000 == 0:
                    print(
                        f"Processed: {total:,}"
                        f" | With history: {with_history:,}",
                        flush=True
                    )

        print("========== PHASE 6.5 RECENT FORM ==========")
        print("Source:", SOURCE)
        print("Output:", OUTPUT)
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

        # 全件完成後に本番テーブルへ切り替え
        with dst:
            dst.execute(f"DROP TABLE IF EXISTS {TABLE}")
            dst.execute(
                f"ALTER TABLE {BUILD} RENAME TO {TABLE}"
            )

        print("\n=== SUMMARY ===")
        print("Processed entries:", f"{total:,}")
        print("Stored rows:", f"{saved:,}")
        print("Entries with previous history:", f"{with_history:,}")
        print(
            "History coverage:",
            f"{with_history / total * 100:.2f}%"
            if total else "N/A"
        )

        print("\n=== SAMPLE ===")
        for row in dst.execute(f"""
            SELECT
                race_date, venue_code, race_no,
                lane, racer_id,
                recent_10_records,
                recent_10_valid_finishes,
                recent_10_win_rate,
                recent_10_avg_st,
                recent_10_st_std,
                days_since_last_race
            FROM {TABLE}
            ORDER BY race_date DESC,
                     venue_code, race_no, lane
            LIMIT 10
        """):
            print(row)

        print("\nPHASE 6.5 RECENT FORM COMPLETE")

    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    main()
