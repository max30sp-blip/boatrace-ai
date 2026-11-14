
import argparse
import sqlite3
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DB = ROOT / "data" / "boatrace.db"
FEATURE_DB = ROOT / "data" / "features_fast.db"

WINDOW = 365
BATCH_SIZE = 5000

OUTPUT_TABLE = "racer_course_features_fast"
BUILD_TABLE = "racer_course_features_fast_build"


def get_columns(conn, table):
    return {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})")
    }


def validate_sources(conn):
    expected = {
        "program_entries": {
            "race_date", "venue_code", "race_no",
            "lane", "racer_id"
        },
        "race_boat_results_v2": {
            "race_date", "venue_code", "race_no",
            "lane", "racer_id", "entry_course",
            "finish_position"
        },
        "race_boat_results_supplement": {
            "race_date", "venue_code", "race_no",
            "boat_no", "racer_id", "entry_course",
            "finish_position"
        }
    }

    for table, required in expected.items():
        actual = get_columns(conn, table)
        missing = required - actual
        if missing:
            raise ValueError(
                f"{table}: missing columns {sorted(missing)}"
            )


def result_sql():
    # v2 の lane は艇番。boat_no は機材番号。
    # supplement の boat_no は艇番。
    return """
        SELECT race_date, venue_code, race_no,
               lane AS lane, racer_id,
               entry_course, finish_position
        FROM race_boat_results_v2

        UNION ALL

        SELECT race_date, venue_code, race_no,
               boat_no AS lane, racer_id,
               entry_course, finish_position
        FROM race_boat_results_supplement

        ORDER BY race_date, venue_code, race_no, lane
    """


def create_build_table(conn):
    # 作成途中のテーブルのみ再作成。
    # 完成済みの特徴量テーブルは残す。
    conn.execute(f"DROP TABLE IF EXISTS {BUILD_TABLE}")

    conn.execute(f"""
        CREATE TABLE {BUILD_TABLE} (
            race_date TEXT NOT NULL,
            venue_code INTEGER NOT NULL,
            race_no INTEGER NOT NULL,
            lane INTEGER NOT NULL,
            racer_id INTEGER NOT NULL,

            course_starts_365d INTEGER NOT NULL,
            course_win_rate_365d REAL,
            course_top3_rate_365d REAL,

            venue_starts_365d INTEGER NOT NULL,
            venue_win_rate_365d REAL,
            venue_top3_rate_365d REAL,

            venue_course_starts_365d INTEGER NOT NULL,
            venue_course_win_rate_365d REAL,
            venue_course_top3_rate_365d REAL,

            PRIMARY KEY (
                race_date, venue_code, race_no, lane
            )
        )
    """)
    conn.commit()


def empty_stats():
    # [出走回数, 1着回数, 3着以内回数]
    return [0, 0, 0]


def change_stat(stats, key, win, top3, delta):
    item = stats[key]
    item[0] += delta
    item[1] += delta * win
    item[2] += delta * top3

    if any(v < 0 for v in item):
        raise RuntimeError(f"Negative counter: {key} {item}")


def compute_rates(item):
    n, wins, top3 = item

    if n == 0:
        return (0, None, None)

    return (
        n,
        wins / n,
        top3 / n
    )


def make_insert_sql():
    return f"""
        INSERT INTO {BUILD_TABLE} (
            race_date, venue_code, race_no, lane, racer_id,
            course_starts_365d,
            course_win_rate_365d,
            course_top3_rate_365d,
            venue_starts_365d,
            venue_win_rate_365d,
            venue_top3_rate_365d,
            venue_course_starts_365d,
            venue_course_win_rate_365d,
            venue_course_top3_rate_365d
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?
        )
    """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--end",
        help="処理最終日 YYYY-MM-DD（省略時は全件）"
    )
    args = parser.parse_args()

    if args.end:
        date.fromisoformat(args.end)

    if not SOURCE_DB.exists():
        raise FileNotFoundError(SOURCE_DB)

    if not FEATURE_DB.exists():
        raise FileNotFoundError(
            "features_fast.db がありません。PHASE 6を先に実行してください。"
        )

    src = sqlite3.connect(
        SOURCE_DB.as_uri() + "?mode=ro",
        uri=True,
        timeout=60
    )
    dst = sqlite3.connect(FEATURE_DB, timeout=60)

    try:
        validate_sources(src)

        # 選手・コース、選手・競艇場、
        # 選手・競艇場・コースの集計を保持
        by_course = defaultdict(empty_stats)
        by_venue = defaultdict(empty_stats)
        by_venue_course = defaultdict(empty_stats)

        # 過去365日分の結果のみを保持
        active_events = deque()

        entry_query = """
            SELECT race_date, venue_code, race_no,
                   lane, racer_id
            FROM program_entries
        """

        entry_params = []
        if args.end:
            entry_query += " WHERE race_date <= ?"
            entry_params.append(args.end)

        entry_query += """
            ORDER BY race_date, venue_code, race_no, lane
        """

        entry_cursor = src.execute(
            entry_query, entry_params
        )
        result_cursor = src.execute(result_sql())
        next_result = result_cursor.fetchone()

        create_build_table(dst)
        insert_sql = make_insert_sql()

        total = 0
        course_covered = 0
        venue_covered = 0
        venue_course_covered = 0
        missing_course_history = 0

        batch = []
        current_day = None
        day_entries = []

        def update_event(event, delta):
            ordinal, racer, venue, course, win, top3 = event

            change_stat(
                by_venue, (racer, venue),
                win, top3, delta
            )

            # 過去の実際の進入コースが不明なら、
            # コース別統計には含めない
            if course is not None:
                change_stat(
                    by_course, (racer, course),
                    win, top3, delta
                )
                change_stat(
                    by_venue_course, (racer, venue, course),
                    win, top3, delta
                )

        def process_day(day, entries):
            nonlocal next_result, total, batch
            nonlocal course_covered, venue_covered
            nonlocal venue_course_covered
            nonlocal missing_course_history

            today = date.fromisoformat(day).toordinal()
            threshold = today - WINDOW

            # 365日の窓より古い履歴を削除
            while (
                active_events
                and active_events[0][0] < threshold
            ):
                old_event = active_events.popleft()
                update_event(old_event, -1)

            # 当日より前の競走結果だけを追加
            while (
                next_result is not None
                and next_result[0] < day
            ):
                (
                    result_day, result_venue, result_race,
                    result_lane, racer_id,
                    entry_course, finish
                ) = next_result

                if racer_id is not None:
                    result_ordinal = date.fromisoformat(
                        result_day
                    ).toordinal()

                    # 365日より古い履歴は追加不要
                    if result_ordinal >= threshold:
                        try:
                            position = (
                                int(finish)
                                if finish is not None else None
                            )
                        except (ValueError, TypeError):
                            position = None

                        try:
                            course = (
                                int(entry_course)
                                if entry_course is not None
                                else None
                            )
                        except (ValueError, TypeError):
                            course = None

                        if course not in (1, 2, 3, 4, 5, 6):
                            course = None

                        if course is None:
                            missing_course_history += 1

                        event = (
                            result_ordinal,
                            int(racer_id),
                            int(result_venue),
                            course,
                            int(position == 1),
                            int(
                                position is not None
                                and 1 <= position <= 3
                            )
                        )

                        active_events.append(event)
                        update_event(event, +1)

                next_result = result_cursor.fetchone()

            for race_day, venue, race, lane, racer_id in entries:
                if racer_id is None:
                    raise ValueError(
                        f"Missing racer ID: {race_day} "
                        f"venue={venue} race={race} lane={lane}"
                    )

                venue = int(venue)
                lane = int(lane)
                racer_id = int(racer_id)

                if lane not in (1, 2, 3, 4, 5, 6):
                    raise ValueError(
                        f"Invalid lane: {race_day} "
                        f"{venue}R{race} lane={lane}"
                    )

                # 現在の艇番を進入コースの代理値にする
                course_stats = compute_rates(
                    by_course[(racer_id, lane)]
                )
                venue_stats = compute_rates(
                    by_venue[(racer_id, venue)]
                )
                venue_course_stats = compute_rates(
                    by_venue_course[(racer_id, venue, lane)]
                )

                course_covered += int(course_stats[0] > 0)
                venue_covered += int(venue_stats[0] > 0)
                venue_course_covered += int(
                    venue_course_stats[0] > 0
                )

                batch.append((
                    race_day, venue, race, lane, racer_id,
                    *course_stats,
                    *venue_stats,
                    *venue_course_stats
                ))

                total += 1

                if len(batch) >= BATCH_SIZE:
                    dst.executemany(insert_sql, batch)
                    dst.commit()
                    batch.clear()

                if total % 100000 == 0:
                    print(
                        f"Processed: {total:,} "
                        f"| course: {course_covered:,} "
                        f"| venue: {venue_covered:,} "
                        f"| venue-course: {venue_course_covered:,}",
                        flush=True
                    )

        print("========== PHASE 6.1 FAST ==========")
        print("Source:", SOURCE_DB)
        print("Output:", FEATURE_DB)
        print("Period: past 365 days")
        print("End date:", args.end or "ALL")

        for entry in entry_cursor:
            day = entry[0]

            if current_day is None:
                current_day = day

            if day != current_day:
                process_day(current_day, day_entries)
                current_day = day
                day_entries = []

            day_entries.append(entry)

        if day_entries:
            process_day(current_day, day_entries)

        if batch:
            dst.executemany(insert_sql, batch)
            dst.commit()

        built = dst.execute(
            f"SELECT COUNT(*) FROM {BUILD_TABLE}"
        ).fetchone()[0]

        if built != total:
            raise RuntimeError(
                f"Count mismatch: processed={total}, stored={built}"
            )

        # 完成してから本番テーブルへ置き換える
        # 中断された場合、旧テーブルは維持される
        dst.execute("BEGIN")
        dst.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")
        dst.execute(
            f"ALTER TABLE {BUILD_TABLE} RENAME TO {OUTPUT_TABLE}"
        )
        dst.commit()

        print("\n=== SUMMARY ===")
        print("Processed entries:", f"{total:,}")
        print("Stored rows:", f"{built:,}")
        print("Course history available:", f"{course_covered:,}")
        print("Venue history available:", f"{venue_covered:,}")
        print(
            "Venue-course history available:",
            f"{venue_course_covered:,}"
        )
        print(
            "Historical records without entry course:",
            f"{missing_course_history:,}"
        )

        if total:
            print(
                "Course coverage:",
                f"{course_covered / total * 100:.2f}%"
            )
            print(
                "Venue coverage:",
                f"{venue_covered / total * 100:.2f}%"
            )
            print(
                "Venue-course coverage:",
                f"{venue_course_covered / total * 100:.2f}%"
            )

        print("\n=== SAMPLE ===")
        for row in dst.execute(f"""
            SELECT
                race_date, venue_code, race_no, lane,
                racer_id,
                course_starts_365d,
                course_win_rate_365d,
                venue_starts_365d,
                venue_course_starts_365d
            FROM {OUTPUT_TABLE}
            ORDER BY race_date DESC, venue_code, race_no, lane
            LIMIT 10
        """):
            print(row)

        print("\nPHASE 6.1 FAST COMPLETE")

    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    main()
