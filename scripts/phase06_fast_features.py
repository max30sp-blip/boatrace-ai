
import argparse
import sqlite3
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DB = ROOT / "data" / "boatrace.db"
OUTPUT_DB = ROOT / "data" / "features_fast.db"

WINDOWS = (30, 90, 180)
BATCH_SIZE = 5000


def columns(conn, table):
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    if not rows:
        raise ValueError(f"Table not found: {table}")
    return {row[1] for row in rows}


def pick(cols, choices, table, required=True):
    for name in choices:
        if name in cols:
            return name
    if required:
        raise ValueError(
            f"{table}: expected one of {choices}, "
            f"available columns={sorted(cols)}"
        )
    return None



def make_result_sql(conn):
    parts = []

    sources = [
        ("race_boat_results_v2", "lane"),
        ("race_boat_results_supplement", "boat_no"),
    ]

    for table, lane_col in sources:
        cols = columns(conn, table)

        if lane_col not in cols:
            raise ValueError(
                f"{table}: missing {lane_col}"
            )

        racer_expr = (
            "r.racer_id"
            if "racer_id" in cols
            else "b.racer_id"
        )

        if "racer_id" in cols:
            join_sql = ""
        else:
            join_sql = f"""
                JOIN program_entries b
                  ON b.race_date = r.race_date
                 AND b.venue_code = r.venue_code
                 AND b.race_no = r.race_no
                 AND b.lane = r.{lane_col}
            """

        parts.append(f"""
            SELECT
                r.race_date,
                r.venue_code,
                r.race_no,
                r.{lane_col} AS boat_no,
                {racer_expr} AS racer_id,
                r.finish_position,
                r.start_timing
            FROM {table} r
            {join_sql}
        """)

    return """
        SELECT * FROM (
    """ + "\nUNION ALL\n".join(parts) + """
        )
        ORDER BY race_date, venue_code, race_no, boat_no
    """



def make_output(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS racer_features_fast (
            race_date TEXT NOT NULL,
            venue_code INTEGER NOT NULL,
            race_no INTEGER NOT NULL,
            boat_no INTEGER NOT NULL,
            racer_id INTEGER NOT NULL,

            starts_30d INTEGER NOT NULL,
            win_rate_30d REAL,
            top3_rate_30d REAL,
            avg_st_30d REAL,

            starts_90d INTEGER NOT NULL,
            win_rate_90d REAL,
            top3_rate_90d REAL,
            avg_st_90d REAL,

            starts_180d INTEGER NOT NULL,
            win_rate_180d REAL,
            top3_rate_180d REAL,
            avg_st_180d REAL,

            PRIMARY KEY (
                race_date, venue_code, race_no, boat_no
            )
        );
    """)


INSERT_SQL = """
    INSERT OR REPLACE INTO racer_features_fast (
        race_date, venue_code, race_no, boat_no, racer_id,
        starts_30d, win_rate_30d, top3_rate_30d, avg_st_30d,
        starts_90d, win_rate_90d, top3_rate_90d, avg_st_90d,
        starts_180d, win_rate_180d, top3_rate_180d, avg_st_180d
    ) VALUES (
        ?, ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?, ?
    )
"""


def summarize(history, today):
    # history: (日付ordinal, 着順, ST) のdeque
    # この時点では当日データは追加されていない
    values = {}

    for days in WINDOWS:
        threshold = today - days
        recent = [
            (finish, st)
            for day, finish, st in history
            if day >= threshold
        ]

        n = len(recent)

        if n == 0:
            values[days] = (0, None, None, None)
            continue

        wins = sum(finish == 1 for finish, _ in recent)
        top3 = sum(
            finish is not None and 1 <= finish <= 3
            for finish, _ in recent
        )

        st_values = [
            st for _, st in recent if st is not None
        ]

        avg_st = (
            sum(st_values) / len(st_values)
            if st_values else None
        )

        values[days] = (
            n,
            wins / n,
            top3 / n,
            avg_st,
        )

    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--end",
        default=None,
        help="処理する最終日 YYYY-MM-DD（省略時は全件）"
    )
    args = parser.parse_args()

    if not SOURCE_DB.exists():
        raise FileNotFoundError(SOURCE_DB)

    if args.end:
        date.fromisoformat(args.end)

    # 元DBは読み取り専用
    src = sqlite3.connect(
        SOURCE_DB.as_uri() + "?mode=ro",
        uri=True,
        timeout=60
    )
    dst = sqlite3.connect(OUTPUT_DB, timeout=60)

    try:
        entry_cols = columns(src, "program_entries")
        required = {
            "race_date", "venue_code", "race_no",
            "lane", "racer_id"
        }
        if not required.issubset(entry_cols):
            raise ValueError(
                f"program_entries columns={sorted(entry_cols)}"
            )

        result_sql = make_result_sql(src)

        entry_sql = """
            SELECT
                race_date, venue_code, race_no,
                lane AS boat_no, racer_id
            FROM program_entries
        """

        params = []
        if args.end:
            entry_sql += " WHERE race_date <= ?"
            params.append(args.end)

        entry_sql += """
            ORDER BY race_date, venue_code, race_no, boat_no
        """

        entry_cursor = src.execute(entry_sql, params)
        result_cursor = src.execute(result_sql)

        make_output(dst)
        dst.commit()

        # 選手ID別の過去180日履歴
        history = defaultdict(deque)

        next_result = result_cursor.fetchone()

        total = 0
        covered = 0
        batch = []

        current_day = None
        day_rows = []

        def flush_day(day, rows, next_result):
            nonlocal total, covered, batch

            today = date.fromisoformat(day).toordinal()

            # 当日より前の結果のみ履歴に追加
            while next_result is not None and next_result[0] < day:
                (
                    result_day, venue, race, boat,
                    racer_id, finish, st
                ) = next_result

                if racer_id is not None:
                    rid = int(racer_id)
                    result_ordinal = date.fromisoformat(
                        result_day
                    ).toordinal()

                    # 着順: 特殊結果はNone扱い
                    try:
                        position = (
                            int(finish)
                            if finish is not None else None
                        )
                    except (ValueError, TypeError):
                        position = None

                    try:
                        start = (
                            float(st) if st is not None else None
                        )
                    except (ValueError, TypeError):
                        start = None

                    if start is not None and not (
                        -1.0 <= start <= 2.0
                    ):
                        start = None

                    history[rid].append(
                        (result_ordinal, position, start)
                    )

                next_result = result_cursor.fetchone()

            for race_day, venue, race, boat, racer_id in rows:
                if racer_id is None:
                    raise ValueError(
                        f"Missing racer_id: {race_day} "
                        f"{venue}R{race} boat={boat}"
                    )

                rid = int(racer_id)
                h = history[rid]

                # 180日の窓から外れた古い記録を削除
                threshold = today - max(WINDOWS)
                while h and h[0][0] < threshold:
                    h.popleft()

                stats = summarize(h, today)

                if stats[180][0] > 0:
                    covered += 1

                batch.append((
                    race_day, venue, race, boat, rid,
                    *stats[30],
                    *stats[90],
                    *stats[180],
                ))

                total += 1

                if len(batch) >= BATCH_SIZE:
                    dst.executemany(INSERT_SQL, batch)
                    dst.commit()
                    batch.clear()

                if total % 100000 == 0:
                    print(
                        f"Processed entries: {total:,} "
                        f"| historical coverage: {covered:,}",
                        flush=True
                    )

            return next_result

        print("========== PHASE 6 FAST ==========")
        print("Source:", SOURCE_DB)
        print("Output:", OUTPUT_DB)
        print("End date:", args.end or "ALL")
        print("Source mode: READ ONLY")

        for entry in entry_cursor:
            day = entry[0]

            if current_day is None:
                current_day = day

            if day != current_day:
                next_result = flush_day(
                    current_day, day_rows, next_result
                )
                day_rows = []
                current_day = day

            day_rows.append(entry)

        if day_rows:
            next_result = flush_day(
                current_day, day_rows, next_result
            )

        if batch:
            dst.executemany(INSERT_SQL, batch)
            dst.commit()

        rows_in_db = dst.execute("""
            SELECT COUNT(*) FROM racer_features_fast
        """).fetchone()[0]

        print("\n=== SUMMARY ===")
        print("Processed entries:", f"{total:,}")
        print("Rows in output DB:", f"{rows_in_db:,}")
        print("Entries with 180d history:", f"{covered:,}")

        if total:
            print(
                "History coverage:",
                f"{covered / total * 100:.2f}%"
            )

        print("\n=== SAMPLE ===")
        for row in dst.execute("""
            SELECT race_date, venue_code, race_no,
                   boat_no, racer_id,
                   starts_30d, win_rate_30d,
                   starts_180d, win_rate_180d
            FROM racer_features_fast
            ORDER BY race_date DESC, venue_code, race_no, boat_no
            LIMIT 10
        """):
            print(row)

        print("\nPHASE 6 FAST COMPLETE")

    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    main()
