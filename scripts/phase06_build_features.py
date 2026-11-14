
import argparse
import sqlite3
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "boatrace.db"

WINDOWS = (30, 90, 180)


def setup(conn):
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS racer_features (
            race_date TEXT NOT NULL,
            venue_code INTEGER NOT NULL,
            race_no INTEGER NOT NULL,
            lane INTEGER NOT NULL,
            racer_id INTEGER NOT NULL,
            starts_30 INTEGER NOT NULL,
            win_rate_30 REAL,
            top3_rate_30 REAL,
            avg_st_30 REAL,
            starts_90 INTEGER NOT NULL,
            win_rate_90 REAL,
            top3_rate_90 REAL,
            avg_st_90 REAL,
            starts_180 INTEGER NOT NULL,
            win_rate_180 REAL,
            top3_rate_180 REAL,
            avg_st_180 REAL,
            same_lane_starts_180 INTEGER NOT NULL,
            same_lane_win_rate_180 REAL,
            same_venue_starts_180 INTEGER NOT NULL,
            same_venue_win_rate_180 REAL,
            PRIMARY KEY(race_date, venue_code, race_no, lane)
        )
    """)
    conn.commit()


def rate(numerator, denominator):
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def make_features(history, day, venue, lane):
    results = []

    for window in WINDOWS:
        samples = [
            r for r in history
            if 0 < (day - r[0]).days <= window
        ]

        n = len(samples)
        wins = sum(r[3] == 1 for r in samples)
        top3 = sum(
            r[3] is not None and r[3] <= 3
            for r in samples
        )

        sts = [
            r[4] for r in samples
            if r[4] is not None and r[4] >= 0
        ]

        results.extend([
            n,
            rate(wins, n),
            rate(top3, n),
            round(sum(sts) / len(sts), 5) if sts else None
        ])

    recent = [
        r for r in history
        if 0 < (day - r[0]).days <= 180
    ]

    same_lane = [r for r in recent if r[2] == lane]
    same_venue = [r for r in recent if r[1] == venue]

    results.extend([
        len(same_lane),
        rate(
            sum(r[3] == 1 for r in same_lane),
            len(same_lane)
        ),
        len(same_venue),
        rate(
            sum(r[3] == 1 for r in same_venue),
            len(same_venue)
        )
    ])

    return results


def build(conn):
    setup(conn)

    entries = conn.execute("""
        SELECT race_date, venue_code, race_no,
               lane, racer_id
        FROM program_entries
        ORDER BY race_date, venue_code, race_no, lane
    """).fetchall()

    # 全履歴を日付順に読み込む。
    # 処理対象日より前の結果だけを履歴に追加する。
    result_cursor = conn.execute("""
        SELECT race_date, racer_id, venue_code,
               lane, finish_position, start_timing,
               entry_course, result_code
        FROM race_boat_results_v2
        ORDER BY race_date
    """)

    next_result = result_cursor.fetchone()
    histories = defaultdict(deque)

    feature_rows = []
    current_day = None

    for index, entry in enumerate(entries, 1):
        day_text, venue, race_no, lane, racer = entry
        day = date.fromisoformat(day_text)

        if day != current_day:
            current_day = day

            while (
                next_result is not None
                and next_result[0] < day_text
            ):
                (
                    result_day, result_racer,
                    result_venue, result_lane,
                    finish, st, course, status
                ) = next_result

                # 欠場・不出走行は成績母数に含めない
                if course is not None:
                    histories[result_racer].append((
                        date.fromisoformat(result_day),
                        result_venue,
                        result_lane,
                        finish,
                        st
                    ))

                next_result = result_cursor.fetchone()

        h = histories[racer]

        # 180日より古い情報を破棄
        while h and (day - h[0][0]).days > 180:
            h.popleft()

        features = make_features(h, day, venue, lane)

        feature_rows.append((
            day_text, venue, race_no, lane, racer,
            *features
        ))

        if index % 10000 == 0:
            print(f"Calculated: {index:,}/{len(entries):,}")

    with conn:
        conn.execute("DELETE FROM racer_features")
        conn.executemany("""
            INSERT INTO racer_features VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?
            )
        """, feature_rows)

    print("\n========== PHASE 06 SUMMARY ==========")
    print("Feature rows:", f"{len(feature_rows):,}")

    covered = conn.execute("""
        SELECT COUNT(*)
        FROM racer_features
        WHERE starts_180 > 0
    """).fetchone()[0]

    print("Rows with 180-day history:", f"{covered:,}")
    print(
        "History coverage:",
        f"{covered / len(feature_rows) * 100:.2f}%"
        if feature_rows else "N/A"
    )

    print("\nSAMPLE FEATURES")

    for row in conn.execute("""
        SELECT race_date, venue_code, race_no,
               lane, racer_id,
               starts_30, win_rate_30,
               starts_180, win_rate_180,
               same_lane_win_rate_180
        FROM racer_features
        ORDER BY race_date DESC, venue_code, race_no, lane
        LIMIT 12
    """):
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()

    if not DB.exists():
        raise FileNotFoundError(DB)

    with sqlite3.connect(DB, timeout=30) as conn:
        build(conn)


if __name__ == "__main__":
    main()
