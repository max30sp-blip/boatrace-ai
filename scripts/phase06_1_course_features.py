
import sqlite3
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "boatrace.db"
DAYS = 365
PRIOR_WEIGHT = 20


def ratio(wins, starts):
    return wins / starts if starts else None


def count_stats(rows):
    n = len(rows)
    wins = sum(r[3] == 1 for r in rows)
    top3 = sum(
        r[3] is not None and r[3] <= 3
        for r in rows
    )
    return n, wins, top3


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS racer_course_features (
            race_date TEXT NOT NULL,
            venue_code INTEGER NOT NULL,
            race_no INTEGER NOT NULL,
            lane INTEGER NOT NULL,
            racer_id INTEGER NOT NULL,
            lane_starts_365 INTEGER,
            lane_win_rate_365 REAL,
            lane_top3_rate_365 REAL,
            venue_starts_365 INTEGER,
            venue_win_rate_365 REAL,
            venue_lane_starts_365 INTEGER,
            venue_lane_win_rate_365 REAL,
            smoothed_lane_win_rate REAL,
            actual_course_starts_365 INTEGER,
            actual_course_win_rate_365 REAL,
            course_change_rate_365 REAL,
            PRIMARY KEY (
                race_date, venue_code, race_no, lane
            )
        )
    """)


def build(conn):
    create_table(conn)

    entries = conn.execute("""
        SELECT race_date, venue_code, race_no,
               lane, racer_id
        FROM program_entries
        ORDER BY race_date, venue_code, race_no, lane
    """).fetchall()

    cursor = conn.execute("""
        SELECT race_date, venue_code, lane,
               finish_position, entry_course,
               racer_id
        FROM race_boat_results_v2
        ORDER BY race_date, venue_code, lane
    """)

    next_result = cursor.fetchone()

    history = defaultdict(deque)
    global_history = defaultdict(deque)
    output = []
    current_date = None

    for index, entry in enumerate(entries, 1):
        day_text, venue, race_no, lane, racer = entry
        day = date.fromisoformat(day_text)

        if day_text != current_date:
            current_date = day_text

            # 予測対象の日より前の結果だけ追加
            while (
                next_result is not None
                and next_result[0] < day_text
            ):
                d, v, l, finish, course, rid = next_result

                if course is not None:
                    item = (
                        date.fromisoformat(d),
                        v, l, finish, course
                    )
                    history[rid].append(item)
                    global_history[l].append(item)

                next_result = cursor.fetchone()

            # 全体の古い履歴を整理
            for l in range(1, 7):
                q = global_history[l]
                while q and (day - q[0][0]).days > DAYS:
                    q.popleft()

        h = history[racer]

        while h and (day - h[0][0]).days > DAYS:
            h.popleft()

        same_lane = [r for r in h if r[2] == lane]
        same_venue = [r for r in h if r[1] == venue]
        venue_lane = [
            r for r in same_lane if r[1] == venue
        ]

        # 過去にその選手が実際に同一コースから
        # 進入した成績（予測レースの実進入ではない）
        same_course = [
            r for r in h if r[4] == lane
        ]

        lane_n, lane_w, lane_t = count_stats(same_lane)
        venue_n, venue_w, _ = count_stats(same_venue)
        vl_n, vl_w, _ = count_stats(venue_lane)
        course_n, course_w, _ = count_stats(same_course)

        changed = sum(
            r[2] != r[4] for r in same_lane
        )

        # その日より前の同枠全体の勝率を事前確率に
        global_rows = global_history[lane]
        global_n, global_w, _ = count_stats(global_rows)

        if global_n:
            baseline = global_w / global_n
            smoothed = (
                lane_w + PRIOR_WEIGHT * baseline
            ) / (lane_n + PRIOR_WEIGHT)
        else:
            smoothed = None

        output.append((
            day_text, venue, race_no, lane, racer,
            lane_n,
            ratio(lane_w, lane_n),
            ratio(lane_t, lane_n),
            venue_n,
            ratio(venue_w, venue_n),
            vl_n,
            ratio(vl_w, vl_n),
            smoothed,
            course_n,
            ratio(course_w, course_n),
            ratio(changed, lane_n)
        ))

        if index % 10000 == 0:
            print(f"Calculated: {index:,}/{len(entries):,}")

    with conn:
        conn.execute("DELETE FROM racer_course_features")
        conn.executemany("""
            INSERT INTO racer_course_features
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, output)

    print("\n========== PHASE 6.1 SUMMARY ==========")
    print("Features:", f"{len(output):,}")

    for row in conn.execute("""
        SELECT race_date, venue_code, race_no, lane,
               racer_id, lane_starts_365,
               lane_win_rate_365,
               smoothed_lane_win_rate,
               course_change_rate_365
        FROM racer_course_features
        ORDER BY race_date DESC,
                 venue_code, race_no, lane
        LIMIT 12
    """):
        print(row)


def main():
    if not DB.exists():
        raise FileNotFoundError(DB)

    with sqlite3.connect(DB, timeout=30) as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        build(conn)


if __name__ == "__main__":
    main()
