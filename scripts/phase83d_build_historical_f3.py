
import sqlite3
from pathlib import Path
from datetime import date
from collections import defaultdict, deque
from itertools import groupby

ROOT = Path(r"C:\boatrace_ai")
RESULT_DB = ROOT / "data" / "boatrace.db"
TRAIN_DB = ROOT / "data" / "train_dataset_v1.db"
OUTPUT_DB = ROOT / "outputs" / "phase83d_historical_f3.db"

HISTORY_START = "2016-01-01"
TARGET_START = "2017-01-01"
TARGET_END = "2024-12-31"

WINDOWS = (30, 90)
ALPHA = 20
PRIOR = 1 / 3
BATCH_SIZE = 10000


def open_readonly(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    con = sqlite3.connect(
        path.as_uri() + "?mode=ro",
        uri=True
    )
    con.execute("PRAGMA query_only=ON")
    return con


def machine_key(venue, number):
    if venue is None or number is None:
        return None
    if not 1 <= number <= 200:
        return None
    return (venue, number)


class RollingHistory:
    def __init__(self, days):
        self.days = days
        self.events = deque()
        self.motor = defaultdict(lambda: [0, 0])
        self.boat = defaultdict(lambda: [0, 0])

    @staticmethod
    def update(store, key, top2, delta):
        if key is None:
            return

        value = store[key]
        value[0] += delta
        value[1] += delta * top2

        if value[0] == 0:
            del store[key]

    def expire(self, day):
        while self.events:
            old, mk, bk, top2 = self.events[0]

            if (day - old).days < self.days:
                break

            self.events.popleft()
            self.update(self.motor, mk, top2, -1)
            self.update(self.boat, bk, top2, -1)

    def add(self, day, mk, bk, finish):
        top2 = int(finish <= 2)
        self.events.append((day, mk, bk, top2))

        self.update(self.motor, mk, top2, 1)
        self.update(self.boat, bk, top2, 1)

    def get(self, kind, key):
        if key is None:
            return None, None

        store = self.motor if kind == "motor" else self.boat
        n, top2 = store.get(key, (0, 0))

        smoothed = (
            top2 + ALPHA * PRIOR
        ) / (n + ALPHA)

        return n, smoothed


def main():
    if OUTPUT_DB.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_DB}\n"
            "既存の検証結果を保護するため停止しました。"
        )

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)

    train = open_readonly(TRAIN_DB)

    target_races = set(train.execute("""
        SELECT race_date, venue_code, race_no
        FROM train_races
        WHERE race_date BETWEEN ? AND ?
    """, (TARGET_START, TARGET_END)).fetchall())

    train.close()

    print("=== PHASE 8.3-D HISTORICAL F3 ===")
    print("Target races:", len(target_races))
    print("History start:", HISTORY_START)
    print("Target period:", TARGET_START, "to", TARGET_END)

    source = open_readonly(RESULT_DB)

    output = sqlite3.connect(OUTPUT_DB)
    output.execute("PRAGMA journal_mode=DELETE")
    output.execute("PRAGMA synchronous=NORMAL")

    output.execute("""
        CREATE TABLE historical_f3 (
            race_date TEXT NOT NULL,
            venue_code INTEGER NOT NULL,
            race_no INTEGER NOT NULL,
            lane INTEGER NOT NULL,

            motor_30_starts INTEGER,
            motor_30_top2_smoothed REAL,
            motor_90_starts INTEGER,
            motor_90_top2_smoothed REAL,

            boat_30_starts INTEGER,
            boat_30_top2_smoothed REAL,
            boat_90_starts INTEGER,
            boat_90_top2_smoothed REAL,

            PRIMARY KEY (
                race_date, venue_code, race_no, lane
            )
        ) WITHOUT ROWID
    """)

    output.execute("""
        CREATE TABLE metadata (
            name TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    metadata = {
        "target_start": TARGET_START,
        "target_end": TARGET_END,
        "history_start": HISTORY_START,
        "alpha": str(ALPHA),
        "prior": str(PRIOR),
        "reset_policy": "unverified_rolling_proxy",
        "same_day_results": "excluded",
    }

    output.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        metadata.items()
    )

    sql = """
        SELECT race_date, venue_code, race_no,
               lane, motor_no, boat_no,
               finish_position
        FROM race_boat_results_v2
        WHERE race_date BETWEEN ? AND ?
        ORDER BY race_date, venue_code, race_no, lane
    """

    cursor = source.execute(
        sql, (HISTORY_START, TARGET_END)
    )

    histories = {
        days: RollingHistory(days)
        for days in WINDOWS
    }

    insert_sql = """
        INSERT INTO historical_f3 VALUES (
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?
        )
    """

    batch = []
    seen_races = set()

    processed = 0
    written = 0
    invalid = 0

    for day_string, grouped in groupby(
        cursor, key=lambda r: r[0]
    ):
        day = date.fromisoformat(day_string)
        rows = list(grouped)

        for history in histories.values():
            history.expire(day)

        # 予測対象日の結果を更新する前に特徴量を生成
        if TARGET_START <= day_string <= TARGET_END:
            for row in rows:
                (
                    d, venue, race, lane,
                    motor, boat, finish
                ) = row

                race_key = (d, venue, race)

                if race_key not in target_races:
                    continue

                seen_races.add(race_key)

                mk = machine_key(venue, motor)
                bk = machine_key(venue, boat)

                m30 = histories[30].get("motor", mk)
                m90 = histories[90].get("motor", mk)
                b30 = histories[30].get("boat", bk)
                b90 = histories[90].get("boat", bk)

                batch.append((
                    d, venue, race, lane,
                    *m30, *m90, *b30, *b90
                ))

                if len(batch) >= BATCH_SIZE:
                    output.executemany(insert_sql, batch)
                    output.commit()
                    written += len(batch)
                    batch.clear()

        # 当日すべてのレースの特徴量計算後に更新
        for row in rows:
            _, venue, _, _, motor, boat, finish = row
            processed += 1

            if finish is None or not 1 <= finish <= 6:
                invalid += 1
                continue

            mk = machine_key(venue, motor)
            bk = machine_key(venue, boat)

            for history in histories.values():
                history.add(day, mk, bk, finish)

        if day.day == 1:
            print(
                "Processed:", day_string,
                "written:", written + len(batch),
                flush=True
            )

    if batch:
        output.executemany(insert_sql, batch)
        written += len(batch)

    output.commit()

    missing = target_races - seen_races

    summary = output.execute("""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT
                race_date || '/' || venue_code || '/' || race_no
            ) AS races,
            SUM(motor_30_starts IS NULL) AS missing_motor,
            SUM(boat_30_starts IS NULL) AS missing_boat
        FROM historical_f3
    """).fetchone()

    incomplete = output.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT COUNT(*) AS n
            FROM historical_f3
            GROUP BY race_date, venue_code, race_no
            HAVING COUNT(*) != 6
        )
    """).fetchone()[0]

    source.close()
    output.close()

    print("\n=== PHASE 8.3-D HISTORICAL F3 COMPLETE ===")
    print("Processed result rows:", processed)
    print("Invalid finishes skipped:", invalid)
    print("Written boat rows:", written)
    print("Output boat rows:", summary[0])
    print("Matched races:", summary[1])
    print("Unmatched races:", len(missing))
    print("Incomplete matched races:", incomplete)
    print("Missing motor keys:", summary[2])
    print("Missing boat keys:", summary[3])
    print("Output:", OUTPUT_DB)

    print("\nMissing race examples:")
    for item in sorted(missing)[:10]:
        print(item)


if __name__ == "__main__":
    main()
