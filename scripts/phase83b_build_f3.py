
import csv
import sqlite3
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict, deque
from itertools import groupby

ROOT = Path(r"C:\boatrace_ai")
DB = ROOT / "data" / "boatrace.db"
OUT = ROOT / "outputs" / "phase83b_f3_preview.csv"

TARGET_START = date(2026, 10, 23)
TARGET_END = date(2026, 10, 29)

WINDOWS = (30, 90, 180)
PRIOR_TOP2 = 1 / 3
PRIOR_STRENGTH = 20

# 集計対象データはターゲットの180日前から
HISTORY_START = TARGET_START - timedelta(days=max(WINDOWS))


class RollingHistory:
    def __init__(self, days):
        self.days = days
        self.queue = deque()
        self.motor = defaultdict(lambda: [0, 0, 0, 0])
        self.boat = defaultdict(lambda: [0, 0, 0, 0])

    @staticmethod
    def update(store, key, finish, sign):
        if key is None:
            return

        stats = store[key]
        stats[0] += sign
        stats[1] += sign * (finish == 1)
        stats[2] += sign * (finish <= 2)
        stats[3] += sign * (finish <= 3)

        if stats[0] == 0:
            del store[key]

    def expire(self, current_day):
        while self.queue:
            old_day, mk, bk, finish = self.queue[0]

            if (current_day - old_day).days < self.days:
                break

            self.queue.popleft()

            self.update(self.motor, mk, finish, -1)
            self.update(self.boat, bk, finish, -1)

    def add(self, current_day, mk, bk, finish):
        if not 1 <= finish <= 6:
            return

        self.queue.append((current_day, mk, bk, finish))

        self.update(self.motor, mk, finish, 1)
        self.update(self.boat, bk, finish, 1)

    def features(self, kind, key):
        store = self.motor if kind == "motor" else self.boat
        n, wins, top2, top3 = store.get(key, [0, 0, 0, 0])

        if n > 0:
            win_rate = wins / n
            top2_rate = top2 / n
            top3_rate = top3 / n
        else:
            win_rate = None
            top2_rate = None
            top3_rate = None

        smooth = (
            (top2 + PRIOR_STRENGTH * PRIOR_TOP2)
            / (n + PRIOR_STRENGTH)
        )

        return [
            n,
            win_rate,
            top2_rate,
            top3_rate,
            smooth
        ]


def machine_key(venue, number):
    if venue is None or number is None:
        return None
    if number <= 0:
        return None
    return (venue, number)


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        DB.as_uri() + "?mode=ro", uri=True
    )
    connection.execute("PRAGMA query_only=ON")

    sql = """
        SELECT race_date, venue_code, race_no,
               lane, racer_id, motor_no,
               boat_no, finish_position
        FROM race_boat_results_v2
        WHERE race_date >= ?
          AND race_date <= ?
        ORDER BY race_date, venue_code, race_no, lane
    """

    cursor = connection.execute(
        sql,
        (HISTORY_START.isoformat(), TARGET_END.isoformat())
    )

    histories = {
        window: RollingHistory(window)
        for window in WINDOWS
    }

    fieldnames = [
        "race_date", "venue_code", "race_no",
        "lane", "racer_id", "motor_no", "boat_no"
    ]

    for kind in ("motor", "boat"):
        for window in WINDOWS:
            prefix = f"{kind}_{window}"
            fieldnames.extend([
                f"{prefix}_starts",
                f"{prefix}_win_rate",
                f"{prefix}_top2_rate",
                f"{prefix}_top3_rate",
                f"{prefix}_top2_smoothed"
            ])

    total = 0
    target_count = 0
    invalid_finishes = 0

    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        for day_string, day_iterator in groupby(
            cursor, key=lambda r: r[0]
        ):
            current_day = date.fromisoformat(day_string)
            day_rows = list(day_iterator)

            # 当日分を加える前に、古い履歴を除外
            for history in histories.values():
                history.expire(current_day)

            # 当日の全レースを予測してから履歴を更新
            # 同じ日の結果を使わないのでリークを防止
            if TARGET_START <= current_day <= TARGET_END:
                for row in day_rows:
                    (
                        race_date, venue, race_no,
                        lane, racer_id, motor_no,
                        boat_no, finish
                    ) = row

                    mk = machine_key(venue, motor_no)
                    bk = machine_key(venue, boat_no)

                    output = [
                        race_date, venue, race_no,
                        lane, racer_id, motor_no, boat_no
                    ]

                    for kind, key in (
                        ("motor", mk), ("boat", bk)
                    ):
                        for window in WINDOWS:
                            output.extend(
                                histories[window].features(kind, key)
                            )

                    writer.writerow(output)
                    target_count += 1

            # 予測対象日の特徴量生成後に履歴を更新
            for row in day_rows:
                _, venue, _, _, _, motor_no, boat_no, finish = row
                total += 1

                if finish is None or not 1 <= finish <= 6:
                    invalid_finishes += 1
                    continue

                mk = machine_key(venue, motor_no)
                bk = machine_key(venue, boat_no)

                for history in histories.values():
                    history.add(current_day, mk, bk, finish)

            if current_day.day == 1:
                print("Processed:", current_day)

    connection.close()

    print("\n=== PHASE 8.3-B F3 COMPLETE ===")
    print("Historical rows processed:", total)
    print("Invalid finishes skipped:", invalid_finishes)
    print("Target feature rows:", target_count)
    print("Feature columns:", len(fieldnames))
    print("Output:", OUT)


if __name__ == "__main__":
    build()
