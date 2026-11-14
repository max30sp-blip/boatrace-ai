
import csv
import sqlite3
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict, deque
from itertools import groupby

ROOT = Path(r"C:\boatrace_ai")
DB = ROOT / "data" / "boatrace.db"
OUT = ROOT / "outputs" / "phase83b_ability_adjusted.csv"

TARGET_START = date(2026, 10, 23)
TARGET_END = date(2026, 10, 29)

WINDOWS = (30, 90, 180)
MAX_WINDOW = 180

# 選手能力推定のウォームアップ期間も確保
HISTORY_START = TARGET_START - timedelta(days=360)

PRIOR = 1 / 3
RACER_ALPHA = 20
LANE_ALPHA = 50


def equipment_key(venue, number):
    if venue is None or number is None or number <= 0:
        return None
    return (venue, number)


class AbilityHistory:
    def __init__(self):
        self.queue = deque()
        self.racers = defaultdict(lambda: [0, 0])
        self.lanes = defaultdict(lambda: [0, 0])

    def expire(self, day):
        while self.queue:
            old_day, racer, lane, top2 = self.queue[0]

            if (day - old_day).days < MAX_WINDOW:
                break

            self.queue.popleft()

            self.racers[racer][0] -= 1
            self.racers[racer][1] -= top2

            self.lanes[lane][0] -= 1
            self.lanes[lane][1] -= top2

    def probability(self, racer, lane):
        rn, rtop2 = self.racers[racer]
        ln, ltop2 = self.lanes[lane]

        racer_prob = (
            rtop2 + RACER_ALPHA * PRIOR
        ) / (rn + RACER_ALPHA)

        lane_prob = (
            ltop2 + LANE_ALPHA * PRIOR
        ) / (ln + LANE_ALPHA)

        # 簡易的な選手能力・艇番補正
        expected = lane_prob + (racer_prob - PRIOR)

        return max(0.02, min(0.98, expected))

    def add(self, day, racer, lane, finish):
        if racer is None or lane is None:
            return

        top2 = int(finish <= 2)

        self.queue.append((day, racer, lane, top2))

        self.racers[racer][0] += 1
        self.racers[racer][1] += top2

        self.lanes[lane][0] += 1
        self.lanes[lane][1] += top2


class EquipmentHistory:
    def __init__(self):
        self.events = deque()
        self.motor = defaultdict(deque)
        self.boat = defaultdict(deque)

    def expire(self, day):
        while self.events:
            old_day, mk, bk, residual = self.events[0]

            if (day - old_day).days < MAX_WINDOW:
                break

            self.events.popleft()

            for store, key in (
                (self.motor, mk),
                (self.boat, bk)
            ):
                if key is not None:
                    q = store[key]
                    q.popleft()
                    if not q:
                        del store[key]

    def add(self, day, mk, bk, residual):
        self.events.append((day, mk, bk, residual))

        if mk is not None:
            self.motor[mk].append((day, residual))

        if bk is not None:
            self.boat[bk].append((day, residual))

    def features(self, kind, key, day, days):
        store = self.motor if kind == "motor" else self.boat
        q = store.get(key, ())

        values = [
            residual
            for old_day, residual in q
            if 0 <= (day - old_day).days < days
        ]

        n = len(values)

        if n == 0:
            return [0, None, None]

        avg = sum(values) / n

        # 正規化は補正残差の推定値を0へ縮小
        smoothed = sum(values) / (n + 20)

        return [n, avg, smoothed]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        DB.as_uri() + "?mode=ro",
        uri=True
    )
    conn.execute("PRAGMA query_only=ON")

    sql = """
    SELECT race_date, venue_code, race_no,
           lane, racer_id, motor_no,
           boat_no, finish_position
    FROM race_boat_results_v2
    WHERE race_date BETWEEN ? AND ?
    ORDER BY race_date, venue_code, race_no, lane
    """

    cursor = conn.execute(
        sql,
        (
            HISTORY_START.isoformat(),
            TARGET_END.isoformat()
        )
    )

    ability = AbilityHistory()
    equipment = EquipmentHistory()

    columns = [
        "race_date", "venue_code", "race_no",
        "lane", "racer_id",
        "expected_top2_probability"
    ]

    for kind in ("motor", "boat"):
        for window in WINDOWS:
            columns.extend([
                f"{kind}_adjusted_{window}_starts",
                f"{kind}_adjusted_{window}_residual",
                f"{kind}_adjusted_{window}_smoothed"
            ])

    processed = 0
    output_count = 0
    invalid = 0

    with OUT.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(columns)

        for day_string, grouped in groupby(
            cursor, key=lambda r: r[0]
        ):
            day = date.fromisoformat(day_string)
            rows = list(grouped)

            ability.expire(day)
            equipment.expire(day)

            # まず当日以前の情報で特徴量を生成
            if TARGET_START <= day <= TARGET_END:
                for row in rows:
                    (
                        race_date, venue, race_no,
                        lane, racer, motor, boat, finish
                    ) = row

                    mk = equipment_key(venue, motor)
                    bk = equipment_key(venue, boat)

                    expected = ability.probability(
                        racer, lane
                    )

                    values = [
                        race_date, venue, race_no,
                        lane, racer, expected
                    ]

                    for kind, key in (
                        ("motor", mk),
                        ("boat", bk)
                    ):
                        for window in WINDOWS:
                            values.extend(
                                equipment.features(
                                    kind, key, day, window
                                )
                            )

                    writer.writerow(values)
                    output_count += 1

            # 全艇の期待確率を先に確定
            # 当日の結果は期待確率に使わない
            updates = []

            for row in rows:
                (
                    _, venue, _, lane,
                    racer, motor, boat, finish
                ) = row

                processed += 1

                if finish is None or not 1 <= finish <= 6:
                    invalid += 1
                    continue

                expected = ability.probability(
                    racer, lane
                )

                actual = int(finish <= 2)
                residual = actual - expected

                mk = equipment_key(venue, motor)
                bk = equipment_key(venue, boat)

                updates.append(
                    (racer, lane, finish, mk, bk, residual)
                )

            # 当日の全レースの期待確率を確定後に更新
            for racer, lane, finish, mk, bk, residual in updates:
                ability.add(day, racer, lane, finish)
                equipment.add(day, mk, bk, residual)

            if day.day == 1:
                print("Processed:", day)

    conn.close()

    print("\n=== PHASE 8.3-B ABILITY COMPLETE ===")
    print("History rows:", processed)
    print("Invalid finishes:", invalid)
    print("Target rows:", output_count)
    print("Columns:", len(columns))
    print("Output:", OUT)


if __name__ == "__main__":
    main()
