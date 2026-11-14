
import csv
import sqlite3
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict, deque
from itertools import groupby

ROOT = Path(r"C:\boatrace_ai")
DB = ROOT / "data" / "boatrace.db"

REGISTRY = ROOT / "data/motor_reset_registry_verified_v3.csv"
BASE = ROOT / "outputs/phase83c_verified_only_v3.csv"
OUT = ROOT / "outputs/phase83d_reset_ability_preview.csv"

START = date(2026, 10, 23)
END = date(2026, 10, 29)
HISTORY_START = START - timedelta(days=360)

WINDOWS = (30, 90, 180)
PRIOR = 1 / 3
RACER_ALPHA = 20
LANE_ALPHA = 50
RESIDUAL_ALPHA = 20


def key(row):
    return (
        str(row["race_date"]),
        str(row["venue_code"]),
        str(row["race_no"]),
        str(row["lane"])
    )


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


registry = read_csv(REGISTRY)

resets = {
    int(r["venue_code"]): date.fromisoformat(
        r["motor_start_date"]
    )
    for r in registry
    if r["status"] == "confirmed"
}

base = read_csv(BASE)

expected_keys = {
    key(r)
    for r in base
    if int(r["venue_code"]) in resets
    and r["motor_date_verified"] == "1"
}

if not expected_keys:
    raise ValueError("No verified target rows")

if len(expected_keys) != sum(
    int(r["venue_code"]) in resets
    and r["motor_date_verified"] == "1"
    for r in base
):
    raise ValueError("Duplicate target keys")


class AbilityHistory:
    def __init__(self):
        self.events = deque()
        self.racers = defaultdict(lambda: [0, 0])
        self.lanes = defaultdict(lambda: [0, 0])

    def expire(self, day):
        while self.events:
            old, racer, lane, y = self.events[0]

            if (day - old).days < 180:
                break

            self.events.popleft()

            for store, item in (
                (self.racers, racer),
                (self.lanes, lane)
            ):
                store[item][0] -= 1
                store[item][1] -= y

                if store[item][0] == 0:
                    del store[item]

    def probability(self, racer, lane):
        rn, ry = self.racers.get(racer, (0, 0))
        ln, ly = self.lanes.get(lane, (0, 0))

        rp = (ry + RACER_ALPHA * PRIOR) / (
            rn + RACER_ALPHA
        )

        lp = (ly + LANE_ALPHA * PRIOR) / (
            ln + LANE_ALPHA
        )

        return max(
            0.02, min(0.98, lp + rp - PRIOR)
        )

    def add(self, day, racer, lane, y):
        self.events.append((day, racer, lane, y))

        for store, item in (
            (self.racers, racer),
            (self.lanes, lane)
        ):
            store[item][0] += 1
            store[item][1] += y


class MotorHistory:
    def __init__(self):
        self.events = defaultdict(deque)

    def expire(self, day):
        for motor_key in list(self.events):
            q = self.events[motor_key]

            while q and (day - q[0][0]).days >= 180:
                q.popleft()

            if not q:
                del self.events[motor_key]

    def add(self, day, motor_key, residual):
        self.events[motor_key].append((day, residual))

    def features(self, day, motor_key, window):
        q = self.events.get(motor_key, ())

        values = [
            residual
            for old, residual in q
            if 0 < (day - old).days < window
        ]

        n = len(values)
        total = sum(values)

        return (
            n,
            total / n if n else "",
            total / (n + RESIDUAL_ALPHA)
        )


conn = sqlite3.connect(
    DB.as_uri() + "?mode=ro",
    uri=True
)
conn.execute("PRAGMA query_only=ON")

cursor = conn.execute("""
    SELECT
        race_date, venue_code, race_no,
        lane, racer_id, motor_no,
        finish_position
    FROM race_boat_results_v2
    WHERE race_date BETWEEN ? AND ?
    ORDER BY race_date, venue_code, race_no, lane
""", (
    HISTORY_START.isoformat(),
    END.isoformat()
))

ability = AbilityHistory()
motors = MotorHistory()

output = []
processed = 0
invalid = 0

for day_string, group in groupby(
    cursor, key=lambda r: r[0]
):
    day = date.fromisoformat(day_string)
    rows = list(group)

    ability.expire(day)
    motors.expire(day)

    # 当日の結果を加える前に特徴量を生成
    if START <= day <= END:
        for row in rows:
            d, venue, race, lane, racer, motor, finish = row

            if venue not in resets:
                continue

            mk = (venue, motor)
            p = ability.probability(racer, lane)

            record = {
                "race_date": d,
                "venue_code": venue,
                "race_no": race,
                "lane": lane,
                "racer_id": racer,
                "motor_no": motor,
                "expected_top2_probability": p,
                "motor_start_date": resets[venue].isoformat()
            }

            for window in WINDOWS:
                n, raw, smooth = motors.features(
                    day, mk, window
                )

                prefix = f"motor_reset_ability_{window}"

                record[f"{prefix}_starts"] = n
                record[f"{prefix}_residual"] = raw
                record[f"{prefix}_smoothed"] = smooth

            output.append(record)

    # 当日全レースの期待確率を先に確定
    updates = []

    for row in rows:
        d, venue, race, lane, racer, motor, finish = row
        processed += 1

        if finish is None or not 1 <= finish <= 6:
            invalid += 1
            continue

        if racer is None or lane is None:
            continue

        p = ability.probability(racer, lane)
        y = int(finish <= 2)
        residual = y - p

        updates.append(
            (venue, racer, lane, motor, y, residual)
        )

    # 同日の結果が期待確率に入らないよう更新
    for venue, racer, lane, motor, y, residual in updates:
        ability.add(day, racer, lane, y)

        # モーター残差は確認済みの交換後のみ
        if (
            venue in resets
            and day >= resets[venue]
            and motor is not None
            and motor > 0
        ):
            motors.add(
                day, (venue, motor), residual
            )

    if day.day == 1:
        print("Processed:", day)

conn.close()

actual_keys = {
    key(r) for r in output
}

if len(actual_keys) != len(output):
    raise ValueError("Duplicate output keys")

missing = expected_keys - actual_keys
extra = actual_keys - expected_keys

print("\n=== KEY CHECK ===")
print("Expected:", len(expected_keys))
print("Actual:", len(output))
print("Missing:", len(missing))
print("Extra:", len(extra))

if missing or extra:
    raise ValueError(
        f"Key mismatch: missing={list(missing)[:5]}, "
        f"extra={list(extra)[:5]}"
    )

if not output:
    raise ValueError("No output rows")

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(output[0].keys())
    )
    writer.writeheader()
    writer.writerows(output)

print("\n=== PHASE 8.3-D RESET ABILITY COMPLETE ===")
print("History rows processed:", processed)
print("Invalid finishes:", invalid)
print("Confirmed venues:", len(resets))
print("Output rows:", len(output))
print("Output columns:", len(output[0]))
print("Output:", OUT)
