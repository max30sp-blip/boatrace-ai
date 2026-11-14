
import csv
import math
import sqlite3
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"C:\boatrace_ai")
DB = ROOT / "data" / "boatrace.db"
INPUT = ROOT / "outputs" / "phase83c_verified_only_v3.csv"
REPORT = ROOT / "outputs" / "phase83c_smoothing_results.csv"

ALPHAS = (5, 10, 20, 50, 100)
PRIOR = 1 / 3
WINDOWS = (30, 90, 180)

VENUES = {
    15: "丸亀",
    17: "宮島",
    23: "唐津"
}


def make_key(row):
    return (
        str(row["race_date"]),
        int(row["venue_code"]),
        int(row["race_no"]),
        int(row["lane"])
    )


def clip(p):
    return max(1e-12, min(1 - 1e-12, p))


def score(predictions):
    n = len(predictions)

    if n == 0:
        return None

    brier = sum(
        (p - y) ** 2 for p, y in predictions
    ) / n

    logloss = -sum(
        y * math.log(clip(p))
        + (1 - y) * math.log(1 - clip(p))
        for p, y in predictions
    ) / n

    return brier, logloss


conn = sqlite3.connect(
    DB.as_uri() + "?mode=ro",
    uri=True
)
conn.execute("PRAGMA query_only=ON")

# 予測対象期間の結果を取得
result_rows = conn.execute("""
    SELECT
        race_date, venue_code, race_no,
        lane, finish_position
    FROM race_boat_results_v2
    WHERE race_date BETWEEN '2026-10-23'
                        AND '2026-10-29'
""").fetchall()

conn.close()

outcomes = {}

for d, venue, race, lane, finish in result_rows:
    key = (d, venue, race, lane)

    if key in outcomes:
        raise ValueError(f"Duplicate result: {key}")

    outcomes[key] = finish

# (venue, window, alpha) -> [(prediction, actual), ...]
results = defaultdict(list)
counts = defaultdict(int)

with INPUT.open(
    encoding="utf-8-sig", newline=""
) as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["motor_date_verified"] != "1":
            continue

        key = make_key(row)
        finish = outcomes.get(key)

        if finish is None or not 1 <= finish <= 6:
            counts["invalid_finish"] += 1
            continue

        venue = int(row["venue_code"])

        if venue not in VENUES:
            raise ValueError(
                f"Unexpected verified venue: {venue}"
            )

        actual = int(finish <= 2)

        for window in WINDOWS:
            prefix = f"reset_motor_{window}"

            n = int(row[f"{prefix}_starts"])
            raw_rate = row[f"{prefix}_top2_rate"]

            if n < 0:
                raise ValueError(f"Negative starts: {key}")

            if n == 0:
                top2_count = 0
            else:
                if raw_rate == "":
                    raise ValueError(
                        f"Missing rate: {key}"
                    )

                rate = float(raw_rate)

                if not 0 <= rate <= 1:
                    raise ValueError(
                        f"Invalid rate: {key}"
                    )

                # CSV内の浮動小数点誤差を考慮
                estimated = rate * n
                top2_count = round(estimated)

                if abs(estimated - top2_count) > 1e-6:
                    raise ValueError(
                        f"Inconsistent rate: {key}"
                    )

            for alpha in ALPHAS:
                probability = (
                    top2_count + alpha * PRIOR
                ) / (n + alpha)

                results[(venue, window, alpha)].append(
                    (probability, actual)
                )

        counts["valid_rows"] += 1


print("=== PHASE 8.3-C SMOOTHING TEST ===")
print("Valid verified rows:", counts["valid_rows"])
print("Invalid finishes:", counts["invalid_finish"])

summary = []

for (venue, window, alpha), predictions in sorted(
    results.items()
):
    metrics = score(predictions)

    if metrics is None:
        continue

    brier, logloss = metrics

    summary.append({
        "venue_code": venue,
        "venue_name": VENUES[venue],
        "window": window,
        "alpha": alpha,
        "rows": len(predictions),
        "brier": brier,
        "logloss": logloss
    })


REPORT.parent.mkdir(
    parents=True, exist_ok=True
)

with REPORT.open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(summary[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary)


print("\n=== OVERALL COMPARISON ===")

overall = []

for window in WINDOWS:
    for alpha in ALPHAS:
        all_predictions = []

        for venue in VENUES:
            all_predictions.extend(
                results.get(
                    (venue, window, alpha), []
                )
            )

        brier, logloss = score(all_predictions)

        overall.append((
            window, alpha, len(all_predictions),
            brier, logloss
        ))

        print(
            f"window={window:3d}",
            f"alpha={alpha:3d}",
            f"n={len(all_predictions):4d}",
            f"Brier={brier:.6f}",
            f"LogLoss={logloss:.6f}"
        )


print("\n=== BEST SETTINGS BY VENUE ===")

for venue, name in VENUES.items():
    venue_rows = [
        r for r in summary
        if r["venue_code"] == venue
    ]

    if not venue_rows:
        print(name, "No valid results")
        continue

    best = min(
        venue_rows, key=lambda r: r["brier"]
    )

    print(
        name,
        f"window={best['window']}",
        f"alpha={best['alpha']}",
        f"Brier={best['brier']:.6f}",
        f"LogLoss={best['logloss']:.6f}"
    )


best_overall = min(
    overall, key=lambda r: r[3]
)

print("\n=== BEST OVERALL (EXPLORATORY) ===")
print("Window:", best_overall[0])
print("Alpha:", best_overall[1])
print("Brier:", round(best_overall[3], 6))
print("LogLoss:", round(best_overall[4], 6))

baseline = score([
    (PRIOR, y)
    for p, y in results[(15, 180, ALPHAS[0])]
] + [
    (PRIOR, y)
    for venue in (17, 23)
    for p, y in results[(venue, 180, ALPHAS[0])]
])

print("\n=== CONSTANT PRIOR BASELINE ===")
print("Brier:", round(baseline[0], 6))
print("LogLoss:", round(baseline[1], 6))

print("\n=== PHASE 8.3-C SMOOTHING COMPLETE ===")
print("Report:", REPORT)
