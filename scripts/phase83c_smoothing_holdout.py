
import csv
import math
import sqlite3
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"C:\boatrace_ai")
DB = ROOT / "data" / "boatrace.db"
INPUT = ROOT / "outputs" / "phase83c_verified_only_v3.csv"
OUTPUT = ROOT / "outputs" / "phase83c_smoothing_holdout.csv"

ALPHAS = (5, 10, 20, 50, 100)
WINDOWS = (30, 90, 180)
PRIOR = 1 / 3
SPLIT_DATE = "2026-10-27"

def key(row):
    return (
        row["race_date"],
        int(row["venue_code"]),
        int(row["race_no"]),
        int(row["lane"])
    )

def evaluate(pairs):
    if not pairs:
        return None

    n = len(pairs)
    brier = 0
    loss = 0

    for p, y in pairs:
        p = max(1e-12, min(1 - 1e-12, p))
        brier += (p - y) ** 2
        loss -= (
            y * math.log(p)
            + (1 - y) * math.log(1 - p)
        )

    return brier / n, loss / n

conn = sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)
conn.execute("PRAGMA query_only=ON")

outcomes = {}

for d, v, r, lane, finish in conn.execute("""
    SELECT race_date, venue_code, race_no,
           lane, finish_position
    FROM race_boat_results_v2
    WHERE race_date BETWEEN '2026-10-23'
                        AND '2026-10-29'
"""):
    k = (d, v, r, lane)

    if k in outcomes:
        raise ValueError(f"Duplicate result: {k}")

    outcomes[k] = finish

conn.close()

# train / test -> setting -> prediction pairs
scores = {
    "train": defaultdict(list),
    "test": defaultdict(list)
}

seen = set()
counts = defaultdict(int)

with INPUT.open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        if row["motor_date_verified"] != "1":
            continue

        k = key(row)

        if k in seen:
            raise ValueError(f"Duplicate feature: {k}")
        seen.add(k)

        finish = outcomes.get(k)

        if finish is None or not 1 <= finish <= 6:
            counts["invalid"] += 1
            continue

        part = (
            "train"
            if row["race_date"] < SPLIT_DATE
            else "test"
        )

        actual = int(finish <= 2)
        scores[part]["baseline"].append((PRIOR, actual))

        for window in WINDOWS:
            prefix = f"reset_motor_{window}"

            n = int(row[f"{prefix}_starts"])
            rate_str = row[f"{prefix}_top2_rate"]

            if n < 0:
                raise ValueError(f"Negative starts: {k}")

            if n == 0:
                successes = 0
            else:
                if rate_str == "":
                    raise ValueError(f"Missing rate: {k}")

                estimated = float(rate_str) * n
                successes = round(estimated)

                if abs(estimated - successes) > 1e-6:
                    raise ValueError(f"Invalid rate: {k}")

                if not 0 <= successes <= n:
                    raise ValueError(f"Invalid successes: {k}")

            for alpha in ALPHAS:
                p = (
                    successes + alpha * PRIOR
                ) / (n + alpha)

                scores[part][(window, alpha)].append(
                    (p, actual)
                )

        counts[part] += 1

settings = [
    (window, alpha)
    for window in WINDOWS
    for alpha in ALPHAS
]

# 設定選択には前半データだけを使用する
selected = min(
    settings,
    key=lambda s: evaluate(scores["train"][s])[0]
)

baseline_setting = (180, 20)

print("=== PHASE 8.3-C HOLDOUT TEST ===")
print("Train rows:", counts["train"])
print("Test rows:", counts["test"])
print("Invalid rows:", counts["invalid"])
print("Selected on train:", selected)

report = []

for part in ("train", "test"):
    print(f"\n=== {part.upper()} RESULTS ===")

    for label, setting in [
        ("constant_prior", "baseline"),
        ("old_alpha20", baseline_setting),
        ("selected", selected)
    ]:
        pairs = scores[part][setting]
        metrics = evaluate(pairs)

        if metrics is None:
            raise ValueError(f"No observations: {part}")

        brier, logloss = metrics

        print(
            f"{label:17s}",
            f"n={len(pairs):4d}",
            f"Brier={brier:.6f}",
            f"LogLoss={logloss:.6f}"
        )

        report.append({
            "split": part,
            "label": label,
            "window": (
                "" if setting == "baseline"
                else setting[0]
            ),
            "alpha": (
                "" if setting == "baseline"
                else setting[1]
            ),
            "rows": len(pairs),
            "brier": brier,
            "logloss": logloss
        })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(report[0].keys())
    )
    writer.writeheader()
    writer.writerows(report)

print("\n=== HOLDOUT IMPROVEMENT ===")

old = evaluate(
    scores["test"][baseline_setting]
)[0]

new = evaluate(
    scores["test"][selected]
)[0]

print("Brier improvement:", round(old - new, 6))

if new < old:
    print("Result: selected setting better on holdout")
elif new > old:
    print("Result: old setting better on holdout")
else:
    print("Result: equal")

print("\n=== PHASE 8.3-C HOLDOUT COMPLETE ===")
print("Output:", OUTPUT)
