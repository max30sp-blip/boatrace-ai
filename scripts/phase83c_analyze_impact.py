
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"C:\boatrace_ai")
INPUT = ROOT / "outputs" / "phase83c_all_venues_preview.csv"
REPORT = ROOT / "outputs" / "phase83c_impact_analysis.csv"

VENUES = {
    1: "桐生", 2: "戸田", 3: "江戸川",
    4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津",
    10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀",
    16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋",
    22: "福岡", 23: "唐津", 24: "大村"
}

stats = defaultdict(lambda: {
    "rows": 0,
    "changed": 0,
    "old_starts": 0,
    "new_starts": 0,
    "abs_diff": 0.0,
    "max_diff": 0.0,
    "positive": 0,
    "negative": 0,
    "start_date": "",
    "status": ""
})

with INPUT.open(
    encoding="utf-8-sig", newline=""
) as f:
    reader = csv.DictReader(f)

    for row in reader:
        venue = int(row["venue_code"])
        s = stats[venue]

        old_n = int(row["motor_180_starts"])
        new_n = int(row["reset_motor_180_starts"])

        old_rate = float(
            row["motor_180_top2_smoothed"]
        )
        new_rate = float(
            row["reset_motor_180_top2_smoothed"]
        )

        diff = new_rate - old_rate
        absolute = abs(diff)

        s["rows"] += 1
        s["old_starts"] += old_n
        s["new_starts"] += new_n
        s["abs_diff"] += absolute
        s["max_diff"] = max(s["max_diff"], absolute)

        if absolute > 1e-9:
            s["changed"] += 1

        if diff > 1e-9:
            s["positive"] += 1
        elif diff < -1e-9:
            s["negative"] += 1

        s["start_date"] = row["reset_motor_start_date"]
        s["status"] = row["reset_evidence_status"]

results = []

for venue in range(1, 25):
    s = stats[venue]
    n = s["rows"]

    if n == 0:
        continue

    results.append({
        "venue_code": venue,
        "venue_name": VENUES[venue],
        "rows": n,
        "changed_rows": s["changed"],
        "changed_ratio": s["changed"] / n,
        "avg_old_starts": s["old_starts"] / n,
        "avg_new_starts": s["new_starts"] / n,
        "mean_absolute_diff": s["abs_diff"] / n,
        "max_absolute_diff": s["max_diff"],
        "positive_changes": s["positive"],
        "negative_changes": s["negative"],
        "start_date": s["start_date"],
        "evidence_status": s["status"]
    })

results.sort(
    key=lambda r: r["mean_absolute_diff"],
    reverse=True
)

REPORT.parent.mkdir(parents=True, exist_ok=True)

with REPORT.open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

print("=== PHASE 8.3-C IMPACT ANALYSIS ===")
print("Venues with races:", len(results))
print("Venues without races:", 24 - len(results))

print("\n=== TOP 15 VENUES BY IMPACT ===")

for rank, row in enumerate(results[:15], 1):
    print(
        f"{rank:2d}. {row['venue_name']:5s}",
        f"rows={row['rows']:3d}",
        f"changed={row['changed_ratio']:.1%}",
        f"old_n={row['avg_old_starts']:.1f}",
        f"new_n={row['avg_new_starts']:.1f}",
        f"MAE={row['mean_absolute_diff']:.4f}",
        f"MAX={row['max_absolute_diff']:.4f}",
        f"status={row['evidence_status']}"
    )

total_rows = sum(r["rows"] for r in results)
total_changed = sum(r["changed_rows"] for r in results)
weighted_mae = sum(
    r["mean_absolute_diff"] * r["rows"]
    for r in results
) / total_rows

print("\n=== OVERALL SUMMARY ===")
print("Total rows:", total_rows)
print("Changed rows:", total_changed)
print("Changed ratio:", round(total_changed / total_rows, 4))
print("Weighted mean absolute difference:", round(weighted_mae, 6))
print("Report:", REPORT)
print("\n=== PHASE 8.3-C IMPACT COMPLETE ===")
