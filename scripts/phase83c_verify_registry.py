
import csv
from pathlib import Path
from collections import Counter

ROOT = Path(r"C:\boatrace_ai")
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

CANDIDATES = DATA / "motor_reset_candidates_2026.csv"
SOURCE = OUTPUTS / "phase83c_all_venues_preview.csv"

REGISTRY = DATA / "motor_reset_registry_verified_v2.csv"
SAFE = OUTPUTS / "phase83c_verified_only_preview.csv"

# 使用開始日そのものを公式サイトで確認した場
VERIFIED = {
    15: {
        "date": "2026-09-17",
        "url": (
            "https://www.marugameboat.jp/asp/htmlmade/"
            "marugame/sp/motor/motor01.htm"
        )
    },
    23: {
        "date": "2026-09-16",
        "url": (
            "https://www.boatrace-karatsu.jp/"
            "modules/datafile/?kind=4"
        )
    }
}

KEY_COLUMNS = (
    "race_date", "venue_code", "race_no", "lane"
)

METRICS = (
    "starts", "win_rate", "top2_rate",
    "top3_rate", "top2_smoothed"
)

RESET_COLUMNS = [
    f"reset_motor_{window}_{metric}"
    for window in (30, 90, 180)
    for metric in METRICS
]

# まず日付候補の台帳を読み込む
with CANDIDATES.open(
    encoding="utf-8-sig", newline=""
) as f:
    candidates = list(csv.DictReader(f))

if len(candidates) != 24:
    raise ValueError("Expected 24 venue records")

registry_rows = []
seen = set()

for row in candidates:
    venue = int(row["venue_code"])

    if venue in seen:
        raise ValueError(f"Duplicate venue: {venue}")
    seen.add(venue)

    confirmed = VERIFIED.get(venue)

    if confirmed:
        if row["motor_start_date"] != confirmed["date"]:
            raise ValueError(
                f"Date mismatch for venue {venue}"
            )

    registry_rows.append({
        "venue_code": venue,
        "venue_name": row["venue_name"],
        "motor_start_date": row["motor_start_date"],
        "status": (
            "confirmed"
            if confirmed else "candidate"
        ),
        "source_url": (
            confirmed["url"]
            if confirmed else ""
        )
    })

with REGISTRY.open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(registry_rows[0])
    )
    writer.writeheader()
    writer.writerows(registry_rows)

registry = {
    int(row["venue_code"]): row
    for row in registry_rows
}

counts = Counter()
keys = set()

with SOURCE.open(
    encoding="utf-8-sig", newline=""
) as src:
    reader = csv.DictReader(src)
    columns = list(reader.fieldnames or [])

    if not columns:
        raise ValueError("Empty source CSV")

    for column in RESET_COLUMNS:
        if column not in columns:
            raise ValueError(f"Missing: {column}")

    extra = [
        "motor_date_verified",
        "motor_date_source"
    ]

    with SAFE.open(
        "w", encoding="utf-8-sig", newline=""
    ) as dst:
        writer = csv.DictWriter(
            dst, fieldnames=columns + extra
        )
        writer.writeheader()

        for row in reader:
            key = tuple(row[c] for c in KEY_COLUMNS)

            if key in keys:
                raise ValueError(f"Duplicate key: {key}")
            keys.add(key)

            venue = int(row["venue_code"])
            info = registry[venue]

            confirmed = info["status"] == "confirmed"

            row["motor_date_verified"] = int(confirmed)
            row["motor_date_source"] = info["source_url"]

            if confirmed:
                if (
                    row["reset_motor_start_date"]
                    != info["motor_start_date"]
                ):
                    raise ValueError(
                        f"Start-date mismatch: {key}"
                    )
                counts["confirmed_rows"] += 1
            else:
                # 未確認の補正値は学習に使えないよう空欄化
                for column in RESET_COLUMNS:
                    row[column] = ""

                counts["candidate_rows"] += 1

            writer.writerow(row)
            counts["total"] += 1

print("=== PHASE 8.3-C REGISTRY VERIFIED ===")
print("Registered venues:", len(registry))
print("Confirmed venues:", len(VERIFIED))
print("Candidate venues:", len(registry) - len(VERIFIED))
print("Total rows:", counts["total"])
print("Confirmed rows:", counts["confirmed_rows"])
print("Candidate rows:", counts["candidate_rows"])
print("Registry:", REGISTRY)
print("Output:", SAFE)
print("=== PHASE 8.3-C-5 COMPLETE ===")
