
import csv
from pathlib import Path
from collections import Counter

ROOT = Path(r"C:\boatrace_ai")

CANDIDATES = ROOT / "data/motor_reset_candidates_2026.csv"
SOURCE = ROOT / "outputs/phase83c_all_venues_preview.csv"

REGISTRY = ROOT / "data/motor_reset_registry_verified_v3.csv"
OUTPUT = ROOT / "outputs/phase83c_verified_only_v3.csv"

VERIFIED = {
    15: {
        "date": "2026-09-17",
        "source": "marugame_official_motor_data"
    },
    17: {
        "date": "2026-10-20",
        "source": "https://www.boatrace-miyajima.com/"
    },
    23: {
        "date": "2026-09-16",
        "source": "karatsu_official_motor_data"
    }
}

WINDOWS = (30, 90, 180)
METRICS = (
    "starts", "win_rate", "top2_rate",
    "top3_rate", "top2_smoothed"
)

RESET_COLS = [
    f"reset_motor_{w}_{metric}"
    for w in WINDOWS
    for metric in METRICS
]

KEYS = (
    "race_date", "venue_code", "race_no", "lane"
)

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])

def save_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

candidates, _ = read_csv(CANDIDATES)
source_rows, source_columns = read_csv(SOURCE)

if len(candidates) != 24:
    raise ValueError("Expected 24 venue records")

if not source_rows:
    raise ValueError("Source CSV is empty")

for col in RESET_COLS:
    if col not in source_columns:
        raise ValueError(f"Missing column: {col}")

registry = {}
registry_rows = []

for row in candidates:
    venue = int(row["venue_code"])

    if venue in registry:
        raise ValueError(f"Duplicate venue: {venue}")

    verified = VERIFIED.get(venue)
    start = row["motor_start_date"]

    if verified and start != verified["date"]:
        raise ValueError(f"Date mismatch: {venue}")

    status = "confirmed" if verified else "candidate"

    item = {
        "venue_code": venue,
        "venue_name": row["venue_name"],
        "motor_start_date": start,
        "status": status,
        "source": verified["source"] if verified else ""
    }

    registry[venue] = item
    registry_rows.append(item)

result = []
counts = Counter()
seen = set()

for original in source_rows:
    row = dict(original)
    venue = int(row["venue_code"])
    info = registry[venue]

    key = tuple(row[k] for k in KEYS)

    if key in seen:
        raise ValueError(f"Duplicate race key: {key}")
    seen.add(key)

    confirmed = info["status"] == "confirmed"

    if confirmed:
        if row["reset_motor_start_date"] != info["motor_start_date"]:
            raise ValueError(f"Date mismatch: {key}")

        counts["confirmed"] += 1
    else:
        # 未確認の交換日で計算した特徴量は使用禁止
        for col in RESET_COLS:
            row[col] = ""

        counts["candidate"] += 1

    row["motor_date_verified"] = int(confirmed)
    row["motor_date_source"] = info["source"]

    result.append(row)

save_csv(
    REGISTRY,
    registry_rows,
    list(registry_rows[0].keys())
)

extra = ["motor_date_verified", "motor_date_source"]

save_csv(
    OUTPUT,
    result,
    source_columns + extra
)

print("=== PHASE 8.3-C REGISTRY V3 COMPLETE ===")
print("Confirmed venues:", len(VERIFIED))
print("Candidate venues:", 24 - len(VERIFIED))
print("Total rows:", len(result))
print("Confirmed rows:", counts["confirmed"])
print("Candidate rows:", counts["candidate"])
print("Output columns:", len(source_columns + extra))
print("Registry:", REGISTRY)
print("Output:", OUTPUT)
