
import csv
import math
from pathlib import Path
from collections import Counter

ROOT = Path(r"C:\boatrace_ai")
OUTPUTS = ROOT / "outputs"

BASE = OUTPUTS / "phase83c_verified_only_v3.csv"
ABILITY = OUTPUTS / "phase83d_reset_ability_preview.csv"
OUT = OUTPUTS / "phase83d_integrated_preview.csv"

KEYS = (
    "race_date", "venue_code", "race_no",
    "lane", "racer_id", "motor_no"
)

WINDOWS = (30, 90, 180)


def load_csv(path):
    with path.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def make_key(row):
    return tuple(row[k] for k in KEYS)


base_rows, base_cols = load_csv(BASE)
ability_rows, ability_cols = load_csv(ABILITY)

ability_index = {}

for row in ability_rows:
    k = make_key(row)

    if k in ability_index:
        raise ValueError(f"Duplicate ability key: {k}")

    ability_index[k] = row

verified = [
    r for r in base_rows
    if r["motor_date_verified"] == "1"
]

verified_keys = [make_key(r) for r in verified]

if len(set(verified_keys)) != len(verified_keys):
    raise ValueError("Duplicate base keys")

missing = set(verified_keys) - set(ability_index)
extra = set(ability_index) - set(verified_keys)

print("=== PHASE 8.3-D FEATURE MERGE ===")
print("Verified base rows:", len(verified))
print("Ability rows:", len(ability_rows))
print("Missing:", len(missing))
print("Extra:", len(extra))

if missing or extra:
    raise ValueError("Join keys do not match")

new_cols = [
    "expected_top2_probability_reset"
]

for window in WINDOWS:
    prefix = f"motor_reset_ability_{window}"
    new_cols.extend([
        f"{prefix}_starts",
        f"{prefix}_residual",
        f"{prefix}_smoothed"
    ])

if set(new_cols) & set(base_cols):
    raise ValueError("Overlapping feature columns")

merged_rows = []
issues = Counter()

for original in verified:
    row = dict(original)
    a = ability_index[make_key(row)]

    if (
        row["reset_motor_start_date"]
        != a["motor_start_date"]
    ):
        issues["start_date_mismatch"] += 1

    p = float(a["expected_top2_probability"])

    if not math.isfinite(p) or not 0.02 <= p <= 0.98:
        issues["invalid_probability"] += 1

    for window in WINDOWS:
        old_n = int(
            row[f"reset_motor_{window}_starts"]
        )

        new_n = int(
            a[f"motor_reset_ability_{window}_starts"]
        )

        if old_n != new_n:
            issues[f"starts_mismatch_{window}"] += 1

        raw = a[f"motor_reset_ability_{window}_residual"]
        smooth = a[f"motor_reset_ability_{window}_smoothed"]

        if new_n < 0:
            issues["negative_starts"] += 1
        elif new_n == 0:
            if raw != "" or float(smooth) != 0.0:
                issues["invalid_zero_history"] += 1
        else:
            if raw == "" or smooth == "":
                issues["missing_residual"] += 1
            else:
                raw_value = float(raw)
                smooth_value = float(smooth)

                expected_smooth = (
                    raw_value * new_n / (new_n + 20)
                )

                if not (
                    math.isfinite(raw_value)
                    and math.isfinite(smooth_value)
                    and -1 <= raw_value <= 1
                    and abs(
                        smooth_value - expected_smooth
                    ) < 1e-9
                ):
                    issues["invalid_residual"] += 1

    for column in new_cols:
        if column == "expected_top2_probability_reset":
            row[column] = a["expected_top2_probability"]
        else:
            row[column] = a[column]

    merged_rows.append(row)

print("\n=== VALIDATION ===")
print("Issues:", dict(issues))

if issues:
    raise ValueError(
        "Validation issues detected. "
        "Integrated CSV was not written."
    )

final_cols = base_cols + new_cols

OUT.parent.mkdir(
    parents=True, exist_ok=True
)

with OUT.open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=final_cols
    )
    writer.writeheader()
    writer.writerows(merged_rows)

print("\n=== PHASE 8.3-D MERGE COMPLETE ===")
print("Rows:", len(merged_rows))
print("Columns:", len(final_cols))
print("Validation issues:", sum(issues.values()))
print("Output:", OUT)
