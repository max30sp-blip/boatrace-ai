
import csv
import math
from pathlib import Path
from collections import Counter

ROOT = Path(r"C:\boatrace_ai")
OUTPUTS = ROOT / "outputs"

BASE = OUTPUTS / "phase83b_f3_ability_preview.csv"
ADJUSTED = OUTPUTS / "phase83b_ability_adjusted.csv"
FINAL = OUTPUTS / "phase83b_f3_final_preview.csv"

KEYS = (
    "race_date", "venue_code", "race_no",
    "lane", "racer_id"
)

WINDOWS = (30, 90, 180)


def load_csv(path):
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames

    if not columns:
        raise ValueError(f"Empty CSV: {path}")

    return rows, columns


def make_key(row):
    return tuple(row[c] for c in KEYS)


def build_index(rows):
    index = {}
    for row in rows:
        key = make_key(row)

        if key in index:
            raise ValueError(f"Duplicate key: {key}")

        index[key] = row

    return index


base_rows, base_columns = load_csv(BASE)
adjusted_rows, adjusted_columns = load_csv(ADJUSTED)

base_index = build_index(base_rows)
adjusted_index = build_index(adjusted_rows)

print("=== 1. KEY VALIDATION ===")
print("Base rows:", len(base_rows))
print("Adjusted rows:", len(adjusted_rows))

missing = set(base_index) - set(adjusted_index)
extra = set(adjusted_index) - set(base_index)

print("Missing adjusted:", len(missing))
print("Extra adjusted:", len(extra))

if missing or extra:
    raise ValueError("Feature keys do not match")


print("\n=== 2. NUMERIC VALIDATION ===")

errors = Counter()
statistics = {}

for kind in ("motor", "boat"):
    for window in WINDOWS:
        name = f"{kind}_adjusted_{window}"
        statistics[name] = []

for row in adjusted_rows:
    try:
        expected = float(
            row["expected_top2_probability"]
        )

        if not math.isfinite(expected) or not 0.02 <= expected <= 0.98:
            errors["invalid_expected_probability"] += 1

        for kind in ("motor", "boat"):
            prev_starts = None

            for window in WINDOWS:
                prefix = f"{kind}_adjusted_{window}"

                starts = int(row[f"{prefix}_starts"])
                raw = row[f"{prefix}_residual"]
                smooth = row[f"{prefix}_smoothed"]

                if starts < 0:
                    errors["negative_starts"] += 1

                if prev_starts is not None and starts < prev_starts:
                    errors["nonmonotonic_starts"] += 1

                prev_starts = starts

                if starts == 0:
                    if raw != "" or smooth != "":
                        errors["zero_starts_with_values"] += 1
                    continue

                if raw == "" or smooth == "":
                    errors["missing_adjustment"] += 1
                    continue

                avg = float(raw)
                smoothed = float(smooth)

                if not (
                    math.isfinite(avg)
                    and math.isfinite(smoothed)
                    and -1 <= avg <= 1
                    and -1 <= smoothed <= 1
                ):
                    errors["invalid_adjustment"] += 1

                expected_smooth = (
                    avg * starts / (starts + 20)
                )

                if abs(smoothed - expected_smooth) > 1e-9:
                    errors["smoothing_mismatch"] += 1

                statistics[prefix].append(smoothed)

    except (ValueError, TypeError, KeyError) as e:
        errors["parse_errors"] += 1
        print("Parse error:", e)
        if errors["parse_errors"] >= 5:
            raise


print("Errors:", dict(errors))

if errors:
    raise ValueError("Validation failed")


print("\n=== 3. ADJUSTMENT DISTRIBUTION ===")

for name, values in statistics.items():
    if not values:
        print(name, "NO DATA")
        continue

    print(
        name,
        "count=", len(values),
        "mean=", round(sum(values) / len(values), 5),
        "min=", round(min(values), 5),
        "max=", round(max(values), 5)
    )


print("\n=== 4. FEATURE MERGE ===")

new_columns = [
    c for c in adjusted_columns if c not in KEYS
]

overlap = set(base_columns) & set(new_columns)

if overlap:
    raise ValueError(
        f"Overlapping columns: {sorted(overlap)}"
    )

final_columns = base_columns + new_columns

FINAL.parent.mkdir(parents=True, exist_ok=True)

with FINAL.open(
    "w", encoding="utf-8-sig", newline=""
) as f:

    writer = csv.DictWriter(
        f, fieldnames=final_columns
    )
    writer.writeheader()

    for row in base_rows:
        key = make_key(row)
        adjusted = adjusted_index[key]

        merged = dict(row)

        for col in new_columns:
            merged[col] = adjusted[col]

        writer.writerow(merged)


print("\n=== PHASE 8.3-B FINALIZE COMPLETE ===")
print("Output rows:", len(base_rows))
print("Output columns:", len(final_columns))
print("Output:", FINAL)
