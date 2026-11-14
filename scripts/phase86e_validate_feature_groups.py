
import csv
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:\boatrace_ai")
DATE = "2026-10-29"

PROGRAM_DB = ROOT / "data/boatrace.db"
TRAIN_DB = ROOT / "data/train_dataset_v1.db"
MODEL_REPORT = ROOT / "models/phase08_1_b_report.json"

OUT_DIR = ROOT / "outputs/phase86e"
SUMMARY_FILE = OUT_DIR / f"summary_{DATE}.json"
DIFF_FILE = OUT_DIR / f"differences_{DATE}.csv"

SOURCES = {
    "f1": (
        ROOT / "data/features_fast.db",
        "racer_course_features_fast",
    ),
    "f2": (
        ROOT / "data/lane_features.db",
        "racer_lane_features_fast",
    ),
    "f3": (
        ROOT / "data/motor_features.db",
        "equipment_features_fast",
    ),
    "f4": (
        ROOT / "data/venue_detail_features.db",
        "venue_detail_features_fast",
    ),
    "f5": (
        ROOT / "data/recent_form_features.db",
        "racer_recent_form_fast",
    ),
    "f6": (
        ROOT / "data/meet_features.db",
        "racer_meet_features",
    ),
}

GROUPS = tuple(SOURCES)


def readonly(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    con = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True,
    )
    con.execute("PRAGMA query_only=ON")
    return con


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def numeric(value):
    if value is None:
        return None

    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def equal(a, b):
    a = numeric(a)
    b = numeric(b)

    if a is None or b is None:
        return a is None and b is None

    return math.isclose(
        a, b, rel_tol=0.0, abs_tol=1e-6
    )


def load_model_groups():
    with MODEL_REPORT.open(
        encoding="utf-8-sig"
    ) as f:
        report = json.load(f)

    names = report["feature_names"]

    if len(names) != 618:
        raise ValueError("Expected 618 model features")

    groups = defaultdict(list)

    for name in names:
        parts = name.split("_", 2)

        if len(parts) != 3:
            raise ValueError(f"Bad feature: {name}")

        boat, group, suffix = parts

        if group in SOURCES:
            groups[group].append(
                (boat, suffix, name)
            )

    for group in GROUPS:
        one_boat = [
            suffix
            for boat, suffix, _ in groups[group]
            if boat == "b1"
        ]

        if len(one_boat) * 6 != len(groups[group]):
            raise ValueError(
                f"Unexpected feature layout: {group}"
            )

    return groups


def load_programs():
    con = readonly(PROGRAM_DB)

    cur = con.execute("""
        SELECT
            venue_code, race_no, lane, racer_id
        FROM program_entries
        WHERE race_date = ?
        ORDER BY venue_code, race_no, lane
    """, (DATE,))

    programs = {}

    for venue, race, lane, racer in cur:
        key = (int(venue), int(race), int(lane))

        if key in programs:
            raise ValueError(
                f"Duplicate program key: {key}"
            )

        programs[key] = (
            int(racer) if racer is not None else None
        )

    con.close()

    if not programs:
        raise ValueError("No program entries")

    print("Program boats:", len(programs))
    return programs


def load_references():
    con = readonly(TRAIN_DB)

    cur = con.execute("""
        SELECT *
        FROM train_races
        WHERE race_date = ?
    """, (DATE,))

    names = [d[0] for d in cur.description]
    references = {}

    for values in cur:
        row = dict(zip(names, values))
        key = (
            int(row["venue_code"]),
            int(row["race_no"]),
        )

        if key in references:
            raise ValueError(
                f"Duplicate training race: {key}"
            )

        references[key] = row

    con.close()

    if not references:
        raise ValueError("No training reference races")

    print("Reference races:", len(references))
    return references


def load_source(group, features, programs):
    db_path, table = SOURCES[group]
    con = readonly(db_path)

    schema = {
        r[1]
        for r in con.execute(
            f"PRAGMA table_info({quote(table)})"
        )
    }

    required = {
        "race_date",
        "venue_code",
        "race_no",
    }

    if not required <= schema:
        raise ValueError(
            f"Missing race keys: {group}"
        )

    if "lane" in schema:
        boat_key = "lane"
    elif "boat_no" in schema:
        boat_key = "boat_no"
    else:
        raise ValueError(
            f"No lane/boat key: {group}"
        )

    suffixes = [
        suffix
        for boat, suffix, _ in features
        if boat == "b1"
    ]

    if len(set(suffixes)) != len(suffixes):
        raise ValueError(
            f"Duplicate suffixes: {group}"
        )

    mapping = {}

    for suffix in suffixes:
        candidates = (
            suffix,
            f"{group}_{suffix}",
        )

        field = next(
            (c for c in candidates if c in schema),
            None
        )

        if field is None:
            raise ValueError(
                f"Source column not found: "
                f"{group} / {suffix}"
            )

        mapping[suffix] = field

    has_racer_id = "racer_id" in schema

    selected = [
        "venue_code",
        "race_no",
        boat_key,
    ]

    if has_racer_id:
        selected.append("racer_id")

    selected.extend(mapping.values())

    query = (
        "SELECT "
        + ", ".join(quote(c) for c in selected)
        + f" FROM {quote(table)}"
        + " WHERE race_date = ?"
    )

    result = {}
    duplicate_count = 0
    unmatched_keys = 0
    racer_mismatches = []

    for raw in con.execute(query, (DATE,)):
        row = dict(zip(selected, raw))

        if any(
            row[k] is None
            for k in ("venue_code", "race_no", boat_key)
        ):
            unmatched_keys += 1
            continue

        key = (
            int(row["venue_code"]),
            int(row["race_no"]),
            int(row[boat_key]),
        )

        if key in result:
            duplicate_count += 1
            continue

        if key not in programs:
            unmatched_keys += 1
            continue

        if has_racer_id:
            expected_racer = programs[key]
            actual_racer = row["racer_id"]

            if (
                expected_racer is not None
                and actual_racer is not None
                and int(actual_racer) != expected_racer
            ):
                racer_mismatches.append({
                    "key": key,
                    "program_racer": expected_racer,
                    "source_racer": actual_racer,
                })
                continue

        result[key] = {
            suffix: row[column]
            for suffix, column in mapping.items()
        }

    con.close()

    if duplicate_count:
        raise ValueError(
            f"{group}: {duplicate_count} duplicate keys"
        )

    if racer_mismatches:
        raise ValueError(
            f"{group}: racer ID mismatches. "
            f"Examples: {racer_mismatches[:5]}"
        )

    print(
        f"{group}:",
        f"{len(result)}/{len(programs)} boat rows",
        f"fields={len(mapping)}",
        f"unmatched_keys={unmatched_keys}",
        f"key={boat_key}",
    )

    return result, mapping


def main():
    print("=== PHASE 8.6-E FEATURE VALIDATION ===")
    print("Date:", DATE)

    if SUMMARY_FILE.exists() or DIFF_FILE.exists():
        raise FileExistsError(
            "Existing phase86e output. "
            "Keep the previous results before re-running."
        )

    groups = load_model_groups()
    programs = load_programs()
    references = load_references()

    results = {}
    differences = []

    for group in GROUPS:
        print(f"\n=== {group.upper()} ===")

        source, mapping = load_source(
            group, groups[group], programs
        )

        statistics = Counter()
        by_feature = defaultdict(Counter)
        source_present = 0

        for key in sorted(programs):
            venue, race, lane = key

            ref = references.get((venue, race))

            if ref is None:
                raise ValueError(
                    f"No reference race: {(venue, race)}"
                )

            values = source.get(key)

            if values is None:
                statistics["missing_source_rows"] += 1
            else:
                source_present += 1

            for suffix in mapping:
                name = f"b{lane}_{group}_{suffix}"

                if name not in ref:
                    raise ValueError(
                        f"Missing reference column: {name}"
                    )

                expected = ref[name]

                actual = (
                    values.get(suffix)
                    if values is not None else None
                )

                statistics["compared"] += 1
                by_feature[suffix]["compared"] += 1

                if equal(actual, expected):
                    statistics["matched"] += 1
                    by_feature[suffix]["matched"] += 1
                    continue

                statistics["mismatched"] += 1
                by_feature[suffix]["mismatched"] += 1

                differences.append({
                    "group": group,
                    "venue_code": venue,
                    "race_no": race,
                    "lane": lane,
                    "racer_id": programs[key],
                    "feature": name,
                    "computed": actual,
                    "reference": expected,
                    "source_row_missing": values is None,
                })

        count = statistics["compared"]
        matched = statistics["matched"]
        rate = 100 * matched / count if count else 0

        print(
            "Source boat coverage:",
            f"{source_present}/{len(programs)}"
        )

        print(
            "Matching values:",
            f"{matched}/{count}"
        )

        print(
            "Match rate:",
            f"{rate:.2f}%"
        )

        if statistics["mismatched"]:
            ranked = sorted(
                by_feature.items(),
                key=lambda x: (
                    -x[1]["mismatched"],
                    x[0]
                )
            )

            print("Most mismatched features:")

            for suffix, counts in ranked[:5]:
                if counts["mismatched"]:
                    print(
                        f"  {suffix}:",
                        counts["mismatched"]
                    )

        results[group] = {
            "database": str(SOURCES[group][0]),
            "table": SOURCES[group][1],
            "program_boats": len(programs),
            "source_boats": source_present,
            "feature_columns": len(groups[group]),
            "compared": count,
            "matched": matched,
            "mismatched": statistics["mismatched"],
            "match_rate": matched / count if count else None,
            "source_rows_missing": statistics[
                "missing_source_rows"
            ],
            "mismatched_features": {
                suffix: counts["mismatched"]
                for suffix, counts in by_feature.items()
                if counts["mismatched"]
            },
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with DIFF_FILE.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        columns = [
            "group", "venue_code", "race_no",
            "lane", "racer_id", "feature",
            "computed", "reference",
            "source_row_missing",
        ]

        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(differences)

    summary = {
        "date": DATE,
        "model": "phase08_1_b",
        "reference_for_validation_only": True,
        "feature_groups": results,
        "prediction_ready": False,
        "notes": [
            "A matching historical value does not prove live availability.",
            "Do not use post-race features in live predictions.",
            "Historical DBs are read-only.",
        ],
    }

    with SUMMARY_FILE.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            summary, f,
            indent=2,
            ensure_ascii=False
        )

    print("\n=== FINAL SUMMARY ===")

    total_matched = 0
    total_compared = 0

    for group, item in results.items():
        total_matched += item["matched"]
        total_compared += item["compared"]

        print(
            group.upper(),
            f"{item['matched']}/{item['compared']}",
            f"{100 * item['match_rate']:.2f}%"
        )

    overall = (
        100 * total_matched / total_compared
        if total_compared else 0
    )

    print(
        "Overall F1-F6:",
        f"{total_matched}/{total_compared}",
        f"{overall:.2f}%"
    )

    print("\n=== PHASE 8.6-E VALIDATION COMPLETE ===")
    print("Summary:", SUMMARY_FILE)
    print("Differences:", DIFF_FILE)


if __name__ == "__main__":
    main()
