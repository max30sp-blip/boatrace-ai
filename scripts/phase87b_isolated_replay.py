
import csv
import json
import math
import sqlite3
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import lightgbm as lgb

ROOT = Path(r"C:\boatrace_ai")
TARGET = "2026-10-29"

DB = ROOT / "data/boatrace.db"
TRAIN_DB = ROOT / "data/train_dataset_v1.db"
REPORT = ROOT / "models/phase08_1_b_report.json"
MODEL = ROOT / "models/phase08_1_b_model.txt"

OUT = ROOT / "outputs/phase87b"

SOURCES = {
    "f1": ("features_fast.db", "racer_course_features_fast"),
    "f2": ("lane_features.db", "racer_lane_features_fast"),
    "f3": ("motor_features.db", "equipment_features_fast"),
    "f4": ("venue_detail_features.db", "venue_detail_features_fast"),
    "f5": ("recent_form_features.db", "racer_recent_form_fast"),
    "f6": ("meet_features.db", "racer_meet_features"),
}

P_MAP = {
    "age": "age",
    "weight_kg": "weight_kg",
    "national_win_rate": "national_win_rate",
    "national_2rate": "national_2rate",
    "local_win_rate": "local_win_rate",
    "local_2rate": "local_2rate",
    "motor_2rate": "motor_2rate",
    "boat_2rate": "boat_2rate",
}

GRADES = {"A1": 3, "A2": 2, "B1": 1, "B2": 0}
WINDOWS = (30, 90, 180)


def readonly(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True,
    )
    conn.execute("PRAGMA query_only=ON")
    return conn


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def numeric(v):
    if v is None or v == "":
        return None
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def equal(a, b):
    a, b = numeric(a), numeric(b)
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= 1e-6


def finish_number(value):
    value = numeric(value)
    if value is not None and value.is_integer():
        if 1 <= value <= 6:
            return int(value)
    return None


def read_programs():
    conn = readonly(DB)
    cur = conn.execute("""
        SELECT *
        FROM program_entries
        WHERE race_date = ?
        ORDER BY venue_code, race_no, lane
    """, (TARGET,))

    names = [d[0] for d in cur.description]
    programs = {}
    races = defaultdict(list)

    for raw in cur:
        entry = dict(zip(names, raw))
        venue = int(entry["venue_code"])
        race = int(entry["race_no"])
        lane = int(entry["lane"])
        key = (venue, race, lane)

        if key in programs:
            raise ValueError(f"Duplicate entry: {key}")

        programs[key] = entry
        races[(venue, race)].append(lane)

    conn.close()

    if not races:
        raise ValueError("No program entries")

    for key, lanes in races.items():
        if sorted(lanes) != [1, 2, 3, 4, 5, 6]:
            raise ValueError(f"Incomplete race: {key}")

    return programs, sorted(races)


def build_p(programs):
    result = {}

    for key, entry in programs.items():
        fields = {}

        for suffix, source in P_MAP.items():
            fields[suffix] = numeric(entry[source])

        grade = str(entry["grade"]).strip().upper()
        if grade not in GRADES:
            raise ValueError(f"Unknown grade: {grade}")

        fields["grade_rank"] = GRADES[grade]
        result[key] = fields

    return result


def build_f0(programs):
    target = date.fromisoformat(TARGET)
    earliest = (
        target - timedelta(days=180)
    ).isoformat()

    racer_ids = {
        int(e["racer_id"])
        for e in programs.values()
    }

    conn = readonly(DB)
    merged = {}

    # 優先順位：v2 > supplement
    # 重複履歴は日付・場・レース・選手で排除する。
    for table in (
        "race_boat_results_v2",
        "race_boat_results_supplement",
    ):
        for d, venue, race, rid, finish, st in conn.execute(
            f"""
            SELECT race_date, venue_code, race_no,
                   racer_id, finish_position, start_timing
            FROM {quote(table)}
            WHERE race_date >= ? AND race_date < ?
            """,
            (earliest, TARGET),
        ):
            if None in (d, venue, race, rid):
                continue

            rid = int(rid)
            if rid not in racer_ids:
                continue

            identity = (
                d, int(venue), int(race), rid
            )

            merged.setdefault(
                identity,
                (
                    date.fromisoformat(d),
                    finish_number(finish),
                    numeric(st),
                ),
            )

    conn.close()

    histories = defaultdict(list)

    for identity, item in merged.items():
        histories[identity[3]].append(item)

    result = {}

    for key, entry in programs.items():
        rid = int(entry["racer_id"])
        history = histories[rid]
        fields = {}

        for window in WINDOWS:
            recent = [
                (f, st)
                for day, f, st in history
                if 0 < (target - day).days <= window
            ]

            n = len(recent)
            starts = [st for _, st in recent if st is not None]

            fields[f"starts_{window}d"] = n
            fields[f"win_rate_{window}d"] = (
                sum(f == 1 for f, _ in recent) / n
                if n else None
            )
            fields[f"top3_rate_{window}d"] = (
                sum(
                    f is not None and 1 <= f <= 3
                    for f, _ in recent
                ) / n
                if n else None
            )
            fields[f"avg_st_{window}d"] = (
                sum(starts) / len(starts)
                if starts else None
            )

        result[key] = fields

    print("F0 unique historical records:", len(merged))
    return result


def load_group(group, programs):
    filename, table = SOURCES[group]
    conn = readonly(ROOT / "outputs/phase87_isolated" / TARGET / "data" / filename)

    schema = {
        row[1]
        for row in conn.execute(
            f"PRAGMA table_info({quote(table)})"
        )
    }

    key_col = (
        "lane" if "lane" in schema
        else "boat_no" if "boat_no" in schema
        else None
    )

    if key_col is None:
        raise ValueError(f"{group}: no boat key")

    cur = conn.execute(
        f"SELECT * FROM {quote(table)} "
        "WHERE race_date = ?",
        (TARGET,),
    )

    names = [item[0] for item in cur.description]
    output = {}

    for raw in cur:
        row = dict(zip(names, raw))

        key = (
            int(row["venue_code"]),
            int(row["race_no"]),
            int(row[key_col]),
        )

        if key not in programs:
            continue

        if key in output:
            raise ValueError(f"Duplicate {group}: {key}")

        if "racer_id" in schema:
            if (
                row["racer_id"] is not None
                and int(row["racer_id"])
                != int(programs[key]["racer_id"])
            ):
                raise ValueError(
                    f"Racer mismatch {group}: {key}"
                )

        output[key] = row

    conn.close()

    if len(output) != len(programs):
        raise ValueError(
            f"{group}: {len(output)}/{len(programs)} rows"
        )

    return output


def load_reference():
    conn = readonly(TRAIN_DB)
    cur = conn.execute(
        "SELECT * FROM train_races WHERE race_date = ?",
        (TARGET,),
    )

    names = [d[0] for d in cur.description]
    result = {}

    for values in cur:
        row = dict(zip(names, values))
        key = (
            int(row["venue_code"]),
            int(row["race_no"]),
        )
        if key in result:
            raise ValueError("Duplicate reference race")
        result[key] = row

    conn.close()
    return result


def main():
    print("=== PHASE 8.7-B ISOLATED HISTORICAL REPLAY ===")

    for path in (DB, TRAIN_DB, REPORT, MODEL):
        if not path.is_file():
            raise FileNotFoundError(path)

    report = json.loads(
        REPORT.read_text(encoding="utf-8-sig")
    )

    feature_names = report["feature_names"]
    class_order = report["class_order"]

    if len(feature_names) != 618:
        raise ValueError("Expected 618 features")

    if len(class_order) != 120:
        raise ValueError("Expected 120 classes")

    booster = lgb.Booster(model_file=str(MODEL))

    if booster.feature_name() != feature_names:
        raise ValueError("Model feature order mismatch")

    programs, races = read_programs()
    p = build_p(programs)
    f0 = build_f0(programs)

    sources = {
        group: load_group(group, programs)
        for group in SOURCES
    }

    reference = load_reference()

    print("Program boats:", len(programs))
    print("Complete races:", len(races))
    print("Reference races:", len(reference))

    # モデルの全618列を、1艇単位の特徴量から組み立てる。
    assembled = []
    mismatch_examples = []
    mismatches = defaultdict(int)
    checked = 0

    for venue, race in races:
        key = (venue, race)
        ref = reference.get(key)

        if ref is None:
            raise ValueError(f"No reference: {key}")

        values = {}

        for lane in range(1, 7):
            boat_key = (venue, race, lane)

            for suffix, value in p[boat_key].items():
                values[f"b{lane}_p_{suffix}"] = value

            for suffix, value in f0[boat_key].items():
                values[f"b{lane}_f0_{suffix}"] = value

            for group, rows in sources.items():
                source = rows[boat_key]

                # モデル側の列名を基準にする。
                prefix = f"b{lane}_{group}_"

                for name in feature_names:
                    if not name.startswith(prefix):
                        continue

                    suffix = name[len(prefix):]
                    column = (
                        suffix if suffix in source
                        else f"{group}_{suffix}"
                    )

                    if column not in source:
                        raise ValueError(
                            f"Missing source column: {name}"
                        )

                    values[name] = numeric(source[column])

        missing = set(feature_names) - set(values)

        if missing:
            raise ValueError(
                f"Missing {len(missing)} features: "
                f"{sorted(missing)[:10]}"
            )

        for name in feature_names:
            checked += 1

            if not equal(values[name], ref[name]):
                group = name.split("_", 2)[1]
                mismatches[group] += 1

                if len(mismatch_examples) < 15:
                    mismatch_examples.append({
                        "race": key,
                        "feature": name,
                        "computed": values[name],
                        "reference": ref[name],
                    })

        assembled.append([
            np.nan
            if values[name] is None
            else float(values[name])
            for name in feature_names
        ])

    total_bad = sum(mismatches.values())

    print("\n=== 618 FEATURE VALIDATION ===")
    print("Checked:", checked)
    print("Matched:", checked - total_bad)
    print("Mismatched:", total_bad)
    print("By group:", dict(mismatches))

    for item in mismatch_examples:
        print("Mismatch example:", item)

    if total_bad:
        print("\nPrediction stopped: feature mismatch")
        print("=== PHASE 8.6-F VALIDATION FAILED ===")
        return

    X = np.asarray(assembled, dtype=np.float32)

    if X.shape != (len(races), 618):
        raise ValueError("Feature matrix shape mismatch")

    predictions = booster.predict(
        X,
        num_iteration=report["best_iteration"],
        num_threads=4,
    )

    if predictions.shape != (len(races), 120):
        raise ValueError("Invalid prediction shape")

    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite predictions")

    if not np.allclose(
        predictions.sum(axis=1), 1.0,
        atol=1e-5
    ):
        raise ValueError("Probabilities do not sum to 1")

    OUT.mkdir(parents=True, exist_ok=True)

    all_file = OUT / f"all_probabilities_{TARGET}.csv"
    top_file = OUT / f"top_predictions_{TARGET}.csv"

    if all_file.exists() or top_file.exists():
        raise FileExistsError(
            "Prediction output already exists"
        )

    all_rows = []
    top_rows = []

    for i, (venue, race) in enumerate(races):
        probs = predictions[i]
        ranking = np.argsort(-probs)

        for rank, index in enumerate(ranking, start=1):
            all_rows.append({
                "race_date": TARGET,
                "venue_code": venue,
                "race_no": race,
                "rank": rank,
                "combination": class_order[index],
                "probability": float(probs[index]),
            })

        for rank, index in enumerate(ranking[:5], start=1):
            top_rows.append({
                "race_date": TARGET,
                "venue_code": venue,
                "race_no": race,
                "rank": rank,
                "combination": class_order[index],
                "probability": float(probs[index]),
            })

    fieldnames = [
        "race_date", "venue_code", "race_no",
        "rank", "combination", "probability"
    ]

    for path, rows in (
        (all_file, all_rows),
        (top_file, top_rows),
    ):
        with path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(rows)

    print("\n=== PREDICTION EXAMPLE ===")

    for row in top_rows[:5]:
        print(
            f"{row['combination']}: "
            f"{row['probability']:.4%}"
        )

    print("\n=== PHASE 8.7-B REPLAY COMPLETE ===")
    print("Races:", len(races))
    print("Combinations:", len(all_rows))
    print("Top predictions:", top_file)
    print("All probabilities:", all_file)
    print("LIVE PREDICTION READY: NO")


if __name__ == "__main__":
    main()
