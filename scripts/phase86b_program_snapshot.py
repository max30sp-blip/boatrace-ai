
import argparse
import json
import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import date

ROOT = Path(r"C:\boatrace_ai")
DB = ROOT / "data/boatrace.db"
REPORT = ROOT / "models/phase08_1_b_report.json"
OUT = ROOT / "outputs/phase86b"

P_FIELDS = {
    "age": "p_age",
    "weight_kg": "p_weight_kg",
    "national_win_rate": "p_national_win_rate",
    "national_2rate": "p_national_2rate",
    "local_win_rate": "p_local_win_rate",
    "local_2rate": "p_local_2rate",
    "motor_2rate": "p_motor_2rate",
    "boat_2rate": "p_boat_2rate",
}

def numeric(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def read_only(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    con = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True
    )
    con.execute("PRAGMA query_only=ON")
    return con

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD（省略時はDBの最新日付）"
    )
    args = parser.parse_args()

    with REPORT.open(encoding="utf-8-sig") as f:
        report = json.load(f)

    model_features = report["feature_names"]

    if len(model_features) != 618:
        raise ValueError("Unexpected model feature count")

    con = read_only(DB)

    latest = con.execute("""
        SELECT MAX(race_date)
        FROM program_entries
    """).fetchone()[0]

    if not latest:
        raise ValueError(
            "program_entries has no race dates"
        )

    target = args.date or latest
    date.fromisoformat(target)

    print("=== PHASE 8.6-B PROGRAM SNAPSHOT ===")
    print("Latest program date:", latest)
    print("Selected date:", target)

    if args.date is None:
        print("Mode: latest date stored in DB")
    else:
        print("Mode: explicitly selected date")

    cur = con.execute("""
        SELECT
            race_date, venue_code, race_no,
            lane, racer_id, racer_name,
            age, branch, weight_kg, grade,
            national_win_rate, national_2rate,
            local_win_rate, local_2rate,
            motor_no, motor_2rate,
            boat_no, boat_2rate, source_file
        FROM program_entries
        WHERE race_date = ?
        ORDER BY venue_code, race_no, lane
    """, (target,))

    columns = [d[0] for d in cur.description]
    rows = [
        dict(zip(columns, r))
        for r in cur.fetchall()
    ]

    con.close()

    if not rows:
        print("No program entries on selected date")
        return

    grouped = defaultdict(list)

    for r in rows:
        key = (
            r["race_date"],
            r["venue_code"],
            r["race_no"]
        )
        grouped[key].append(r)

    complete = []
    incomplete = []
    issues = defaultdict(int)

    expected_p_names = {
        f"b{lane}_{field}"
        for lane in range(1, 7)
        for field in (
            list(P_FIELDS.values()) + ["p_grade_rank"]
        )
    }

    missing_model_columns = (
        expected_p_names - set(model_features)
    )

    if missing_model_columns:
        raise ValueError(
            f"Unexpected P columns: {missing_model_columns}"
        )

    for race_key, entries in sorted(grouped.items()):
        lanes = [r["lane"] for r in entries]

        if (
            len(entries) != 6
            or sorted(lanes) != [1, 2, 3, 4, 5, 6]
        ):
            incomplete.append({
                "race": list(race_key),
                "lanes": lanes
            })
            continue

        p_features = {}
        raw_entries = []

        for entry in sorted(
            entries,
            key=lambda r: r["lane"]
        ):
            lane = int(entry["lane"])

            if not entry["racer_id"]:
                issues["missing_racer_id"] += 1

            if entry["motor_no"] is None:
                issues["missing_motor_no"] += 1

            if entry["boat_no"] is None:
                issues["missing_boat_no"] += 1

            for source, suffix in P_FIELDS.items():
                name = f"b{lane}_{suffix}"
                p_features[name] = numeric(
                    entry[source]
                )

                if p_features[name] is None:
                    issues["missing_numeric_p"] += 1

            # 等級の数値変換は既存学習処理と
            # 照合できるまで確定させない
            p_features[
                f"b{lane}_p_grade_rank"
            ] = None

            raw_entries.append({
                "lane": lane,
                "racer_id": entry["racer_id"],
                "racer_name": entry["racer_name"],
                "grade": entry["grade"],
                "motor_no": entry["motor_no"],
                "boat_no": entry["boat_no"],
                "source_file": entry["source_file"]
            })

        complete.append({
            "race_date": race_key[0],
            "venue_code": race_key[1],
            "race_no": race_key[2],
            "entries": raw_entries,
            "p_features": p_features,
            "feature_status": "P_RAW_ONLY",
            "prediction_ready": False
        })

    print("\n=== PROGRAM VALIDATION ===")
    print("Program rows:", len(rows))
    print("Total races:", len(grouped))
    print("Complete six-boat races:", len(complete))
    print("Incomplete races:", len(incomplete))
    print("Issues:", dict(issues))

    print("\n=== MODEL COVERAGE ===")
    print("Total model features:", len(model_features))
    print("P input fields:", len(P_FIELDS) * 6)
    print("Grade fields pending:", 6)
    print("F0-F6 pending:", 564)
    print("Prediction ready: NO")

    if complete:
        example = complete[0]
        print("\n=== FIRST COMPLETE RACE ===")
        print(
            "Date:", example["race_date"],
            "Venue:", example["venue_code"],
            "Race:", example["race_no"]
        )

        for e in example["entries"]:
            print(
                f"Lane {e['lane']}:",
                f"Racer={e['racer_id']}",
                f"Motor={e['motor_no']}",
                f"Boat={e['boat_no']}",
                f"Grade={e['grade']}"
            )

    output = {
        "target_date": target,
        "latest_program_date": latest,
        "model_feature_count": len(model_features),
        "prediction_ready": False,
        "complete_races": complete,
        "incomplete_races": incomplete,
        "issues": dict(issues)
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"program_snapshot_{target}.json"

    if path.exists():
        raise FileExistsError(
            f"Snapshot already exists: {path}"
        )

    with path.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            output, f,
            ensure_ascii=False,
            indent=2
        )

    print("\n=== PHASE 8.6-B SNAPSHOT COMPLETE ===")
    print("Output:", path)

if __name__ == "__main__":
    main()
