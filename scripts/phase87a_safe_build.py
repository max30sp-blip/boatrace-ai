import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(r"C:\boatrace_ai")
SOURCE_DB = ROOT / "data" / "boatrace.db"
RAW_PROGRAMS = ROOT / "data" / "extracted" / "programs"
SCRIPT_DIR = ROOT / "scripts"

JOBS = [
    ("phase06_fast_features.py", "features_fast.db", "racer_features_fast"),
    ("phase06_1_fast_course_features.py", "features_fast.db", "racer_course_features_fast"),
    ("phase06_2_lane_features.py", "lane_features.db", "racer_lane_features_fast"),
    ("phase06_3_motor_features.py", "motor_features.db", "equipment_features_fast"),
    ("phase06_4_venue_detail_features.py", "venue_detail_features.db", "venue_detail_features_fast"),
    ("phase06_5_recent_form_features.py", "recent_form_features.db", "racer_recent_form_fast"),
    ("phase06_6_meet_features.py", "meet_features.db", "racer_meet_features"),
]


def readonly(db_path):
    db = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    return db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="YYYY-MM-DD (default: latest stored program)")
    args = parser.parse_args()

    if not SOURCE_DB.is_file():
        raise FileNotFoundError(SOURCE_DB)
    if not RAW_PROGRAMS.is_dir():
        raise FileNotFoundError(RAW_PROGRAMS)

    with readonly(SOURCE_DB) as db:
        latest = db.execute("SELECT MAX(race_date) FROM program_entries").fetchone()[0]
        if latest is None:
            raise RuntimeError("program_entries is empty")
        target = args.date or latest
        date.fromisoformat(target)
        if target > latest:
            raise RuntimeError(f"Requested {target}, but latest imported program is {latest}")
        program_rows = db.execute(
            "SELECT COUNT(*) FROM program_entries WHERE race_date = ?", (target,)
        ).fetchone()[0]
        complete_races = db.execute("""
            SELECT COUNT(*) FROM (
                SELECT venue_code, race_no
                FROM program_entries WHERE race_date = ?
                GROUP BY venue_code, race_no
                HAVING COUNT(*) = 6 AND COUNT(DISTINCT lane) = 6
            )
        """, (target,)).fetchone()[0]

    if program_rows == 0 or complete_races * 6 != program_rows:
        raise RuntimeError(
            f"Incomplete programs on {target}: {program_rows} boat rows, {complete_races} complete races"
        )

    b_file = "b" + target[2:].replace("-", "") + ".txt"
    if not list(RAW_PROGRAMS.glob("*/" + b_file)):
        raise FileNotFoundError(
            f"F6 requires original program B-file ({b_file}) under {RAW_PROGRAMS}"
        )

    work = ROOT / "outputs" / "phase87_isolated" / target
    if work.exists():
        raise FileExistsError(f"Isolated output already exists: {work}")

    (work / "data").mkdir(parents=True)
    (work / "scripts").mkdir()

    print("=== PHASE 8.7-A ISOLATED PRE-RACE FEATURE BUILD ===", flush=True)
    print("Target:", target, "| boats:", program_rows, flush=True)
    print("Source database: READ ONLY", SOURCE_DB, flush=True)
    print("Raw program files: READ ONLY", RAW_PROGRAMS, flush=True)
    print("Isolated output:", work, flush=True)

    results = []
    for i, (name, out_name, table) in enumerate(JOBS, 1):
        src = SCRIPT_DIR / name
        if not src.is_file():
            raise FileNotFoundError(src)
        source = src.read_text(encoding="utf-8-sig")
        old = 'ROOT / "data" / "boatrace.db"'
        if source.count(old) != 1:
            raise RuntimeError(f"Unexpected source DB configuration in {name}")
        source = source.replace(old, f'Path(r"{SOURCE_DB}")')
        if name == "phase06_6_meet_features.py":
            old_programs = 'ROOT / "data" / "extracted" / "programs"'
            if source.count(old_programs) != 1:
                raise RuntimeError("Unexpected B-file path configuration")
            source = source.replace(old_programs, f'Path(r"{RAW_PROGRAMS}")')
        copied = work / "scripts" / name
        copied.write_text(source, encoding="utf-8")

        print(f"\n=== [{i}/7] {name} ===", flush=True)
        subprocess.run(
            [sys.executable, "-X", "utf8", "-u", str(copied), "--end", target],
            check=True,
        )

        out_path = work / "data" / out_name
        if not out_path.is_file():
            raise FileNotFoundError(out_path)
        with readonly(out_path) as db:
            n = db.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE race_date = ?', (target,)
            ).fetchone()[0]
        print("TARGET ROWS:", n, "/", program_rows, flush=True)
        if n != program_rows:
            raise RuntimeError(f"{name}: expected {program_rows} rows, got {n}")
        results.append({"script": name, "db": str(out_path), "table": table, "target_rows": n})

    manifest = {
        "target_date": target,
        "program_boat_rows": program_rows,
        "complete_races": complete_races,
        "uses_original_source_db_read_only": True,
        "uses_prior_day_results_only": True,
        "feature_built_before_same_day_results": True,
        "feature_sources": results,
        "live_ready": False,
        "note": "Availability/timestamp and model input assembly still require validation",
    }
    manifest_path = work / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== PHASE 8.7-A ISOLATED BUILD COMPLETE ===", flush=True)
    print("Manifest:", manifest_path)
    print("LIVE READY: NO (next: integrate 618 features and predict)")


if __name__ == "__main__":
    main()
