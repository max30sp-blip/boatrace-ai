
import ast
import csv
import math
import sqlite3
from pathlib import Path
from collections import Counter

ROOT = Path(r"C:\boatrace_ai")
SCRIPTS = ROOT / "scripts"
CSV = ROOT / "outputs/phase86c/f0_comparison_2026-10-29.csv"
DB = ROOT / "data/boatrace.db"

SOURCE_FILES = [
    "phase06_fast_features.py",
    "phase06_build_features.py",
    "phase07_build_dataset.py",
]

KEYWORDS = (
    "starts", "win_rate", "top3_rate",
    "avg_st", "finish_position",
    "start_timing", "race_boat_results",
    "history", "window"
)


def read_number(value):
    if value is None or str(value).strip() == "":
        return None

    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def inspect_sources():
    print("\n=== 1. ORIGINAL F0 IMPLEMENTATION ===")

    for name in SOURCE_FILES:
        path = SCRIPTS / name

        if not path.is_file():
            print("NOT FOUND:", path)
            continue

        lines = path.read_text(
            encoding="utf-8-sig",
            errors="replace"
        ).splitlines()

        print(f"\nFILE: {name}")
        print("Lines:", len(lines))

        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError as e:
            print("AST parse failed:", e)
            continue

        functions = [
            node for node in ast.walk(tree)
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            )
        ]

        for fn in functions:
            body = "\n".join(
                lines[fn.lineno - 1:fn.end_lineno]
            ).lower()

            if not any(word in body for word in KEYWORDS):
                continue

            print(
                f"FUNCTION {fn.name}: "
                f"L{fn.lineno}-L{fn.end_lineno}"
            )

        # 重要なSQL・日付境界・集計条件を表示
        selected = set()

        for i, line in enumerate(lines):
            lower = line.lower()

            if any(word in lower for word in (
                "from race_",
                "finish_position",
                "start_timing",
                "timedelta",
                "days <",
                "days <=",
                "days >",
                "days >=",
                "365",
                "180",
                "90",
                "30",
                "valid_finish",
                "valid_st",
                "f0_",
            )):
                for j in range(
                    max(0, i - 2),
                    min(len(lines), i + 3)
                ):
                    selected.add(j)

        print("\nRelevant code:")

        for i in sorted(selected)[:130]:
            print(f"L{i+1:04d}: {lines[i]}")

        if len(selected) > 130:
            print("Additional relevant lines omitted")


def inspect_differences():
    print("\n=== 2. F0 DIFFERENCE PATTERNS ===")

    if not CSV.is_file():
        raise FileNotFoundError(CSV)

    with CSV.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    print("Compared boats:", len(rows))

    for window in (30, 90, 180):
        print(f"\n--- WINDOW {window} DAYS ---")

        for field in (
            "starts", "win_rate",
            "top3_rate", "avg_st"
        ):
            suffix = f"{field}_{window}d"
            differences = []
            missing = Counter()
            examples = []

            for row in rows:
                a = read_number(
                    row[f"computed_{suffix}"]
                )
                b = read_number(
                    row[f"reference_{suffix}"]
                )

                if a is None or b is None:
                    if a is None and b is None:
                        continue

                    missing[
                        "computed_null"
                        if a is None else "reference_null"
                    ] += 1
                    continue

                diff = a - b

                if abs(diff) <= 1e-6:
                    continue

                differences.append(diff)

                if len(examples) < 3:
                    examples.append((
                        row["venue_code"],
                        row["race_no"],
                        row["lane"],
                        row["racer_id"],
                        round(a, 6),
                        round(b, 6),
                        round(diff, 6)
                    ))

            pos = sum(d > 0 for d in differences)
            neg = sum(d < 0 for d in differences)

            mean_abs = (
                sum(abs(d) for d in differences)
                / len(differences)
                if differences else 0
            )

            print(
                f"{field:12s}",
                f"diffs={len(differences):3d}",
                f"positive={pos:3d}",
                f"negative={neg:3d}",
                f"mean_abs={mean_abs:.6f}",
                f"nulls={dict(missing)}"
            )

            if field == "starts":
                rounded = Counter(
                    int(round(d)) for d in differences
                )
                print(
                    "  Start-count differences:",
                    rounded.most_common(10)
                )

                for example in examples:
                    print("  Example:", example)


def inspect_tables():
    print("\n=== 3. SOURCE TABLE COVERAGE ===")

    if not DB.is_file():
        raise FileNotFoundError(DB)

    con = sqlite3.connect(
        DB.resolve().as_uri() + "?mode=ro",
        uri=True
    )

    con.execute("PRAGMA query_only=ON")

    for table in (
        "race_boat_results",
        "race_boat_results_v2",
        "race_boat_results_supplement"
    ):
        exists = con.execute("""
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type='table' AND name=?
        """, (table,)).fetchone()[0]

        if not exists:
            continue

        columns = {
            x[1] for x in con.execute(
                f"PRAGMA table_info({table})"
            )
        }

        if not {"race_date", "racer_id"} <= columns:
            continue

        count = con.execute(f"""
            SELECT COUNT(*),
                   MIN(race_date),
                   MAX(race_date),
                   COUNT(DISTINCT racer_id)
            FROM {table}
            WHERE race_date >= '2026-05-01'
              AND race_date < '2026-10-29'
        """).fetchone()

        print(table, count)

    con.close()


if __name__ == "__main__":
    print("=== PHASE 8.6-C F0 DIAGNOSTIC ===")

    inspect_sources()
    inspect_differences()
    inspect_tables()

    print("\n=== PHASE 8.6-C DIAGNOSTIC COMPLETE ===")
