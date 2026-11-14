
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(r"C:\boatrace_ai")
DB = ROOT / "data" / "train_dataset_v1.db"
SCRIPT = ROOT / "scripts" / "phase08_1_scale_training.py"
REPORT = ROOT / "models" / "phase08_1_b_report.json"
MODEL = ROOT / "models" / "phase08_1_b_model.txt"

KEYWORDS = [
    "split", "train", "valid", "test", "feature",
    "label", "seed", "random", "num_leaves",
    "learning_rate", "early_stopping",
    "num_boost_round", "categorical",
    "objective", "metric", "date"
]


def inspect_database():
    print("\n=== 1. TRAINING DATABASE ===")

    if not DB.exists():
        print("Database not found:", DB)
        return

    conn = sqlite3.connect(
        DB.as_uri() + "?mode=ro", uri=True
    )
    conn.execute("PRAGMA query_only=ON")

    tables = conn.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        ORDER BY name
    """).fetchall()

    for (table,) in tables:
        safe = table.replace('"', '""')

        columns = conn.execute(
            f'PRAGMA table_info("{safe}")'
        ).fetchall()

        names = [r[1] for r in columns]

        print("\nTABLE:", table)
        print("COLUMNS:", len(names))
        print("NAMES:", ", ".join(names))

        count = conn.execute(
            f'SELECT COUNT(*) FROM "{safe}"'
        ).fetchone()[0]

        print("ROWS:", count)

        date_columns = [
            c for c in names
            if c.lower() in (
                "race_date", "date", "event_date"
            )
        ]

        for dc in date_columns:
            escaped = dc.replace('"', '""')
            result = conn.execute(f"""
                SELECT MIN("{escaped}"), MAX("{escaped}")
                FROM "{safe}"
            """).fetchone()

            print("DATE RANGE:", dc, result)

    conn.close()


def inspect_training_script():
    print("\n=== 2. TRAINING SCRIPT ===")

    if not SCRIPT.exists():
        print("Script not found:", SCRIPT)
        return

    lines = SCRIPT.read_text(
        encoding="utf-8-sig",
        errors="replace"
    ).splitlines()

    print("SCRIPT:", SCRIPT)
    print("TOTAL LINES:", len(lines))

    # 学習条件に関係する箇所とその周辺を表示
    selected = set()

    for i, line in enumerate(lines):
        stripped = line.strip().lower()

        if any(k in stripped for k in KEYWORDS):
            for j in range(
                max(0, i - 2),
                min(len(lines), i + 4)
            ):
                selected.add(j)

    # 長すぎる出力を避ける
    indices = sorted(selected)
    print("RELEVANT LINES:", len(indices))

    for i in indices[:350]:
        print(f"L{i+1:04d}: {lines[i]}")

    if len(indices) > 350:
        print("... additional lines omitted")


def inspect_report():
    print("\n=== 3. BASELINE MODEL REPORT ===")

    if not REPORT.exists():
        print("Report not found:", REPORT)
        return

    data = json.loads(
        REPORT.read_text(encoding="utf-8-sig")
    )

    print("REPORT:", REPORT)

    print("TOP LEVEL KEYS:")
    if isinstance(data, dict):
        print(list(data.keys()))

    def walk(value, prefix="", depth=0):
        if depth > 6:
            return

        if isinstance(value, dict):
            for k, v in value.items():
                name = f"{prefix}.{k}" if prefix else k

                if isinstance(v, (dict, list)):
                    walk(v, name, depth + 1)
                elif any(
                    word in name.lower()
                    for word in KEYWORDS + [
                        "top1", "top3", "loss",
                        "accuracy", "rows", "score",
                        "best_iteration"
                    ]
                ):
                    print(f"{name}: {v}")

        elif isinstance(value, list):
            print(prefix, "list length:", len(value))

            if len(value) <= 5:
                for i, item in enumerate(value):
                    walk(item, f"{prefix}[{i}]", depth + 1)

    walk(data)


def inspect_model():
    print("\n=== 4. MODEL METADATA ===")

    if not MODEL.exists():
        print("Model not found:", MODEL)
        return

    print("MODEL:", MODEL)
    print("SIZE:", MODEL.stat().st_size)

    # LightGBMモデルのヘッダーだけ読む
    with MODEL.open(
        encoding="utf-8",
        errors="replace"
    ) as f:
        for i, line in enumerate(f):
            if i >= 12:
                break

            if len(line) > 250:
                line = line[:250] + " ..."

            print(line.rstrip())


if __name__ == "__main__":
    print("=== PHASE 8.3-D BASELINE AUDIT ===")

    inspect_database()
    inspect_training_script()
    inspect_report()
    inspect_model()

    print("\n=== PHASE 8.3-D BASELINE AUDIT COMPLETE ===")
