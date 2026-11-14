
import argparse
import importlib.util
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

ROOT = Path(r"C:\boatrace_ai")

BASE_SCRIPT = ROOT / "scripts/phase08_1_scale_training.py"
BASE_REPORT = ROOT / "models/phase08_1_b_report.json"
TRAIN_DB = ROOT / "data/train_dataset_v1.db"
F3_DB = ROOT / "outputs/phase83d_wide_f3.db"

SAVE_DIR = ROOT / "models/phase84a_candidate"

THREADS = 4
ROUNDS = 250

for path in (BASE_SCRIPT, BASE_REPORT, TRAIN_DB, F3_DB):
    if not path.is_file():
        raise FileNotFoundError(path)

with BASE_REPORT.open(encoding="utf-8-sig") as f:
    baseline_report = json.load(f)

original_features = baseline_report["feature_names"]

if len(original_features) != 618:
    raise ValueError("Baseline feature count mismatch")

if baseline_report["train_period"] != [
    "2017-01-01", "2023-12-31"
]:
    raise ValueError("Unexpected baseline train period")

if baseline_report["valid_period"] != [
    "2024-01-01", "2024-12-31"
]:
    raise ValueError("Unexpected baseline valid period")

if baseline_report["train_rows"] != 383201:
    raise ValueError("Unexpected baseline train rows")

if baseline_report["valid_rows"] != 10000:
    raise ValueError("Unexpected baseline valid rows")


# 追加F3列はDBに存在する順番で取得
check = sqlite3.connect(str(F3_DB))

extra_columns = [
    row[1]
    for row in check.execute(
        "PRAGMA table_info(wide_f3)"
    )
    if row[1].startswith("b")
    and "_new_" in row[1]
]

total, matched = check.execute("""
    SELECT COUNT(*), SUM(f3_matched)
    FROM wide_f3
""").fetchone()

check.close()

if len(extra_columns) != 48:
    raise ValueError(
        f"Expected 48 F3 columns: {len(extra_columns)}"
    )

if len(set(extra_columns)) != 48:
    raise ValueError("Duplicate F3 columns")

if (total, matched) != (438488, 438088):
    raise ValueError(
        f"Unexpected F3 coverage: {(total, matched)}"
    )

new_features = original_features + extra_columns

if len(new_features) != 666:
    raise ValueError("Expected 666 features")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 既存の安定版モデルは上書きしない
if any(SAVE_DIR.glob("*")):
    raise FileExistsError(
        "Candidate output directory is not empty: "
        f"{SAVE_DIR}\n"
        "既存の実験結果を保護するため停止しました。"
    )


# 既存学習プログラムを読み込む
spec = importlib.util.spec_from_file_location(
    "phase08_original",
    BASE_SCRIPT
)

baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

if not hasattr(baseline, "main"):
    raise RuntimeError("Baseline main() not found")

# 学習済み安定版とは別の場所へ保存する
baseline.MODEL_DIR = SAVE_DIR



def enhanced_feature_columns(connection):
    columns = [
        row[1]
        for row in connection.execute(
            "PRAGMA temp.table_info(train_races)"
        )
    ]

    if not columns:
        raise ValueError(
            "F3 TEMP VIEW was not created"
        )

    features = columns[4:]

    if len(features) != 666:
        raise ValueError(
            f"Expected 666 features, got {len(features)}"
        )

    if features != new_features:
        raise ValueError(
            "Feature order differs from expected order"
        )

    print("\n=== FEATURE CHECK ===", flush=True)
    print("Original:", len(original_features))
    print("Added:", len(extra_columns))
    print("Total:", len(features))

    return features



# 元の学習DBを変更せず、
# TEMP VIEW上で既存データと新F3を結合する
original_connect = sqlite3.connect



def enhanced_connect(database, *args, **kwargs):
    db_string = str(database)

    # 通常パス・SQLite URI形式の両方に対応
    is_training_db = (
        "train_dataset_v1.db" in db_string.lower()
    )

    connection = original_connect(
        database, *args, **kwargs
    )

    if not is_training_db:
        return connection

    print(
        "\n=== ATTACHING HISTORICAL F3 ===",
        flush=True
    )

    connection.execute(
        "ATTACH DATABASE ? AS extra_f3",
        (str(F3_DB),)
    )

    selects = ",\n".join(
        f'w."{c}" AS "{c}"'
        for c in extra_columns
    )

    connection.execute(f"""
        CREATE TEMP VIEW train_races AS
        SELECT
            t.*,
            {selects}
        FROM main.train_races t
        LEFT JOIN extra_f3.wide_f3 w
          ON t.race_date = w.race_date
         AND t.venue_code = w.venue_code
         AND t.race_no = w.race_no
    """)

    columns = [
        row[1]
        for row in connection.execute(
            "PRAGMA temp.table_info(train_races)"
        )
    ]

    print("Merged columns:", len(columns))
    print("Original features:", len(original_features))
    print("Additional features:", len(extra_columns))

    if len(columns) != 670:
        connection.close()
        raise ValueError(
            f"Expected 670 columns, got {len(columns)}"
        )

    # 特徴量の重複・欠落を検証
    if len(set(columns)) != len(columns):
        connection.close()
        raise ValueError("Duplicate column names")

    if columns[4:] != new_features:
        connection.close()
        raise ValueError(
            "Merged feature order mismatch"
        )

    count = connection.execute("""
        SELECT COUNT(*)
        FROM temp.train_races
        WHERE race_date BETWEEN '2017-01-01'
                            AND '2024-12-31'
    """).fetchone()[0]

    if count != 438488:
        connection.close()
        raise ValueError(
            f"Race count changed: {count}"
        )

    print("Merged race rows:", count, flush=True)

    return connection



# 既存コード側の学習条件を固定する
def baseline_args(self, *args, **kwargs):
    return argparse.Namespace(
        stage="B",
        train_limit=400000,
        valid_limit=10000,
        rounds=ROUNDS,
        threads=THREADS,
        save=True
    )


baseline.feature_columns = enhanced_feature_columns

print("=== PHASE 8.4-A F3 TRAINING ===")
print("Baseline features:", 618)
print("New F3 features:", 48)
print("Total features:", 666)
print("Train: 2017-2023")
print("Validation: 2024")
print("Save directory:", SAVE_DIR)
print("Stable model: unchanged")
print("Starting training...", flush=True)


# 既存学習コードの抽出ロジックと評価処理を再利用
with patch.object(
    sqlite3, "connect", enhanced_connect
), patch.object(
    argparse.ArgumentParser,
    "parse_args",
    baseline_args
):
    baseline.main()


print("\n=== PHASE 8.4-A TRAINING COMPLETE ===")

# 新しい結果ファイルを検出して安定版と比較
reports = list(SAVE_DIR.glob("*_report.json"))

if len(reports) != 1:
    print(
        "Training finished, but expected exactly "
        "one report:", len(reports)
    )
else:
    with reports[0].open(encoding="utf-8-sig") as f:
        new_report = json.load(f)

    print("\n=== BASELINE VS F3 ===")

    for metric in ("top1", "top3", "logloss"):
        old = baseline_report["lightgbm"][metric]
        new = new_report["lightgbm"][metric]

        print(
            f"{metric:8s}",
            f"old={old:.6f}",
            f"new={new:.6f}",
            f"delta={new-old:+.6f}"
        )

    print(
        "Best iteration:",
        new_report["best_iteration"]
    )

    print("Report:", reports[0])
