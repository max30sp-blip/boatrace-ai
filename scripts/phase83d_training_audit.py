
from pathlib import Path
import csv
import re

ROOT = Path(r"C:\boatrace_ai")
SCRIPTS = ROOT / "scripts"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

KEYWORDS = [
    "lightgbm",
    "lgb.train",
    "lgbmclassifier",
    "train_test_split",
    "early_stopping",
    "log_loss",
    "top1",
    "top3",
    "train_data",
    "valid_data",
    "feature_cols",
    "feature_columns",
    "predict_proba",
]

MODEL_SUFFIXES = {
    ".pkl", ".joblib", ".txt",
    ".json", ".bin"
}


def scan_scripts():
    print("\n=== 1. TRAINING SCRIPT CANDIDATES ===")

    candidates = []

    for path in sorted(SCRIPTS.rglob("*.py")):
        try:
            lines = path.read_text(
                encoding="utf-8-sig",
                errors="replace"
            ).splitlines()
        except OSError:
            continue

        matches = []

        for i, line in enumerate(lines, 1):
            lower = line.lower()

            for keyword in KEYWORDS:
                if keyword in lower:
                    matches.append((i, keyword, line.strip()))
                    break

        if matches:
            candidates.append(
                (path, len(lines), matches)
            )

    ranked = sorted(
        candidates,
        key=lambda item: (
            -len(item[2]),
            str(item[0])
        )
    )

    print("Candidate files:", len(ranked))

    for path, lines, matches in ranked[:20]:
        print("\nFILE:", path.relative_to(ROOT))
        print("Total lines:", lines)
        print("Keyword matches:", len(matches))

        for line_no, keyword, content in matches[:8]:
            print(
                f"  L{line_no}: "
                f"[{keyword}] {content[:140]}"
            )


def scan_models():
    print("\n=== 2. MODEL FILES ===")

    if not MODELS.exists():
        print("Models folder not found")
        return

    files = [
        p for p in MODELS.rglob("*")
        if p.is_file()
        and p.suffix.lower() in MODEL_SUFFIXES
    ]

    for path in sorted(files)[:50]:
        print(
            path.relative_to(ROOT),
            "size=", path.stat().st_size
        )

    print("Model files found:", len(files))


def scan_outputs():
    print("\n=== 3. TRAINING DATA CANDIDATES ===")

    if not OUTPUTS.exists():
        return

    candidates = []

    for path in OUTPUTS.rglob("*"):
        if not path.is_file():
            continue

        name = path.name.lower()

        if not any(k in name for k in (
            "train", "valid", "test",
            "feature", "dataset",
            "phase81", "phase82"
        )):
            continue

        if path.suffix.lower() not in (
            ".csv", ".parquet",
            ".pkl", ".joblib"
        ):
            continue

        candidates.append(path)

    for path in sorted(candidates)[:50]:
        print(
            path.relative_to(ROOT),
            "size=", path.stat().st_size
        )

    print("Candidate datasets:", len(candidates))


def inspect_f3():
    print("\n=== 4. CURRENT F3 PREVIEW ===")

    path = OUTPUTS / "phase83d_integrated_preview.csv"

    if not path.exists():
        print("F3 preview not found")
        return

    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        count = sum(1 for _ in reader)

    print("Rows:", count)
    print("Columns:", len(columns))

    print("\nReset feature candidates:")

    for column in columns:
        if (
            column.startswith("reset_motor_")
            or column.startswith("motor_reset_ability_")
        ):
            print(" ", column)


if __name__ == "__main__":
    print("=== PHASE 8.3-D TRAINING AUDIT ===")
    scan_scripts()
    scan_models()
    scan_outputs()
    inspect_f3()
    print("\n=== PHASE 8.3-D TRAINING AUDIT COMPLETE ===")
