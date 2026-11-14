
import argparse
import itertools
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models" / "conditional_v1"
DATA_FILE = ROOT / "data" / "training" / "trifecta_dataset.csv"
OUTPUT_DIR = ROOT / "outputs" / "predictions"

COMBOS = list(itertools.permutations(range(1, 7), 3))
COMBO_NAMES = ["-".join(map(str, c)) for c in COMBOS]

# 既存CSVの競艇場番号列
VENUE_CANDIDATES = [
    "venue_code",
    "venue",
    "stadium_code",
    "place_code",
]


def load_model():
    metadata_path = MODEL_DIR / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    features = metadata["features"]
    models = []

    for stage in range(1, 4):
        path = MODEL_DIR / f"stage{stage}.txt"

        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        model = lgb.Booster(model_file=str(path))
        models.append(model)

    return models, features, metadata


def load_dataset(features, metadata):
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

    required = [
        "race_date",
        "race_no",
        "target_combo",
        *features
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing[:15]}")

    # 予測には学習終了日より未来のデータだけを使用
    train_end = str(metadata["train_end"])
    df["race_date"] = pd.to_datetime(
        df["race_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    df = df[df["race_date"] > train_end].copy()

    # このモデルは7日間で試作したもの。
    # テスト日より未来のデータには適用しない。
    test_end = str(metadata.get("test_date", "2026-10-29"))
    df = df[df["race_date"] <= test_end].copy()

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["race_no"] = pd.to_numeric(
        df["race_no"], errors="raise"
    ).astype(int)

    return df


def get_venue_column(df):
    for name in VENUE_CANDIDATES:
        if name in df.columns:
            return name

    raise ValueError(
        "競艇場コードの列が見つかりません。\n"
        "CSVの列名を確認し、VENUE_CANDIDATESに追加してください。"
    )


def masked_probability(model, x, forbidden_lanes):
    probabilities = np.asarray(
        model.predict(x), dtype=float
    ).reshape(-1, 6)

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Invalid model probabilities")

    probabilities = np.clip(probabilities, 0.0, None)

    for row_no, forbidden in enumerate(forbidden_lanes):
        for lane in forbidden:
            probabilities[row_no, lane - 1] = 0.0

    totals = probabilities.sum(axis=1, keepdims=True)

    if np.any(totals <= 0):
        raise ValueError("Probability normalization failed")

    return probabilities / totals


def predict_120(models, row, features):
    # 元のPHASE 8.2と同じ列順・追加特徴量を使用
    x = row[features].to_frame().T.reset_index(drop=True)

    for col in features:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    first_p = masked_probability(
        models[0], x, [[]]
    )[0]

    # 1着候補ごとに2着確率を計算
    second_p = {}

    for first in range(1, 7):
        x2 = x.copy()
        x2["first_lane"] = first

        p = masked_probability(
            models[1], x2, [[first]]
        )[0]

        second_p[first] = p

    # 1着・2着候補ごとに3着確率を計算
    third_p = {}

    for first in range(1, 7):
        for second in range(1, 7):
            if first == second:
                continue

            x3 = x.copy()
            x3["first_lane"] = first
            x3["second_lane"] = second

            p = masked_probability(
                models[2], x3, [[first, second]]
            )[0]

            third_p[(first, second)] = p

    results = []

    for first, second, third in COMBOS:
        probability = (
            first_p[first - 1]
            * second_p[first][second - 1]
            * third_p[(first, second)][third - 1]
        )

        results.append({
            "combination": f"{first}-{second}-{third}",
            "probability": float(probability),
        })

    result = pd.DataFrame(results)

    result = result.sort_values(
        "probability", ascending=False,
        kind="stable"
    ).reset_index(drop=True)

    result.insert(
        0, "rank", np.arange(1, len(result) + 1)
    )

    total = result["probability"].sum()

    if len(result) != 120 or not np.isclose(
        total, 1.0, atol=1e-6
    ):
        raise ValueError(
            f"Invalid probability total: {total}"
        )

    result["probability_pct"] = (
        result["probability"] * 100.0
    )

    return result


def show_available(df, venue_col):
    available = df[[
        "race_date", venue_col, "race_no"
    ]].drop_duplicates()

    available = available.sort_values(
        ["race_date", venue_col, "race_no"]
    )

    print("\n=== AVAILABLE RACES ===")

    for (date, venue), group in available.groupby(
        ["race_date", venue_col], sort=True
    ):
        races = ",".join(
            str(int(r)) for r in group["race_no"]
        )

        print(
            f"Date={date} "
            f"Venue={venue} "
            f"Races={races}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="PHASE 9: 3連単120通り予測"
    )

    parser.add_argument(
        "--date", default="2026-10-29"
    )
    parser.add_argument(
        "--venue", type=int
    )
    parser.add_argument(
        "--race", type=int
    )
    parser.add_argument(
        "--top", type=int, default=10
    )
    parser.add_argument(
        "--list", action="store_true"
    )

    args = parser.parse_args()

    models, features, metadata = load_model()
    df = load_dataset(features, metadata)
    venue_col = get_venue_column(df)

    df[venue_col] = pd.to_numeric(
        df[venue_col], errors="raise"
    ).astype(int)

    if args.list:
        show_available(df, venue_col)
        return

    if args.venue is None or args.race is None:
        parser.error(
            "予測には --venue と --race が必要です。"
            "確認する場合は --list を使ってください。"
        )

    selected = df[
        (df["race_date"] == args.date)
        & (df[venue_col] == args.venue)
        & (df["race_no"] == args.race)
    ]

    if len(selected) != 1:
        raise ValueError(
            "指定レースが1件に絞れませんでした。"
            f" Found={len(selected)}\n"
            "--list で登録済みレースを確認してください。"
        )

    row = selected.iloc[0]

    prediction = predict_120(
        models, row, features
    )

    print("\n========== PHASE 9 ==========")
    print("Date:", args.date)
    print("Venue:", args.venue)
    print("Race:", args.race)
    print("Model: conditional_v1")
    print("Combinations:", len(prediction))
    print(
        "Probability total:",
        round(prediction["probability"].sum(), 8)
    )

    print("\n=== TOP PREDICTIONS ===")

    top_k = max(1, min(args.top, 120))

    for _, r in prediction.head(top_k).iterrows():
        print(
            f"{int(r['rank']):3d}. "
            f"{r['combination']:5s} "
            f"{r['probability_pct']:7.3f}%"
        )

    actual = row["target_combo"]
    print("\nActual result:", actual)

    matched = prediction[
        prediction["combination"] == actual
    ]

    if not matched.empty:
        actual_rank = int(matched.iloc[0]["rank"])
        print("Actual result rank:", actual_rank)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / (
        f"{args.date}_venue{args.venue:02d}"
        f"_race{args.race:02d}.csv"
    )

    prediction.to_csv(
        output_path, index=False,
        encoding="utf-8-sig"
    )

    print("\nCSV saved:", output_path)
    print("PHASE 9 COMPLETE")


if __name__ == "__main__":
    main()
