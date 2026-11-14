
import itertools
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "training" / "trifecta_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 7日間のみ使用
START = "2026-10-23"
TRAIN_END = "2026-10-27"
VALID_DAY = "2026-10-28"
TEST_DAY = "2026-10-29"

# 必ず120通りのラベルを作成
COMBOS = [
    "-".join(map(str, combo))
    for combo in itertools.permutations(range(1, 7), 3)
]

COMBO_TO_ID = {
    combo: i for i, combo in enumerate(COMBOS)
}


def evaluate(name, model, dataset, labels):
    predictions = model.predict(dataset)
    predictions = np.asarray(predictions).reshape(-1, 120)

    # 確率分布の整合性を確認
    assert np.all(np.isfinite(predictions))
    assert np.allclose(predictions.sum(axis=1), 1, atol=1e-4)

    order = np.argsort(-predictions, axis=1)

    print(f"\n=== {name} ===")
    print("Races:", len(labels))

    for k in (1, 3, 5):
        hits = np.any(order[:, :k] == labels[:, None], axis=1)
        print(f"Top {k} accuracy: {hits.mean() * 100:.2f}%")

    correct = predictions[np.arange(len(labels)), labels]
    logloss = -np.log(np.clip(correct, 1e-15, 1)).mean()
    print(f"Log loss: {logloss:.5f}")

    return predictions, order


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")

    # 過去データが追加されても7日分に固定
    df = df[
        (df["race_date"] >= START)
        & (df["race_date"] <= TEST_DAY)
    ].copy()

    drop = ["race_date", "race_no", "target_combo"]
    feature_cols = [c for c in df.columns if c not in drop]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if not df["target_combo"].isin(COMBO_TO_ID).all():
        raise ValueError("Invalid trifecta target detected")

    df["label"] = df["target_combo"].map(COMBO_TO_ID)

    train = df[df["race_date"] <= TRAIN_END]
    valid = df[df["race_date"] == VALID_DAY]
    test = df[df["race_date"] == TEST_DAY]

    if min(len(train), len(valid), len(test)) == 0:
        raise RuntimeError("Train/valid/test has empty data")

    # 学習期間にすべて欠損している列は使わない
    usable = [
        c for c in feature_cols
        if train[c].notna().any()
    ]

    x_train = train[usable]
    x_valid = valid[usable]
    x_test = test[usable]

    print("========== PHASE 08 ==========")
    print("Train races:", len(train))
    print("Validation races:", len(valid))
    print("Test races:", len(test))
    print("Features:", len(usable))
    print("Classes:", len(COMBOS))

    train_data = lgb.Dataset(
        x_train,
        label=train["label"].to_numpy(),
        feature_name=usable
    )

    valid_data = lgb.Dataset(
        x_valid,
        label=valid["label"].to_numpy(),
        reference=train_data
    )

    params = {
        "objective": "multiclass",
        "num_class": 120,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 5,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.9,
        "seed": 42,
        "num_threads": 4,
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=80,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(10, verbose=True),
            lgb.log_evaluation(10)
        ]
    )

    evaluate(
        "VALIDATION",
        model,
        x_valid,
        valid["label"].to_numpy()
    )

    predictions, rankings = evaluate(
        "TEST",
        model,
        x_test,
        test["label"].to_numpy()
    )

    print("\n=== SAMPLE PREDICTION ===")

    for rank, idx in enumerate(rankings[0, :10], 1):
        print(
            f"{rank:2d}. {COMBOS[idx]:5s} "
            f"{predictions[0, idx] * 100:.3f}%"
        )

    actual = test.iloc[0]["target_combo"]
    print("Actual result:", actual)

    model.save_model(
        str(MODEL_DIR / "trifecta_initial.txt")
    )

    with (MODEL_DIR / "trifecta_labels.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "combinations": COMBOS,
                "features": usable,
                "train_start": START,
                "train_end": TRAIN_END,
                "validation_day": VALID_DAY,
                "test_day": TEST_DAY
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nMODEL SAVED")
    print("Status: prototype training completed")


if __name__ == "__main__":
    main()
