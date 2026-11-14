
import itertools
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "training" / "trifecta_dataset.csv"
OUT = ROOT / "models" / "conditional_v1"
OUT.mkdir(parents=True, exist_ok=True)

COMBOS = list(itertools.permutations(range(1, 7), 3))
COMBO_ID = {"-".join(map(str, c)): i for i, c in enumerate(COMBOS)}

START = "2026-10-23"
TRAIN_END = "2026-10-27"
VALID_DATE = "2026-10-28"
TEST_DATE = "2026-10-29"


def load_data():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df = df[df.race_date.between(START, TEST_DATE)].copy()

    excluded = {"race_date", "race_no", "target_combo"}
    cols = [c for c in df.columns if c not in excluded]

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if not df.target_combo.isin(COMBO_ID).all():
        raise ValueError("Invalid trifecta label")

    train = df[df.race_date <= TRAIN_END].copy()
    valid = df[df.race_date == VALID_DATE].copy()
    test = df[df.race_date == TEST_DATE].copy()

    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Empty train/valid/test split")

    usable = [c for c in cols if train[c].notna().any()]

    return train, valid, test, usable


def labels(df):
    return np.array(
        [list(map(int, x.split("-"))) for x in df.target_combo],
        dtype=np.int32
    )


def make_inputs(df, cols, stage, prefixes=None):
    x = df[cols].copy().reset_index(drop=True)

    if stage >= 2:
        x["first_lane"] = prefixes[:, 0]

    if stage >= 3:
        x["second_lane"] = prefixes[:, 1]

    return x


def train_models(train, valid, cols):
    train_y = labels(train)
    valid_y = labels(valid)
    models = []

    for stage in range(1, 4):
        x_train = make_inputs(
            train, cols, stage, train_y
        )
        x_valid = make_inputs(
            valid, cols, stage, valid_y
        )

        y_train = train_y[:, stage - 1] - 1
        y_valid = valid_y[:, stage - 1] - 1

        params = {
            "objective": "multiclass",
            "num_class": 6,
            "metric": "multi_logloss",
            "learning_rate": 0.04,
            "num_leaves": 15,
            "max_depth": 4,
            "min_data_in_leaf": 15,
            "seed": 42,
            "num_threads": 4,
            "verbosity": -1,
        }

        ds_train = lgb.Dataset(
            x_train, label=y_train
        )
        ds_valid = lgb.Dataset(
            x_valid, label=y_valid,
            reference=ds_train
        )

        print(f"\n=== TRAIN STAGE {stage} ===")

        model = lgb.train(
            params,
            ds_train,
            num_boost_round=120,
            valid_sets=[ds_valid],
            callbacks=[
                lgb.early_stopping(15, verbose=True),
                lgb.log_evaluation(20)
            ]
        )

        model.save_model(
            str(OUT / f"stage{stage}.txt")
        )
        models.append(model)

    return models


def masked_prob(model, x, forbidden):
    p = np.asarray(model.predict(x), dtype=float)
    p = p.reshape(-1, 6)

    p = np.clip(p, 0, None)

    for row_index, lanes in enumerate(forbidden):
        for lane in lanes:
            p[row_index, lane - 1] = 0.0

    totals = p.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Invalid conditional probabilities")

    return p / totals


def predict_all(models, df, cols):
    n = len(df)
    base = df[cols].reset_index(drop=True)

    first_p = masked_prob(
        models[0], base, [[] for _ in range(n)]
    )

    # 2着: 1着候補6通りをまとめて予測
    second_base = pd.concat([base] * 6, ignore_index=True)
    first_prefix = np.concatenate([
        np.full(n, a) for a in range(1, 7)
    ])

    second_x = second_base.copy()
    second_x["first_lane"] = first_prefix

    second_p = masked_prob(
        models[1],
        second_x,
        [[int(a)] for a in first_prefix]
    )

    # 3着: 1着・2着の候補30通り
    pairs = [
        (a, b)
        for a in range(1, 7)
        for b in range(1, 7)
        if a != b
    ]

    third_base = pd.concat(
        [base] * len(pairs), ignore_index=True
    )

    third_first = np.concatenate([
        np.full(n, a) for a, b in pairs
    ])
    third_second = np.concatenate([
        np.full(n, b) for a, b in pairs
    ])

    third_x = third_base.copy()
    third_x["first_lane"] = third_first
    third_x["second_lane"] = third_second

    third_p = masked_prob(
        models[2],
        third_x,
        [
            [int(a), int(b)]
            for a, b in zip(third_first, third_second)
        ]
    )

    pair_index = {
        pair: i for i, pair in enumerate(pairs)
    }

    probabilities = np.zeros((n, 120))

    for combo_index, (a, b, c) in enumerate(COMBOS):
        second_index = (a - 1) * n
        third_index = pair_index[(a, b)] * n

        probabilities[:, combo_index] = (
            first_p[:, a - 1]
            * second_p[second_index:second_index+n, b-1]
            * third_p[third_index:third_index+n, c-1]
        )

    if not np.allclose(
        probabilities.sum(axis=1), 1, atol=1e-5
    ):
        raise ValueError("120 probabilities do not sum to 1")

    return probabilities


def evaluate(name, probabilities, df):
    target = np.array([
        COMBO_ID[c] for c in df.target_combo
    ])

    order = np.argsort(-probabilities, axis=1)

    print(f"\n=== {name} ===")
    print("Races:", len(df))

    for k in (1, 3, 5):
        hit = np.any(
            order[:, :k] == target[:, None], axis=1
        )
        print(f"Top {k}: {100 * hit.mean():.2f}%")

    p_true = probabilities[
        np.arange(len(target)), target
    ]
    logloss = -np.log(
        np.clip(p_true, 1e-15, 1)
    ).mean()

    print(f"Log loss: {logloss:.5f}")

    return order


def main():
    train, valid, test, cols = load_data()

    print("========== PHASE 8.2 ==========")
    print("Train:", len(train))
    print("Validation:", len(valid))
    print("Test:", len(test))
    print("Base features:", len(cols))

    models = train_models(train, valid, cols)

    val_p = predict_all(models, valid, cols)
    test_p = predict_all(models, test, cols)

    evaluate("VALIDATION", val_p, valid)
    order = evaluate("TEST", test_p, test)

    print("\n=== SAMPLE TOP 10 ===")
    for rank, idx in enumerate(order[0, :10], 1):
        combo = "-".join(map(str, COMBOS[idx]))
        print(
            f"{rank:2d}. {combo}: "
            f"{test_p[0, idx] * 100:.3f}%"
        )

    print("Actual:", test.iloc[0].target_combo)

    with (OUT / "metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump({
            "features": cols,
            "combinations": [
                "-".join(map(str, c)) for c in COMBOS
            ],
            "train_end": TRAIN_END,
            "validation_date": VALID_DATE,
            "test_date": TEST_DATE,
        }, f, indent=2, ensure_ascii=False)

    print("\nPHASE 8.2 COMPLETE")
    print("Models:", OUT)


if __name__ == "__main__":
    main()
