
import argparse
import json
import sqlite3
from itertools import permutations
from pathlib import Path

import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
DB = DATA / "train_dataset_v1.db"

CLASSES = [
    "-".join(map(str, p))
    for p in permutations(range(1, 7), 3)
]
CLASS_TO_ID = {
    label: i for i, label in enumerate(CLASSES)
}
NUM_CLASSES = len(CLASSES)


def quote(value):
    return '"' + value.replace('"', '""') + '"'


def open_readonly(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True,
        timeout=120
    )


def get_feature_columns(con):
    info = list(con.execute(
        "PRAGMA table_info(train_races)"
    ))

    if not info:
        raise RuntimeError("train_races table not found")

    exclude = {
        "race_date",
        "venue_code",
        "race_no",
        "label",
    }

    names = [
        row[1]
        for row in info
        if row[1] not in exclude
    ]

    if len(names) != 618:
        raise RuntimeError(
            f"Expected 618 features, found {len(names)}"
        )

    return names


def read_period(
    con,
    features,
    start,
    end,
    limit,
    title
):
    """
    SQLiteから指定件数だけ読み込む。
    大量のDataFrameを作らずfloat32で保持。
    """
    select_cols = ["label"] + [
        quote(c) for c in features
    ]

    sql = f"""
        SELECT {", ".join(select_cols)}
        FROM train_races
        WHERE race_date BETWEEN ? AND ?
        ORDER BY race_date, venue_code, race_no
        LIMIT ?
    """

    cursor = con.execute(
        sql, (start, end, limit)
    )

    n_features = len(features)

    X = np.empty(
        (limit, n_features),
        dtype=np.float32
    )
    y = np.empty(limit, dtype=np.int32)

    count = 0

    for row in cursor:
        label = row[0]

        if label not in CLASS_TO_ID:
            raise RuntimeError(
                f"Invalid label: {label}"
            )

        y[count] = CLASS_TO_ID[label]

        # 欠損値はNaNとして保持
        X[count] = np.fromiter(
            (
                np.nan if value is None
                else float(value)
                for value in row[1:]
            ),
            dtype=np.float32,
            count=n_features
        )

        count += 1

        if count % 10000 == 0:
            print(
                f"{title}: loaded {count:,}",
                flush=True
            )

    if count == 0:
        raise RuntimeError(
            f"No rows found for {title}"
        )

    X = X[:count].copy()
    y = y[:count].copy()

    infinity = np.isinf(X)

    if infinity.any():
        print(
            f"{title}: converting "
            f"{int(infinity.sum())} infinite values to NaN"
        )
        X[infinity] = np.nan

    print(
        f"{title}: {count:,} races, "
        f"{n_features} features",
        flush=True
    )

    return X, y


def accuracy_top1(probs, true_y):
    predicted = np.argmax(probs, axis=1)
    return float(np.mean(predicted == true_y))


def accuracy_top3(probs, true_y):
    top3 = np.argpartition(
        probs, kth=-3, axis=1
    )[:, -3:]

    correct = np.any(
        top3 == true_y[:, None],
        axis=1
    )

    return float(np.mean(correct))


def log_loss(probs, true_y):
    selected = probs[
        np.arange(len(true_y)), true_y
    ]

    selected = np.clip(
        selected, 1e-15, 1.0
    )

    return float(
        -np.mean(np.log(selected))
    )


def evaluate(name, probs, true_y):
    top1 = accuracy_top1(probs, true_y)
    top3 = accuracy_top3(probs, true_y)
    loss = log_loss(probs, true_y)

    print(f"\n=== {name} ===")
    print(f"Top1 accuracy: {top1:.4%}")
    print(f"Top3 accuracy: {top3:.4%}")
    print(f"Log loss:      {loss:.6f}")

    return {
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "log_loss": loss,
    }


def prior_baseline(y_train, n_valid):
    """
    学習データ内の3連単出現頻度だけで予測。
    Laplace平滑化で全クラスに確率を与える。
    """
    counts = np.bincount(
        y_train,
        minlength=NUM_CLASSES
    ).astype(np.float64)

    prior = (counts + 1.0) / (
        counts.sum() + NUM_CLASSES
    )

    return np.broadcast_to(
        prior, (n_valid, NUM_CLASSES)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-limit",
        type=int,
        default=30000
    )
    parser.add_argument(
        "--valid-limit",
        type=int,
        default=5000
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=80
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4
    )
    parser.add_argument(
        "--save",
        action="store_true"
    )

    args = parser.parse_args()

    if min(
        args.train_limit,
        args.valid_limit,
        args.rounds,
        args.threads
    ) <= 0:
        raise ValueError(
            "All numeric arguments must be positive"
        )

    print("=" * 65)
    print("PHASE 8.0 LIGHTGBM TRIFECTA BASELINE")
    print("=" * 65)

    print("Training period: 2023")
    print("Validation period: 2024")
    print("Final test period: NOT USED")
    print("Train limit:", args.train_limit)
    print("Valid limit:", args.valid_limit)
    print("Classes:", NUM_CLASSES)
    print("Boosting rounds:", args.rounds)
    print("Threads:", args.threads)
    print("Save model:", args.save)

    con = open_readonly(DB)

    try:
        features = get_feature_columns(con)

        print("\n=== LOADING DATA ===", flush=True)

        X_train, y_train = read_period(
            con,
            features,
            "2023-01-01",
            "2023-12-31",
            args.train_limit,
            "TRAIN"
        )

        X_valid, y_valid = read_period(
            con,
            features,
            "2024-01-01",
            "2024-12-31",
            args.valid_limit,
            "VALID"
        )

    finally:
        con.close()

    print("\n=== CLASS COVERAGE ===")
    print(
        "Train classes:",
        len(np.unique(y_train))
    )
    print(
        "Valid classes:",
        len(np.unique(y_valid))
    )

    train_null_rate = float(
        np.mean(np.isnan(X_train))
    )
    valid_null_rate = float(
        np.mean(np.isnan(X_valid))
    )

    print(
        f"Train missing cells: {train_null_rate:.2%}"
    )
    print(
        f"Valid missing cells: {valid_null_rate:.2%}"
    )

    print("\n=== CONSTANT 1-2-3 BASELINE ===")

    constant_label = CLASS_TO_ID["1-2-3"]

    constant_hits = np.mean(
        y_valid == constant_label
    )

    print(
        f"Always 1-2-3 Top1: {constant_hits:.4%}"
    )

    baseline_probs = prior_baseline(
        y_train, len(y_valid)
    )

    prior_metrics = evaluate(
        "HISTORICAL FREQUENCY BASELINE",
        baseline_probs,
        y_valid
    )

    print("\n=== LIGHTGBM TRAINING ===", flush=True)

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=features,
        free_raw_data=True
    )

    valid_set = lgb.Dataset(
        X_valid,
        label=y_valid,
        reference=train_set,
        free_raw_data=True
    )

    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": NUM_CLASSES,

        "learning_rate": 0.07,
        "num_leaves": 15,
        "max_depth": 6,
        "min_data_in_leaf": 100,
        "max_bin": 63,

        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,

        "lambda_l2": 2.0,
        "num_threads": args.threads,
        "force_col_wise": True,

        "seed": 42,
        "feature_fraction_seed": 42,
        "bagging_seed": 42,
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=args.rounds,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=min(
                    15, max(3, args.rounds // 4)
                ),
                verbose=True
            ),
            lgb.log_evaluation(period=10)
        ]
    )

    best_iteration = model.best_iteration

    if not best_iteration:
        best_iteration = args.rounds

    print("\n=== VALIDATION PREDICTION ===")

    probs = model.predict(
        X_valid,
        num_iteration=best_iteration,
        num_threads=args.threads
    )

    probs = np.asarray(probs)

    if probs.shape != (
        len(y_valid), NUM_CLASSES
    ):
        raise RuntimeError(
            f"Unexpected prediction shape: {probs.shape}"
        )

    if not np.isfinite(probs).all():
        raise RuntimeError(
            "Prediction contains NaN or infinity"
        )

    metrics = evaluate(
        "LIGHTGBM",
        probs,
        y_valid
    )

    print("\n=== COMPARISON ===")

    improvement_top1 = (
        metrics["top1_accuracy"]
        - float(constant_hits)
    )

    improvement_loss = (
        prior_metrics["log_loss"]
        - metrics["log_loss"]
    )

    print(
        "Top1 vs always 1-2-3:",
        f"{improvement_top1:+.4%}pt"
    )

    print(
        "Log loss improvement vs frequency:",
        f"{improvement_loss:+.6f}"
    )

    print("\n=== FEATURE IMPORTANCE TOP 25 ===")

    importance = model.feature_importance(
        importance_type="gain"
    )

    sorted_indices = np.argsort(
        importance
    )[::-1][:25]

    for rank, idx in enumerate(
        sorted_indices, 1
    ):
        print(
            f"{rank:2d}. "
            f"{features[idx]:50s}"
            f" {importance[idx]:.2f}"
        )

    print("\n=== EXAMPLE PREDICTIONS ===")

    for i in range(min(10, len(y_valid))):
        top = np.argsort(
            probs[i]
        )[::-1][:3]

        predictions = [
            (
                CLASSES[k],
                round(float(probs[i, k]), 4)
            )
            for k in top
        ]

        print(
            f"Race {i+1:2d}"
            f" actual={CLASSES[y_valid[i]]}"
            f" predicted={predictions}"
        )

    report = {
        "phase": "8.0",
        "status": "provisional_baseline",
        "train_period": "2023",
        "validation_period": "2024",
        "final_test_used": False,
        "train_races": int(len(y_train)),
        "valid_races": int(len(y_valid)),
        "features": len(features),
        "num_classes": NUM_CLASSES,
        "best_iteration": int(best_iteration),
        "train_class_coverage": int(
            len(np.unique(y_train))
        ),
        "constant_123_top1": float(constant_hits),
        "frequency_baseline": prior_metrics,
        "lightgbm": metrics,
        "feature_names": features,
        "class_order": CLASSES,
        "params": params,
    }

    if args.save:
        MODELS.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = (
            MODELS / "phase08_baseline_model.txt"
        )
        report_path = (
            MODELS / "phase08_baseline_report.json"
        )

        model.save_model(
            str(model_path),
            num_iteration=best_iteration
        )

        report_path.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        print("\n=== SAVED ===")
        print("Model:", model_path)
        print("Report:", report_path)
    else:
        print(
            "\nModel not saved. "
            "Add --save to save the baseline."
        )

    print("\nPHASE 8.0 BASELINE COMPLETE")


if __name__ == "__main__":
    main()
