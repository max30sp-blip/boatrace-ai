
import argparse
import gc
import json
import sqlite3
from itertools import permutations
from pathlib import Path

import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB_PATH = DATA / "train_dataset_v1.db"
MODEL_DIR = ROOT / "models"

CLASSES = [
    "-".join(map(str, p))
    for p in permutations(range(1, 7), 3)
]
CLASS_MAP = {name: i for i, name in enumerate(CLASSES)}
N_CLASSES = 120

PERIODS = {
    "A": ("2022-01-01", "2023-12-31"),
    "B": ("2017-01-01", "2023-12-31"),
}

VALID_START = "2024-01-01"
VALID_END = "2024-12-31"


def quote(s):
    return '"' + s.replace('"', '""') + '"'


def connect_readonly():
    if not DB_PATH.is_file():
        raise FileNotFoundError(DB_PATH)

    return sqlite3.connect(
        DB_PATH.resolve().as_uri() + "?mode=ro",
        uri=True,
        timeout=120
    )


def feature_columns(con):
    columns = [
        row[1]
        for row in con.execute(
            "PRAGMA table_info(train_races)"
        )
    ]

    excluded = {
        "race_date", "venue_code", "race_no", "label"
    }
    features = [
        name for name in columns
        if name not in excluded
    ]

    if len(features) != 618:
        raise RuntimeError(
            f"Expected 618 features, got {len(features)}"
        )

    return features


def count_period(con, start, end):
    return con.execute("""
        SELECT COUNT(*)
        FROM train_races
        WHERE race_date BETWEEN ? AND ?
    """, (start, end)).fetchone()[0]


def load_period(
    con, features, start, end, max_rows, title
):
    """
    期間全体から均等間隔で抽出する。
    max_rows が期間件数以上なら全件使用。

    先頭N件だけを使わないことで、
    検証データの年初偏りを減らす。
    """
    total = count_period(con, start, end)

    if total == 0:
        raise RuntimeError(f"{title}: no data")

    selected = min(total, max_rows)
    dimension = len(features)

    print(
        f"{title}: available={total:,}, "
        f"selected={selected:,}",
        flush=True
    )

    # float32で確保し、余計なコピーを減らす
    X = np.empty(
        (selected, dimension), dtype=np.float32
    )
    y = np.empty(selected, dtype=np.int32)

    cols = ", ".join(
        ["label"] + [quote(f) for f in features]
    )

    cursor = con.execute(f"""
        SELECT {cols}
        FROM train_races
        WHERE race_date BETWEEN ? AND ?
        ORDER BY race_date, venue_code, race_no
    """, (start, end))

    # 各抽出位置は重複しないようにする
    target_positions = (
        (np.arange(selected, dtype=np.int64) * total)
        // selected
    )

    pos = 0
    taken = 0

    for row in cursor:
        if taken >= selected:
            break

        if pos != int(target_positions[taken]):
            pos += 1
            continue

        label = row[0]

        if label not in CLASS_MAP:
            raise RuntimeError(
                f"Unknown label: {label}"
            )

        y[taken] = CLASS_MAP[label]

        X[taken, :] = np.fromiter(
            (
                np.nan if value is None
                else float(value)
                for value in row[1:]
            ),
            dtype=np.float32,
            count=dimension
        )

        taken += 1
        pos += 1

        if taken % 10000 == 0:
            print(
                f"{title}: loaded {taken:,}",
                flush=True
            )

    if taken != selected:
        raise RuntimeError(
            f"{title}: read mismatch "
            f"{taken} != {selected}"
        )

    bad = ~np.isfinite(X) & ~np.isnan(X)
    if bad.any():
        print(
            f"{title}: infinity to NaN "
            f"{int(bad.sum())}",
            flush=True
        )
        X[bad] = np.nan

    missing_rate = float(np.isnan(X).mean())

    print(
        f"{title}: ready, "
        f"missing={missing_rate:.2%}",
        flush=True
    )

    return X, y, total


def evaluate(probs, y):
    predicted = np.argmax(probs, axis=1)

    top1 = float(
        np.mean(predicted == y)
    )

    # 確率が高い上位3クラスを取得
    top3_indices = np.argpartition(
        probs, -3, axis=1
    )[:, -3:]

    top3 = float(
        np.mean(
            np.any(
                top3_indices == y[:, None],
                axis=1
            )
        )
    )

    true_probs = probs[
        np.arange(len(y)), y
    ]

    loss = float(
        -np.log(
            np.clip(true_probs, 1e-15, 1.0)
        ).mean()
    )

    return {
        "top1": top1,
        "top3": top3,
        "logloss": loss,
    }


def print_metrics(name, metrics):
    print(f"\n=== {name} ===")
    print(f"Top1:    {metrics['top1']:.4%}")
    print(f"Top3:    {metrics['top3']:.4%}")
    print(f"LogLoss: {metrics['logloss']:.6f}")


def frequency_baseline(y_train, n_valid):
    counts = np.bincount(
        y_train, minlength=N_CLASSES
    ).astype(np.float64)

    probabilities = (
        (counts + 1.0)
        / (counts.sum() + N_CLASSES)
    )

    return np.broadcast_to(
        probabilities,
        (n_valid, N_CLASSES)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        choices=["A", "B"],
        default="A"
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=400000
    )
    parser.add_argument(
        "--valid-limit",
        type=int,
        default=10000
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=60
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
            "All numeric options must be positive"
        )

    start, end = PERIODS[args.stage]

    print("=" * 66)
    print(f"PHASE 8.1-{args.stage} SCALE TRAINING")
    print("=" * 66)

    print("Train:", start, "to", end)
    print("Valid:", VALID_START, "to", VALID_END)
    print("Final test: NOT USED")
    print("Train limit:", args.train_limit)
    print("Valid limit:", args.valid_limit)
    print("Boosting rounds:", args.rounds)
    print("Threads:", args.threads)
    print("Save:", args.save, flush=True)

    con = connect_readonly()

    try:
        features = feature_columns(con)

        X_train, y_train, available_train = load_period(
            con, features,
            start, end,
            args.train_limit,
            "TRAIN"
        )

        X_valid, y_valid, available_valid = load_period(
            con, features,
            VALID_START, VALID_END,
            args.valid_limit,
            "VALID"
        )

    finally:
        con.close()

    print("\n=== DATA SHAPES ===")
    print("Train:", X_train.shape)
    print("Valid:", X_valid.shape)
    print("Train classes:", len(np.unique(y_train)))
    print("Valid classes:", len(np.unique(y_valid)))
    print(
        "Raw feature memory (MiB):",
        round(
            (X_train.nbytes + X_valid.nbytes)
            / (1024 ** 2),
            1
        )
    )

    if len(np.unique(y_train)) != N_CLASSES:
        raise RuntimeError(
            "Training data does not contain all 120 classes"
        )

    prior = frequency_baseline(
        y_train, len(y_valid)
    )

    prior_metrics = evaluate(
        prior, y_valid
    )

    print_metrics(
        "FREQUENCY BASELINE", prior_metrics
    )

    constant_123 = float(
        np.mean(
            y_valid == CLASS_MAP["1-2-3"]
        )
    )

    print(
        f"Always 1-2-3: {constant_123:.4%}"
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

    # メモリ使用量を考慮した初期設定
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": N_CLASSES,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 6,
        "min_data_in_leaf": 150,
        "max_bin": 31,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 3.0,
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
                stopping_rounds=min(12, max(3, args.rounds // 4)),
                verbose=True
            ),
            lgb.log_evaluation(period=10)
        ]
    )

    iteration = model.best_iteration or args.rounds

    print("\n=== PREDICTION ===", flush=True)

    probs = model.predict(
        X_valid,
        num_iteration=iteration,
        num_threads=args.threads
    )

    del train_set, valid_set
    gc.collect()

    probs = np.asarray(probs)

    if probs.shape != (len(y_valid), N_CLASSES):
        raise RuntimeError(
            f"Prediction shape error: {probs.shape}"
        )

    if not np.isfinite(probs).all():
        raise RuntimeError(
            "Prediction contains NaN/Infinity"
        )

    model_metrics = evaluate(
        probs, y_valid
    )

    print_metrics(
        "LIGHTGBM", model_metrics
    )

    print("\n=== IMPROVEMENT ===")
    print(
        "Top1 vs 1-2-3:",
        f"{model_metrics['top1'] - constant_123:+.4%}pt"
    )
    print(
        "Top3 vs frequency:",
        f"{model_metrics['top3'] - prior_metrics['top3']:+.4%}pt"
    )
    print(
        "LogLoss improvement:",
        f"{prior_metrics['logloss'] - model_metrics['logloss']:+.6f}"
    )

    importance = model.feature_importance(
        importance_type="gain"
    )

    order = np.argsort(
        importance
    )[::-1][:20]

    print("\n=== TOP 20 FEATURES ===")
    for i, idx in enumerate(order, 1):
        print(
            f"{i:2d}. "
            f"{features[idx]:48s}"
            f" {importance[idx]:.2f}"
        )

    report = {
        "stage": args.stage,
        "train_period": [start, end],
        "valid_period": [VALID_START, VALID_END],
        "final_test_used": False,
        "train_available": available_train,
        "valid_available": available_valid,
        "train_rows": int(len(y_train)),
        "valid_rows": int(len(y_valid)),
        "features": len(features),
        "classes": N_CLASSES,
        "best_iteration": int(iteration),
        "baseline": prior_metrics,
        "always_123_top1": constant_123,
        "lightgbm": model_metrics,
        "feature_names": features,
        "class_order": CLASSES,
        "params": params,
    }

    if args.save:
        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        suffix = f"phase08_1_{args.stage.lower()}"

        model_file = MODEL_DIR / f"{suffix}_model.txt"
        report_file = MODEL_DIR / f"{suffix}_report.json"

        model.save_model(
            str(model_file),
            num_iteration=iteration
        )

        report_file.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        print("\n=== SAVED ===")
        print(model_file)
        print(report_file)

    print(f"\nPHASE 8.1-{args.stage} COMPLETE")


if __name__ == "__main__":
    main()
