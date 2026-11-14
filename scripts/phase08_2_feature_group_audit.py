
import argparse
import json
import sqlite3
from pathlib import Path
from collections import defaultdict

import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

DB_PATH = DATA / "train_dataset_v1.db"
MODEL_PATH = MODELS / "phase08_1_b_model.txt"
REPORT_PATH = MODELS / "phase08_1_b_report.json"

GROUPS = {
    "P": "番組表・選手基本情報",
    "F0": "選手の過去成績",
    "F1": "進入コース・当地成績",
    "F2": "艇番別成績",
    "F3": "モーター・ボート成績",
    "F4": "競艇場・コース傾向",
    "F5": "直近5・10・20走",
    "F6": "今節成績・今節ST",
}


def q(name):
    return '"' + name.replace('"', '""') + '"'


def connect_ro(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True
    )


def feature_group(name):
    # 例: b1_f3_motor_win_rate_30d
    parts = name.split("_", 2)

    if len(parts) < 3:
        return None

    lane, prefix, rest = parts

    if lane not in {f"b{i}" for i in range(1, 7)}:
        return None

    if prefix == "p":
        return "P"

    if prefix in {f"f{i}" for i in range(7)}:
        return prefix.upper()

    return None


def load_validation(con, features, limit):
    """
    2024年の年間全体から均等に抽出。
    PHASE 8.1の抽出方法と合わせる。
    """

    total = con.execute("""
        SELECT COUNT(*)
        FROM train_races
        WHERE race_date BETWEEN
              '2024-01-01' AND '2024-12-31'
    """).fetchone()[0]

    n = min(total, limit)

    if n <= 0:
        raise RuntimeError("No validation data")

    X = np.empty(
        (n, len(features)), dtype=np.float32
    )
    y = []

    positions = (
        np.arange(n, dtype=np.int64) * total
    ) // n

    select_cols = ", ".join(
        ["label"] + [q(c) for c in features]
    )

    sql = f"""
        SELECT {select_cols}
        FROM train_races
        WHERE race_date BETWEEN
              '2024-01-01' AND '2024-12-31'
        ORDER BY race_date, venue_code, race_no
    """

    cursor = con.execute(sql)

    pos = 0
    taken = 0

    for row in cursor:
        if taken >= n:
            break

        if pos != int(positions[taken]):
            pos += 1
            continue

        y.append(row[0])

        X[taken] = np.fromiter(
            (
                np.nan if v is None else float(v)
                for v in row[1:]
            ),
            dtype=np.float32,
            count=len(features)
        )

        taken += 1
        pos += 1

        if taken % 1000 == 0:
            print(
                f"Loaded validation: {taken:,}",
                flush=True
            )

    if taken != n:
        raise RuntimeError(
            f"Validation read mismatch: {taken}/{n}"
        )

    X[np.isinf(X)] = np.nan

    return X, y


def metrics(probs, y):
    probs = np.asarray(probs, dtype=np.float64)

    if probs.shape != (len(y), 120):
        raise RuntimeError(
            f"Unexpected probability shape: {probs.shape}"
        )

    if not np.isfinite(probs).all():
        raise RuntimeError("Non-finite probabilities")

    correct_prob = probs[
        np.arange(len(y)), y
    ]

    loss = float(
        -np.log(
            np.clip(correct_prob, 1e-15, 1.0)
        ).mean()
    )

    top1 = float(
        np.mean(np.argmax(probs, axis=1) == y)
    )

    top3 = np.argpartition(
        probs, -3, axis=1
    )[:, -3:]

    top3_acc = float(
        np.mean(
            np.any(top3 == y[:, None], axis=1)
        )
    )

    return {
        "logloss": loss,
        "top1": top1,
        "top3": top3_acc,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--valid-limit",
        type=int,
        default=3000
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=2
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4
    )

    args = parser.parse_args()

    if min(
        args.valid_limit,
        args.repeats,
        args.threads
    ) <= 0:
        raise ValueError("Arguments must be positive")

    print("=" * 65)
    print("PHASE 8.2 FEATURE GROUP PERMUTATION AUDIT")
    print("=" * 65)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    if not REPORT_PATH.is_file():
        raise FileNotFoundError(REPORT_PATH)

    report = json.loads(
        REPORT_PATH.read_text(encoding="utf-8")
    )

    features = report["feature_names"]
    classes = report["class_order"]

    if len(features) != 618 or len(classes) != 120:
        raise RuntimeError("Invalid saved metadata")

    class_to_id = {
        name: i for i, name in enumerate(classes)
    }

    model = lgb.Booster(
        model_file=str(MODEL_PATH)
    )

    if model.feature_name() != features:
        raise RuntimeError(
            "Saved model feature order does not match report"
        )

    print("Model:", MODEL_PATH.name)
    print("Model iteration:", model.current_iteration())
    print("Features:", len(features))
    print("Validation period: 2024")
    print("Validation sample:", args.valid_limit)
    print("Permutation repeats:", args.repeats)

    columns_by_group = defaultdict(list)
    unknown = []

    for idx, name in enumerate(features):
        group = feature_group(name)

        if group is None:
            unknown.append(name)
        else:
            columns_by_group[group].append(idx)

    if unknown:
        print("Unknown feature names:", unknown[:20])
        raise RuntimeError(
            "Unclassified features found"
        )

    print("\n=== FEATURE GROUP COUNTS ===")

    for group, description in GROUPS.items():
        print(
            f"{group:3s} "
            f"{len(columns_by_group[group]):3d} "
            f"{description}"
        )

    con = connect_ro(DB_PATH)

    try:
        X, labels = load_validation(
            con, features, args.valid_limit
        )
    finally:
        con.close()

    y = np.array(
        [class_to_id[label] for label in labels],
        dtype=np.int32
    )

    print("\n=== ORIGINAL PERFORMANCE ===")

    original_prob = model.predict(
        X,
        num_threads=args.threads
    )

    baseline = metrics(original_prob, y)

    print(
        f"Top1: {baseline['top1']:.4%}"
    )
    print(
        f"Top3: {baseline['top3']:.4%}"
    )
    print(
        f"LogLoss: {baseline['logloss']:.6f}"
    )

    print("\n=== PERMUTATION TEST ===")
    print(
        "Positive delta LogLoss means performance"
        " became worse after shuffling."
    )

    results = []
    rng = np.random.default_rng(42)

    for group, description in GROUPS.items():

        indices = columns_by_group[group]

        if not indices:
            continue

        losses = []
        top1_values = []
        top3_values = []

        for repeat in range(args.repeats):

            # 元の配列を変更しない
            X_test = X.copy()

            permutation = rng.permutation(
                len(X_test)
            )

            # 同グループの全列を同じ順番で入れ替える
            # グループ内の列同士の関係は保持する
            X_test[:, indices] = X[
                permutation[:, None],
                np.asarray(indices)[None, :]
            ]

            probs = model.predict(
                X_test,
                num_threads=args.threads
            )

            result = metrics(probs, y)

            losses.append(result["logloss"])
            top1_values.append(result["top1"])
            top3_values.append(result["top3"])

        shuffled_loss = float(np.mean(losses))
        shuffled_top1 = float(np.mean(top1_values))
        shuffled_top3 = float(np.mean(top3_values))

        delta_loss = (
            shuffled_loss - baseline["logloss"]
        )

        delta_top1 = (
            shuffled_top1 - baseline["top1"]
        )

        delta_top3 = (
            shuffled_top3 - baseline["top3"]
        )

        item = {
            "group": group,
            "description": description,
            "columns": len(indices),
            "shuffled_logloss": shuffled_loss,
            "delta_logloss": delta_loss,
            "delta_top1": delta_top1,
            "delta_top3": delta_top3,
        }

        results.append(item)

        print(
            f"{group:3s} {description:20s}"
            f" | delta_loss={delta_loss:+.6f}"
            f" | delta_top1={delta_top1:+.2%}pt"
            f" | delta_top3={delta_top3:+.2%}pt",
            flush=True
        )

    print("\n=== RANKING BY DELTA LOGLOSS ===")

    results.sort(
        key=lambda x: x["delta_logloss"],
        reverse=True
    )

    for rank, item in enumerate(results, 1):
        print(
            f"{rank:2d}. "
            f"{item['group']:3s} "
            f"{item['description']:20s} "
            f"{item['delta_logloss']:+.6f}"
        )

    print("\n=== INTERPRETATION ===")
    print("Large positive delta: model depends on group")
    print("Near zero: limited marginal effect in this test")
    print("Negative delta: investigate possible harmful features")
    print(
        "This is permutation importance, "
        "not a leakage proof or causal effect."
    )

    print("\nPHASE 8.2-A COMPLETE")


if __name__ == "__main__":
    main()
