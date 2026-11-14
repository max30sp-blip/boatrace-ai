
import json
import os
import sqlite3
from pathlib import Path
from datetime import date
from collections import defaultdict, deque
from itertools import groupby

import numpy as np
import lightgbm as lgb

ROOT = Path(r"C:\boatrace_ai")

TRAIN_DB = ROOT / "data/train_dataset_v1.db"
RESULT_DB = ROOT / "data/boatrace.db"

OLD_MODEL = ROOT / "models/phase08_1_b_model.txt"
NEW_MODEL = ROOT / "models/phase84a_candidate/phase08_1_b_model.txt"

OLD_REPORT = ROOT / "models/phase08_1_b_report.json"
NEW_REPORT = ROOT / "models/phase84a_candidate/phase08_1_b_report.json"

F3_DB = ROOT / "outputs/phase84b_2025_f3.db"
TEST_REPORT = ROOT / "outputs/phase84b_2025_test.json"

HISTORY_START = "2024-09-01"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

ALPHA = 20
PRIOR = 1 / 3
WINDOWS = (30, 90)
BATCH_SIZE = 500

FEATURES = [
    "motor_30_starts",
    "motor_30_top2_smoothed",
    "motor_90_starts",
    "motor_90_top2_smoothed",
    "boat_30_starts",
    "boat_30_top2_smoothed",
    "boat_90_starts",
    "boat_90_top2_smoothed",
]

NEW_COLUMNS = [
    f"b{lane}_new_{feature}"
    for lane in range(1, 7)
    for feature in FEATURES
]


def require(path):
    if not path.is_file():
        raise FileNotFoundError(path)


def read_json(path):
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def machine_key(venue, number):
    if venue is None or number is None:
        return None

    number = int(number)

    if not 1 <= number <= 200:
        return None

    return int(venue), number


class RollingHistory:
    def __init__(self, days):
        self.days = days
        self.events = deque()
        self.motor = defaultdict(lambda: [0, 0])
        self.boat = defaultdict(lambda: [0, 0])

    def change(self, store, key, y, delta):
        if key is None:
            return

        values = store[key]
        values[0] += delta
        values[1] += delta * y

        if values[0] == 0:
            del store[key]

    def expire(self, day):
        while self.events:
            old, mk, bk, y = self.events[0]

            if (day - old).days < self.days:
                break

            self.events.popleft()
            self.change(self.motor, mk, y, -1)
            self.change(self.boat, bk, y, -1)

    def add(self, day, mk, bk, finish):
        y = int(finish <= 2)
        self.events.append((day, mk, bk, y))
        self.change(self.motor, mk, y, 1)
        self.change(self.boat, bk, y, 1)

    def features(self, kind, key):
        if key is None:
            return None, None

        store = self.motor if kind == "motor" else self.boat
        n, top2 = store.get(key, (0, 0))

        p = (top2 + ALPHA * PRIOR) / (n + ALPHA)

        return n, p


def build_2025_f3():
    if F3_DB.exists():
        print("Existing 2025 F3:", F3_DB)
        return

    temp = F3_DB.with_suffix(".building.db")

    if temp.exists():
        raise FileExistsError(
            f"Previous incomplete build exists: {temp}"
        )

    train = sqlite3.connect(str(TRAIN_DB))

    targets = set(train.execute("""
        SELECT race_date, venue_code, race_no
        FROM train_races
        WHERE race_date BETWEEN ? AND ?
    """, (TEST_START, TEST_END)).fetchall())

    train.close()

    print("\n=== BUILDING 2025 F3 ===", flush=True)
    print("Target races:", len(targets), flush=True)

    source = sqlite3.connect(str(RESULT_DB))
    output = sqlite3.connect(str(temp))

    cols = ", ".join(
        f'"{c}" REAL' for c in NEW_COLUMNS
    )

    output.execute(f"""
        CREATE TABLE wide_f3 (
            race_date TEXT NOT NULL,
            venue_code INTEGER NOT NULL,
            race_no INTEGER NOT NULL,
            f3_matched INTEGER NOT NULL,
            {cols},
            PRIMARY KEY (
                race_date, venue_code, race_no
            )
        ) WITHOUT ROWID
    """)

    insert_cols = 4 + len(NEW_COLUMNS)
    placeholders = ",".join("?" for _ in range(insert_cols))

    insert_sql = (
        "INSERT INTO wide_f3 VALUES "
        f"({placeholders})"
    )

    cursor = source.execute("""
        SELECT
            race_date, venue_code, race_no,
            lane, motor_no, boat_no,
            finish_position
        FROM race_boat_results_v2
        WHERE race_date BETWEEN ? AND ?
        ORDER BY race_date, venue_code, race_no, lane
    """, (HISTORY_START, TEST_END))

    histories = {
        w: RollingHistory(w) for w in WINDOWS
    }

    batch = []
    matched = set()
    invalid = 0

    for day_str, day_group in groupby(
        cursor, key=lambda r: r[0]
    ):
        day = date.fromisoformat(day_str)
        day_rows = list(day_group)

        for h in histories.values():
            h.expire(day)

        # 当日の着順を学習履歴へ追加する前に計算
        if TEST_START <= day_str <= TEST_END:
            race_rows = defaultdict(dict)

            for row in day_rows:
                d, venue, race, lane, motor, boat, finish = row
                race_key = (d, venue, race)

                if race_key not in targets:
                    continue

                if lane in race_rows[race_key]:
                    raise ValueError(
                        f"Duplicate lane: {race_key} {lane}"
                    )

                race_rows[race_key][lane] = row

            for race_key, lanes in race_rows.items():
                if set(lanes) != set(range(1, 7)):
                    continue

                values = list(race_key) + [1]

                for lane in range(1, 7):
                    row = lanes[lane]
                    venue = row[1]
                    mk = machine_key(venue, row[4])
                    bk = machine_key(venue, row[5])

                    m30 = histories[30].features("motor", mk)
                    m90 = histories[90].features("motor", mk)
                    b30 = histories[30].features("boat", bk)
                    b90 = histories[90].features("boat", bk)

                    values.extend(
                        (*m30, *m90, *b30, *b90)
                    )

                batch.append(tuple(values))
                matched.add(race_key)

                if len(batch) >= 5000:
                    output.executemany(insert_sql, batch)
                    output.commit()
                    batch.clear()

        # 全レースの特徴量計算後に更新
        for row in day_rows:
            _, venue, _, _, motor, boat, finish = row

            if finish is None or not 1 <= finish <= 6:
                invalid += 1
                continue

            mk = machine_key(venue, motor)
            bk = machine_key(venue, boat)

            for h in histories.values():
                h.add(day, mk, bk, finish)

        if day.day == 1:
            print(
                "Processed:", day_str,
                "matched:", len(matched),
                flush=True
            )

    if batch:
        output.executemany(insert_sql, batch)

    output.commit()

    actual = output.execute(
        "SELECT COUNT(*) FROM wide_f3"
    ).fetchone()[0]

    source.close()
    output.close()

    if actual != len(matched):
        raise ValueError("Generated row count mismatch")

    if actual == 0:
        raise ValueError("No 2025 F3 rows generated")

    os.replace(temp, F3_DB)

    print("\n=== 2025 F3 COMPLETE ===")
    print("Target races:", len(targets))
    print("Matched races:", actual)
    print("Unmatched races:", len(targets) - actual)
    print("Invalid historical finishes:", invalid)
    print("Output:", F3_DB, flush=True)


def evaluate():
    if TEST_REPORT.exists():
        raise FileExistsError(
            f"Test report already exists: {TEST_REPORT}\n"
            "2025年テストを繰り返し最適化しないため停止します。"
        )

    old_report = read_json(OLD_REPORT)
    new_report = read_json(NEW_REPORT)

    old_features = old_report["feature_names"]
    new_features = new_report["feature_names"]

    if len(old_features) != 618:
        raise ValueError("Old features mismatch")

    if new_features != old_features + NEW_COLUMNS:
        raise ValueError("New feature order mismatch")

    if old_report["class_order"] != new_report["class_order"]:
        raise ValueError("Class order mismatch")

    if old_report["train_period"] != [
        "2017-01-01", "2023-12-31"
    ]:
        raise ValueError("Unexpected training period")

    if new_report["train_period"] != old_report["train_period"]:
        raise ValueError("Training periods differ")

    if new_report["valid_period"] != old_report["valid_period"]:
        raise ValueError("Validation periods differ")

    class_map = {
        name: i
        for i, name in enumerate(old_report["class_order"])
    }

    old_model = lgb.Booster(model_file=str(OLD_MODEL))
    new_model = lgb.Booster(model_file=str(NEW_MODEL))

    if old_model.feature_name() != old_features:
        raise ValueError("Old model feature order mismatch")

    if new_model.feature_name() != new_features:
        raise ValueError("New model feature order mismatch")

    con = sqlite3.connect(str(TRAIN_DB))

    con.execute(
        "ATTACH DATABASE ? AS f3",
        (str(F3_DB),)
    )

    def quote(name):
        return '"' + name.replace('"', '""') + '"'

    selected = (
        [f"t.{quote(c)}" for c in old_features]
        + [f"w.{quote(c)}" for c in NEW_COLUMNS]
    )

    query = f"""
        SELECT
            w.f3_matched,
            t.label,
            {", ".join(selected)}
        FROM main.train_races t
        LEFT JOIN f3.wide_f3 w
          ON t.race_date = w.race_date
         AND t.venue_code = w.venue_code
         AND t.race_no = w.race_no
        WHERE t.race_date BETWEEN ? AND ?
        ORDER BY t.race_date, t.venue_code, t.race_no
    """

    cursor = con.execute(
        query, (TEST_START, TEST_END)
    )

    metrics = {
        "old": {
            "top1": 0, "top3": 0, "loss": 0.0
        },
        "new": {
            "top1": 0, "top3": 0, "loss": 0.0
        }
    }

    total = 0
    matched = 0

    print("\n=== 2025 OUT-OF-TIME TEST ===", flush=True)

    while True:
        rows = cursor.fetchmany(BATCH_SIZE)

        if not rows:
            break

        labels = [r[1] for r in rows]

        if any(label not in class_map for label in labels):
            raise ValueError("Unknown label in 2025 data")

        y = np.asarray(
            [class_map[label] for label in labels],
            dtype=np.int32
        )

        X_new = np.asarray([
            [
                np.nan if v is None else float(v)
                for v in row[2:]
            ]
            for row in rows
        ], dtype=np.float32)

        if X_new.shape[1] != 666:
            raise ValueError("Expected 666 columns")

        X_old = X_new[:, :618]

        predictions = {
            "old": old_model.predict(
                X_old,
                num_iteration=old_report["best_iteration"],
                num_threads=4
            ),
            "new": new_model.predict(
                X_new,
                num_iteration=new_report["best_iteration"],
                num_threads=4
            )
        }

        indices = np.arange(len(y))

        for name, probs in predictions.items():
            if probs.shape != (len(y), 120):
                raise ValueError("Invalid probability shape")

            if not np.isfinite(probs).all():
                raise ValueError("Non-finite probabilities")

            top1 = np.argmax(probs, axis=1)
            top3 = np.argpartition(
                probs, -3, axis=1
            )[:, -3:]

            true_probs = np.clip(
                probs[indices, y], 1e-15, 1.0
            )

            metrics[name]["top1"] += int(
                np.sum(top1 == y)
            )

            metrics[name]["top3"] += int(
                np.sum(
                    np.any(top3 == y[:, None], axis=1)
                )
            )

            metrics[name]["loss"] += float(
                -np.log(true_probs).sum()
            )

        matched += sum(
            row[0] == 1 for row in rows
        )

        total += len(rows)

        if total % 5000 == 0:
            print(
                "Evaluated:", total,
                flush=True
            )

    con.close()

    if total == 0:
        raise ValueError("No 2025 evaluation rows")

    results = {}

    for name, m in metrics.items():
        results[name] = {
            "top1": m["top1"] / total,
            "top3": m["top3"] / total,
            "logloss": m["loss"] / total,
            "top1_hits": m["top1"],
            "top3_hits": m["top3"]
        }

    print("\n=== 2025 TEST RESULTS ===")
    print("Test races:", total)
    print("F3 matched:", matched)
    print("F3 missing:", total - matched)

    for name in ("old", "new"):
        m = results[name]

        print(f"\n{name.upper()} MODEL")
        print("Top1:", f"{m['top1']:.4%}")
        print("Top3:", f"{m['top3']:.4%}")
        print("LogLoss:", f"{m['logloss']:.6f}")

    print("\n=== NEW MINUS OLD ===")

    for metric in ("top1", "top3", "logloss"):
        difference = (
            results["new"][metric]
            - results["old"][metric]
        )

        print(metric, f"{difference:+.6f}")

    record = {
        "test_period": [TEST_START, TEST_END],
        "test_races": total,
        "f3_matched": matched,
        "baseline": results["old"],
        "candidate": results["new"],
        "reset_policy": "unverified_rolling_proxy",
        "trained_models_unchanged": True
    }

    TEST_REPORT.parent.mkdir(
        parents=True, exist_ok=True
    )

    with TEST_REPORT.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            record, f, indent=2,
            ensure_ascii=False
        )

    print("\n=== PHASE 8.4-B COMPLETE ===")
    print("Report:", TEST_REPORT)


def main():
    required = (
        TRAIN_DB, RESULT_DB,
        OLD_MODEL, NEW_MODEL,
        OLD_REPORT, NEW_REPORT
    )

    for path in required:
        require(path)

    build_2025_f3()
    evaluate()


if __name__ == "__main__":
    main()
