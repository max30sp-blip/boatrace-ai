
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "training" / "trifecta_dataset.csv"

COMBOS = [
    "-".join(map(str, c))
    for c in itertools.permutations(range(1, 7), 3)
]

COMBO_ID = {c: i for i, c in enumerate(COMBOS)}


def evaluate(name, data, probabilities):
    labels = data["target_combo"].map(COMBO_ID).to_numpy()

    p = np.asarray(probabilities, dtype=float)
    order = np.argsort(-p)

    print(f"\n=== {name} ===")
    print("Races:", len(labels))

    for k in (1, 3, 5):
        accuracy = np.isin(labels, order[:k]).mean()
        print(f"Top {k}: {accuracy * 100:.2f}%")

    logloss = -np.log(
        np.clip(p[labels], 1e-15, 1)
    ).mean()

    print(f"Log loss: {logloss:.5f}")


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")

    df = df[
        (df["race_date"] >= "2026-10-23")
        & (df["race_date"] <= "2026-10-29")
    ].copy()

    train = df[df["race_date"] <= "2026-10-27"]
    valid = df[df["race_date"] == "2026-10-28"]
    test = df[df["race_date"] == "2026-10-29"]

    if min(len(train), len(valid), len(test)) == 0:
        raise RuntimeError("Dataset split is empty")

    # 未出現の組み合わせにも確率を与える
    counts = train["target_combo"].value_counts()
    alpha = 1.0

    frequency = np.array([
        counts.get(c, 0) + alpha for c in COMBOS
    ], dtype=float)

    frequency /= frequency.sum()

    uniform = np.full(120, 1 / 120)

    print("========== PHASE 8.1 ==========")
    print("Train:", len(train))
    print("Validation:", len(valid))
    print("Test:", len(test))

    for name, subset in (
        ("VALIDATION", valid),
        ("TEST", test)
    ):
        evaluate(name + " / UNIFORM", subset, uniform)
        evaluate(name + " / FREQUENCY", subset, frequency)

    print("\nBASELINE TOP 10")

    for i in np.argsort(-frequency)[:10]:
        print(f"{COMBOS[i]}: {frequency[i]*100:.3f}%")

    print("\nStatus: baseline evaluation completed")


if __name__ == "__main__":
    main()
