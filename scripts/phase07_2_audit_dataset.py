
import sqlite3
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

TRAIN_DB = DATA / "train_dataset_v1.db"
SOURCE_DB = DATA / "boatrace.db"

EXPECTED = 594_149

SUSPECT_WORDS = (
    "finish", "result", "payout", "refund",
    "trifecta", "hit", "target", "label",
    "future", "next_race", "after_race",
)

def connect_ro(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True
    )


def main():
    print("=" * 65)
    print("PHASE 7.2 DATASET QUALITY AUDIT")
    print("READ ONLY")
    print("=" * 65, flush=True)

    con = connect_ro(TRAIN_DB)

    try:
        cols = list(con.execute(
            "PRAGMA table_info(train_races)"
        ))

        if not cols:
            raise RuntimeError("train_races not found")

        names = [r[1] for r in cols]
        features = [
            c for c in names
            if c not in (
                "race_date", "venue_code",
                "race_no", "label"
            )
        ]

        total = con.execute(
            "SELECT COUNT(*) FROM train_races"
        ).fetchone()[0]

        print("\n=== BASIC ===")
        print("Rows:", f"{total:,}")
        print("Features:", len(features))
        print("Total columns:", len(names))
        print("Expected rows:", EXPECTED)
        print("Row count OK:", total == EXPECTED)

        if total != EXPECTED:
            raise RuntimeError("Unexpected row count")

        print("\n=== YEAR DISTRIBUTION ===")

        yearly = list(con.execute("""
            SELECT
                SUBSTR(race_date, 1, 4) AS year,
                COUNT(*),
                COUNT(DISTINCT label)
            FROM train_races
            GROUP BY SUBSTR(race_date, 1, 4)
            ORDER BY year
        """))

        for year, n, label_count in yearly:
            print(
                f"{year}: races={n:,} "
                f"labels={label_count}"
            )

        print("\n=== CHRONOLOGICAL SPLIT ===")

        splits = [
            ("WARMUP", "2016-01-01", "2016-12-31"),
            ("TRAIN", "2017-01-01", "2023-12-31"),
            ("VALID", "2024-01-01", "2024-12-31"),
            ("TEST", "2025-01-01", "2026-10-29"),
        ]

        for label, start, end in splits:
            n, kinds = con.execute("""
                SELECT COUNT(*), COUNT(DISTINCT label)
                FROM train_races
                WHERE race_date BETWEEN ? AND ?
            """, (start, end)).fetchone()

            print(
                f"{label:7s} "
                f"races={n:,} "
                f"labels={kinds}"
            )

        print("\n=== LABEL VALIDITY ===")

        all_labels = [
            f"{a}-{b}-{c}"
            for a in range(1, 7)
            for b in range(1, 7)
            for c in range(1, 7)
            if len({a, b, c}) == 3
        ]

        observed = dict(con.execute("""
            SELECT label, COUNT(*)
            FROM train_races
            GROUP BY label
        """))

        invalid = {
            key: value
            for key, value in observed.items()
            if key not in all_labels
        }

        missing = [
            x for x in all_labels
            if x not in observed
        ]

        print("Expected classes:", len(all_labels))
        print("Observed classes:", len(observed))
        print("Invalid labels:", invalid)
        print("Missing classes:", missing)

        print("\n=== FEATURE NAME AUDIT ===")

        suspicious = [
            c for c in features
            if any(w in c.lower() for w in SUSPECT_WORDS)
        ]

        print("Potential leakage names:")
        if suspicious:
            for name in suspicious:
                print("  REVIEW:", name)
        else:
            print("  None by name")

        # 数値に変換できない値は、サンプルで検出する
        print("\n=== SAMPLE MISSING COVERAGE ===")
        print("Sample: up to 300 races per year")
        print("This is NOT full-dataset coverage.")

        nulls = Counter()
        non_nulls = Counter()
        sample_count = 0

        col_index = {
            name: i for i, name in enumerate(names)
        }

        for year, _, _ in yearly:
            rows = con.execute("""
                SELECT *
                FROM train_races
                WHERE race_date >= ?
                  AND race_date < ?
                ORDER BY race_date, venue_code, race_no
                LIMIT 300
            """, (f"{year}-01-01",
                  f"{int(year)+1}-01-01"))

            for row in rows:
                sample_count += 1

                for name in features:
                    value = row[col_index[name]]

                    if value is None:
                        nulls[name] += 1
                    else:
                        non_nulls[name] += 1

        print("Sampled races:", sample_count)

        ranking = sorted(
            features,
            key=lambda c: (
                nulls[c] / max(sample_count, 1)
            ),
            reverse=True
        )

        print("\nTop 25 NULL columns:")

        for name in ranking[:25]:
            n = nulls[name]
            pct = 100 * n / max(sample_count, 1)

            print(
                f"{pct:6.2f}% "
                f"NULL={n:5d} "
                f"{name}"
            )

        empty = [
            c for c in features
            if non_nulls[c] == 0
        ]

        print("\nAll-NULL in sample:", len(empty))
        for name in empty[:30]:
            print("  EMPTY:", name)

        print("\n=== GRADE ENCODING ===")

        for lane in range(1, 7):
            col = f"b{lane}_p_grade_rank"

            if col not in names:
                print(col, "NOT FOUND")
                continue

            distribution = list(con.execute(f"""
                SELECT "{col}", COUNT(*)
                FROM train_races
                GROUP BY "{col}"
                ORDER BY "{col}"
            """))

            print(f"Boat {lane}:", distribution)

        print("\n=== SOURCE LABEL SCHEMA ===")

        base = connect_ro(SOURCE_DB)

        try:
            tables = [
                "trifecta_results",
                "trifecta_payouts_supplement",
                "race_boat_results_v2",
                "race_boat_results_supplement",
            ]

            for table in tables:
                info = list(base.execute(
                    f'PRAGMA table_info("{table}")'
                ))

                print(f"\n{table}:")

                if not info:
                    print("  TABLE MISSING")
                    continue

                print("  Columns:")
                print(" ", [r[1] for r in info])

                print("  Example rows:")

                for row in base.execute(
                    f'SELECT * FROM "{table}" LIMIT 2'
                ):
                    print(" ", row)

        finally:
            base.close()

        print("\n=== FINAL ===")
        if invalid:
            print("LABEL AUDIT FAILED")
        else:
            print("DATASET STRUCTURE CHECK COMPLETE")

        print("PHASE 7.2 INITIAL AUDIT COMPLETE")

    finally:
        con.close()


if __name__ == "__main__":
    main()
