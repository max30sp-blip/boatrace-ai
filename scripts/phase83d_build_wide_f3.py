
import sqlite3
from pathlib import Path

ROOT = Path(r"C:\boatrace_ai")

TRAIN_DB = ROOT / "data/train_dataset_v1.db"
F3_DB = ROOT / "outputs/phase83d_historical_f3.db"
OUT_DB = ROOT / "outputs/phase83d_wide_f3.db"

START = "2017-01-01"
END = "2024-12-31"

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


def readonly_uri(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve().as_uri() + "?mode=ro"


def main():
    if OUT_DB.exists():
        raise FileExistsError(
            f"Output already exists: {OUT_DB}\n"
            "既存の出力を保護するため停止しました。"
        )

    OUT_DB.parent.mkdir(
        parents=True, exist_ok=True
    )

    con = sqlite3.connect(OUT_DB)

    try:
        if not TRAIN_DB.is_file():
            raise FileNotFoundError(TRAIN_DB)

        if not F3_DB.is_file():
            raise FileNotFoundError(F3_DB)

        con.execute(
            "ATTACH DATABASE ? AS train_src",
            (str(TRAIN_DB),)
        )

        con.execute(
            "ATTACH DATABASE ? AS f3_src",
            (str(F3_DB),)
        )

        cols = []

        for lane in range(1, 7):
            for feature in FEATURES:
                cols.append(f"b{lane}_new_{feature}")

        column_sql = ",\n".join(
            f'"{c}" REAL' for c in cols
        )

        con.execute(f"""
            CREATE TABLE wide_f3 (
                race_date TEXT NOT NULL,
                venue_code INTEGER NOT NULL,
                race_no INTEGER NOT NULL,
                f3_matched INTEGER NOT NULL,
                {column_sql},
                PRIMARY KEY (
                    race_date, venue_code, race_no
                )
            ) WITHOUT ROWID
        """)

        con.execute("""
            CREATE TABLE metadata (
                name TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        con.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [
                ("start_date", START),
                ("end_date", END),
                ("feature_count", str(len(cols))),
                (
                    "reset_policy",
                    "unverified_rolling_proxy"
                ),
                (
                    "missing_policy",
                    "NULL_for_unmatched_races"
                ),
                (
                    "leakage_policy",
                    "same_day_results_excluded"
                ),
                (
                    "production_ready",
                    "false"
                ),
            ]
        )

        joins = []

        for lane in range(1, 7):
            joins.append(f"""
                LEFT JOIN f3_src.historical_f3 f{lane}
                  ON f{lane}.race_date = t.race_date
                 AND f{lane}.venue_code = t.venue_code
                 AND f{lane}.race_no = t.race_no
                 AND f{lane}.lane = {lane}
            """)

        select_cols = []

        for lane in range(1, 7):
            for feature in FEATURES:
                select_cols.append(
                    f'f{lane}."{feature}"'
                )

        select_sql = ",\n".join(select_cols)

        # 全6艇が揃ったレースのみmatched=1
        match_condition = " AND ".join(
            f"f{lane}.lane IS NOT NULL"
            for lane in range(1, 7)
        )

        # 既存のレース集合は変更しない
        sql = f"""
            INSERT INTO wide_f3
            SELECT
                t.race_date,
                t.venue_code,
                t.race_no,
                CASE WHEN {match_condition}
                     THEN 1 ELSE 0 END,
                {select_sql}
            FROM train_src.train_races t
            {' '.join(joins)}
            WHERE t.race_date BETWEEN ? AND ?
        """

        print("=== PHASE 8.3-D WIDE F3 BUILD ===")
        print("Converting boat features to race features")
        print("New features:", len(cols), flush=True)

        con.execute(sql, (START, END))
        con.commit()

        total, matched = con.execute("""
            SELECT
                COUNT(*),
                COALESCE(SUM(f3_matched), 0)
            FROM wide_f3
        """).fetchone()

        partial = con.execute("""
            SELECT COUNT(*)
            FROM wide_f3
            WHERE f3_matched = 0
        """).fetchone()[0]

        duplicate = con.execute("""
            SELECT COUNT(*) FROM (
                SELECT race_date, venue_code, race_no
                FROM wide_f3
                GROUP BY race_date, venue_code, race_no
                HAVING COUNT(*) > 1
            )
        """).fetchone()[0]

        # 未対応行がすべてNULLかを検証
        nonnull_condition = " OR ".join(
            f'"{c}" IS NOT NULL' for c in cols
        )

        incomplete_has_values = con.execute(f"""
            SELECT COUNT(*)
            FROM wide_f3
            WHERE f3_matched = 0
              AND ({nonnull_condition})
        """).fetchone()[0]

        print("\n=== VALIDATION ===")
        print("Total races:", total)
        print("Matched races:", matched)
        print("Unmatched races:", partial)
        print("Duplicate races:", duplicate)
        print(
            "Unmatched rows with partial features:",
            incomplete_has_values
        )

        # 予期した行数・照合件数と違えば停止
        if total != 438488:
            raise ValueError(
                f"Unexpected race count: {total}"
            )

        if matched != 438088:
            raise ValueError(
                f"Unexpected matched count: {matched}"
            )

        if duplicate:
            raise ValueError("Duplicate races")

        if incomplete_has_values:
            raise ValueError(
                "Partially matched races detected"
            )

        print("\n=== PHASE 8.3-D WIDE F3 COMPLETE ===")
        print("Rows:", total)
        print("Added features:", len(cols))
        print("Output:", OUT_DB)

    finally:
        con.close()


if __name__ == "__main__":
    main()
