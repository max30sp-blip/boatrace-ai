
import sqlite3
import math
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

EXPECTED_ROWS = 3_619_728

KEYS = ("race_date", "venue_code", "race_no", "lane")
REQUIRED = KEYS + ("racer_id",)

# PHASE 6のみboat_noが出走艇番を表す
def key_columns(alias):
    if alias == "basic":
        return ("race_date", "venue_code", "race_no", "boat_no")
    return KEYS

# DBファイルとテーブル名を明示的に指定
SOURCES = [
    ("base", "boatrace.db", "program_entries"),
    ("basic", "features_fast.db", "racer_features_fast"),
    ("course", "features_fast.db", "racer_course_features_fast"),
    ("lane", "lane_features.db", "racer_lane_features_fast"),
    ("motor", "motor_features.db", "equipment_features_fast"),
    ("venue", "venue_detail_features.db", "venue_detail_features_fast"),
    ("recent", "recent_form_features.db", "racer_recent_form_fast"),
    ("meet", "meet_features.db", "racer_meet_features"),
]


def readonly_uri(path):
    return path.resolve().as_uri() + "?mode=ro"


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def columns(con, schema, table):
    return {
        row[1]: row
        for row in con.execute(
            f"PRAGMA {quote(schema)}.table_info({quote(table)})"
        )
    }


def has_unique_key(con, schema, table, cols):
    """
    結合キーの一意性を制約から確認。
    主キーまたはUNIQUE INDEXを検査する。
    """
    info = list(con.execute(
        f"PRAGMA {quote(schema)}.table_info({quote(table)})"
    ))

    pk = [
        row[1]
        for row in sorted(info, key=lambda r: r[5])
        if row[5] > 0
    ]

    if pk == list(cols):
        return True

    for idx in con.execute(
        f"PRAGMA {quote(schema)}.index_list({quote(table)})"
    ):
        index_name = idx[1]
        is_unique = idx[2]

        if not is_unique:
            continue

        index_cols = [
            x[2]
            for x in con.execute(
                f"PRAGMA {quote(schema)}.index_info({quote(index_name)})"
            )
        ]

        if index_cols == list(cols):
            return True

    return False


def check_ranges(con, schema, table, cols):
    """
    勝率・3着以内率・進入変更率等の
    0～1の範囲を監査。
    """
    rate_cols = [
        c for c in cols
        if (
            "rate" in c.lower()
            or "probability" in c.lower()
        )
        and not c.lower().endswith("2rate")
    ]

    problems = []

    for name in rate_cols:
        # 0～1以外、およびNaNを検出
        sql = f"""
            SELECT COUNT(*)
            FROM {quote(schema)}.{quote(table)}
            WHERE {quote(name)} IS NOT NULL
              AND (
                  {quote(name)} < 0
                  OR {quote(name)} > 1
                  OR {quote(name)} != {quote(name)}
              )
        """

        count = con.execute(sql).fetchone()[0]

        if count:
            problems.append((name, count))

    return rate_cols, problems


def audit_join(con, schema, table, alias):
    """
    出走キーと選手IDを全件照合。
    PHASE 6のboat_noは出走艇番として照合する。
    """
    feature_keys = key_columns(alias)

    conditions = " AND ".join(
        f'p."{base_key}" = f."{feature_key}"'
        for base_key, feature_key in zip(KEYS, feature_keys)
    )

    sql = f"""
        SELECT
            COUNT(*) AS total,
            COALESCE(SUM(
                CASE WHEN f.race_date IS NULL
                THEN 1 ELSE 0 END
            ), 0) AS missing,
            COALESCE(SUM(
                CASE
                    WHEN f.race_date IS NOT NULL
                    AND (
                        f.racer_id IS NULL
                        OR p.racer_id IS NULL
                        OR f.racer_id != p.racer_id
                    )
                    THEN 1 ELSE 0
                END
            ), 0) AS mismatched
        FROM base.program_entries p
        LEFT JOIN {quote(schema)}.{quote(table)} f
          ON {conditions}
    """

    return con.execute(sql).fetchone()


def main():
    print("=" * 64)
    print("PHASE 7.0 FEATURE INTEGRITY AUDIT")
    print("SOURCE MODE: READ ONLY")
    print("=" * 64, flush=True)

    base_path = DATA / "boatrace.db"

    if not base_path.exists():
        raise FileNotFoundError(base_path)

    # メモリ上の接続に読み取り専用DBをATTACH
    # 元ファイルへの書き込みは行わない
    con = sqlite3.connect(
    ":memory:",
    timeout=120,
    uri=True
)

    errors = []
    warnings = []
    summary = []

    try:

        attached = set()

        for alias, filename, table in SOURCES:
            path = DATA / filename

            if not path.exists():
                raise FileNotFoundError(path)

            if alias == "base":
                con.execute(
                    "ATTACH DATABASE ? AS base",
                    (readonly_uri(path),)
                )
                attached.add(alias)

            elif filename == "features_fast.db":
                if "features" not in attached:
                    con.execute(
                        "ATTACH DATABASE ? AS features",
                        (readonly_uri(path),)
                    )
                    attached.add("features")

            else:
                con.execute(
                    f"ATTACH DATABASE ? AS {quote(alias)}",
                    (readonly_uri(path),)
                )
                attached.add(alias)

        # 全DBの接続完了後に読み取り専用を設定
        con.execute("PRAGMA query_only=ON")

        print("\n=== TABLE AUDIT ===", flush=True)

        table_infos = []

        for alias, filename, table in SOURCES:
            schema = (
                "features"
                if filename == "features_fast.db"
                else alias
            )

            cols = columns(con, schema, table)

            if not cols:
                raise RuntimeError(
                    f"Missing table: {filename} / {table}"
                )

            required = key_columns(alias) + ("racer_id",)
            missing_cols = [
                c for c in required if c not in cols
            ]

            if missing_cols:
                raise RuntimeError(
                    f"{table}: missing columns {missing_cols}"
                )

            target = f"{quote(schema)}.{quote(table)}"

            count = con.execute(
                f"SELECT COUNT(*) FROM {target}"
            ).fetchone()[0]

            unique = has_unique_key(
                    con, schema, table, key_columns(alias)
                    )

            # PHASE 6はboat_no、それ以外はlane
            lane_col = "boat_no" if alias == "basic" else "lane"

            invalid_keys = con.execute(f"""
                SELECT COUNT(*)
                FROM {target}
                WHERE race_date IS NULL
                OR venue_code IS NULL
                OR race_no IS NULL
                OR {quote(lane_col)} IS NULL
                OR racer_id IS NULL
                OR {quote(lane_col)} NOT BETWEEN 1 AND 6
                OR race_no NOT BETWEEN 1 AND 12
                OR venue_code NOT BETWEEN 1 AND 24
            """).fetchone()[0]

            if count != EXPECTED_ROWS:
                errors.append(
                    f"{alias}: row count {count}"
                )

            if invalid_keys:
                errors.append(
                    f"{alias}: invalid keys {invalid_keys}"
                )

            if not unique:
                warnings.append(
                    f"{alias}: no matching UNIQUE key constraint"
                )

            print(
                f"{alias:8s} rows={count:,}"
                f" unique_key={unique}"
                f" invalid_keys={invalid_keys}",
                flush=True
            )

            table_infos.append((
                alias, schema, table, cols, count, unique
            ))

        print("\n=== JOIN AUDIT ===", flush=True)

        for alias, schema, table, cols, count, unique in table_infos:
            if alias == "base":
                continue

            # インデックスが無い場合、大規模JOINは
            # 著しく遅くなるため実行しない
            if not unique:
                warnings.append(
                    f"{alias}: join not audited; index review needed"
                )
                print(
                    f"{alias:8s} JOIN SKIPPED (NO UNIQUE KEY)",
                    flush=True
                )
                continue

            total, missing, mismatched = audit_join(
                con, schema, table, alias
            )

            print(
                f"{alias:8s}"
                f" checked={total:,}"
                f" missing={missing:,}"
                f" racer_mismatch={mismatched:,}",
                flush=True
            )

            if missing or mismatched:
                errors.append(
                    f"{alias}: missing={missing}, "
                    f"mismatch={mismatched}"
                )

            summary.append((alias, missing, mismatched))

        print("\n=== RATE RANGE AUDIT ===", flush=True)

        for alias, schema, table, cols, count, unique in table_infos:
            if alias == "base":
                continue

            rate_cols, problems = check_ranges(
                con, schema, table, cols
            )

            print(
                f"{alias:8s}"
                f" rate_columns={len(rate_cols)}"
                f" invalid={sum(x[1] for x in problems)}",
                flush=True
            )

            for name, n in problems:
                errors.append(
                    f"{alias}.{name}: {n} invalid values"
                )

        print("\n=== MEET SPECIFIC AUDIT ===")

        meet = "meet.racer_meet_features"

        checks = {
            "first_day_st": f"""
                SELECT COUNT(*) FROM {meet}
                WHERE meet_day = 1
                  AND (
                    meet_st_count != 0
                    OR meet_avg_st IS NOT NULL
                  )
            """,
            "unsafe_st_overflow": f"""
                SELECT COUNT(*) FROM {meet}
                WHERE meet_st_count > meet_records
                  AND (
                    meet_day < 8
                    OR meet_st_partial != 1
                    OR meet_avg_st IS NOT NULL
                  )
            """,
            "partial_st_with_average": f"""
                SELECT COUNT(*) FROM {meet}
                WHERE meet_st_partial = 1
                  AND meet_avg_st IS NOT NULL
            """,
            "unknown_form_chars": f"""
                SELECT COUNT(*) FROM {meet}
                WHERE meet_unknown_chars > 0
            """,
        }

        for name, sql in checks.items():
            result = con.execute(sql).fetchone()[0]
            print(f"{name}: {result}", flush=True)

            if result:
                errors.append(f"{name}: {result}")

        print("\n=== MISSING VALUE COVERAGE ===")

        inspect = [
            ("recent", "racer_recent_form_fast",
             "recent_10_avg_st"),
            ("meet", "racer_meet_features",
             "meet_avg_st"),
            ("meet", "racer_meet_features",
             "meet_st_partial"),
        ]

        for schema, table, field in inspect:
            cols = columns(con, schema, table)

            if field not in cols:
                warnings.append(
                    f"Missing coverage column: {table}.{field}"
                )
                continue

            nulls = con.execute(f"""
                SELECT COUNT(*)
                FROM {quote(schema)}.{quote(table)}
                WHERE {quote(field)} IS NULL
            """).fetchone()[0]

            print(
                f"{table}.{field}: NULL={nulls:,}",
                flush=True
            )

        print("\n=== FINAL RESULT ===")

        if errors:
            print("AUDIT FAILED")
            for e in errors:
                print("ERROR:", e)
        else:
            print("STRUCTURAL AUDIT PASSED")

        for w in warnings:
            print("WARNING:", w)

        print("\nPHASE 7.0 AUDIT COMPLETE")

    finally:
        con.close()


if __name__ == "__main__":
    main()
