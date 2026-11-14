
import argparse
import sqlite3
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = DATA / "train_dataset_v1.db"

TABLE = "train_races"
BUILD = "train_races_build"

KEYS = ("race_date", "venue_code", "race_no", "lane")
EXCLUDE = set(KEYS) | {
    "racer_id", "source_file", "racer_name",
    "motor_no", "boat_no"
}

# schema, DB file, table, SQL alias
SOURCES = [
    ("base", "boatrace.db", "program_entries", "p"),
    ("features", "features_fast.db", "racer_features_fast", "f0"),
    ("features", "features_fast.db", "racer_course_features_fast", "f1"),
    ("lane_db", "lane_features.db", "racer_lane_features_fast", "f2"),
    ("motor_db", "motor_features.db", "equipment_features_fast", "f3"),
    ("venue_db", "venue_detail_features.db", "venue_detail_features_fast", "f4"),
    ("recent_db", "recent_form_features.db", "racer_recent_form_fast", "f5"),
    ("meet_db", "meet_features.db", "racer_meet_features", "f6"),
]

GRADE = {
    "B2": 0,
    "B1": 1,
    "A2": 2,
    "A1": 3,
}


def q(value):
    return '"' + value.replace('"', '""') + '"'


def db_uri(filename):
    path = DATA / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve().as_uri() + "?mode=ro"


def get_columns(con, schema, table):
    return list(con.execute(
        f"PRAGMA {q(schema)}.table_info({q(table)})"
    ))


def numeric_columns(rows):
    selected = []
    for row in rows:
        name = row[1]
        declared = (row[2] or "").upper()

        if name in EXCLUDE:
            continue

        # 数値型として定義された列のみ使用
        if any(x in declared for x in (
            "INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC"
        )):
            selected.append(name)

    return selected


def label_from_results(rows):
    """1着・2着・3着の艇番が一意に決まる場合だけ採用。"""
    if len(rows) != 6:
        return None

    places = {1: [], 2: [], 3: []}

    for row in rows:
        lane = row[3]
        finish = row[-1]

        try:
            finish = int(finish)
        except (ValueError, TypeError):
            continue

        if finish in places:
            places[finish].append(lane)

    if not all(len(places[x]) == 1 for x in (1, 2, 3)):
        return None

    boats = tuple(places[x][0] for x in (1, 2, 3))

    if len(set(boats)) != 3:
        return None

    return "-".join(map(str, boats))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--limit-races", type=int, default=0)
    args = parser.parse_args()

    if args.limit_races < 0:
        raise ValueError("--limit-races must be >= 0")

    # 読み込み用の一時的な接続
    src = sqlite3.connect(":memory:", uri=True, timeout=120)

    # 新しい出力ファイル
    dst = None

    try:
        attached = set()

        for schema, filename, table, alias in SOURCES:
            if schema not in attached:
                src.execute(
                    f"ATTACH DATABASE ? AS {q(schema)}",
                    (db_uri(filename),)
                )
                attached.add(schema)

        src.execute("PRAGMA query_only=ON")

        selections = []
        feature_names = []

        for schema, filename, table, alias in SOURCES:
            cols = get_columns(src, schema, table)

            if not cols:
                raise RuntimeError(f"Missing table: {table}")

            names = numeric_columns(cols)

            for name in names:
                output_name = f"{alias}_{name}"
                feature_names.append(output_name)
                selections.append(
                    f"{alias}.{q(name)} AS {q(output_name)}"
                )

            if alias == "p":
                actual = {x[1] for x in cols}
                if "grade" in actual:
                    feature_names.append("p_grade_rank")
                    selections.append(
                        "p.grade AS p_grade_rank"
                    )

        if not feature_names:
            raise RuntimeError("No feature columns found")

        # 計算に必要なキーを先頭に配置
        select_parts = [
            "p.race_date",
            "p.venue_code",
            "p.race_no",
            "p.lane",
            "p.racer_id",
            *selections,
            "COALESCE(r.finish_position, s.finish_position)"
            " AS finish_position"
        ]

        joins = []

        for schema, filename, table, alias in SOURCES[1:]:
            lane_col = "boat_no" if alias == "f0" else "lane"

            joins.append(f"""
                INNER JOIN {q(schema)}.{q(table)} {alias}
                    ON {alias}.race_date = p.race_date
                   AND {alias}.venue_code = p.venue_code
                   AND {alias}.race_no = p.race_no
                   AND {alias}.{lane_col} = p.lane
                   AND {alias}.racer_id = p.racer_id
            """)

        joins.append("""
            LEFT JOIN base.race_boat_results_v2 r
              ON r.race_date = p.race_date
             AND r.venue_code = p.venue_code
             AND r.race_no = p.race_no
             AND r.lane = p.lane
             AND r.racer_id = p.racer_id
        """)

        joins.append("""
            LEFT JOIN base.race_boat_results_supplement s
              ON s.race_date = p.race_date
             AND s.venue_code = p.venue_code
             AND s.race_no = p.race_no
             AND s.boat_no = p.lane
             AND s.racer_id = p.racer_id
        """)

        conditions = ["""
            EXISTS (
                SELECT 1
                FROM base.trifecta_results t
                WHERE t.race_date = p.race_date
                  AND t.venue_code = p.venue_code
                  AND t.race_no = p.race_no
            )
        """]

        params = []

        if args.start:
            conditions.append("p.race_date >= ?")
            params.append(args.start)

        if args.end:
            conditions.append("p.race_date <= ?")
            params.append(args.end)

        sql = f"""
            SELECT {", ".join(select_parts)}
            FROM base.program_entries p
            {" ".join(joins)}
            WHERE {" AND ".join(conditions)}
            ORDER BY
                p.race_date,
                p.venue_code,
                p.race_no,
                p.lane
        """

        # 6艇 × 各艇特徴量
        wide_columns = []

        for lane in range(1, 7):
            for name in feature_names:
                wide_columns.append(
                    f"b{lane}_{name}"
                )

        dst = sqlite3.connect(OUTPUT, timeout=120)
        dst.execute(f"DROP TABLE IF EXISTS {BUILD}")

        column_defs = [
            "race_date TEXT NOT NULL",
            "venue_code INTEGER NOT NULL",
            "race_no INTEGER NOT NULL",
            "label TEXT NOT NULL",
        ]

        column_defs += [
            f"{q(name)} REAL"
            for name in wide_columns
        ]

        column_defs.append(
            "PRIMARY KEY(race_date, venue_code, race_no)"
        )

        dst.execute(
            f"CREATE TABLE {BUILD} ({', '.join(column_defs)})"
        )
        dst.commit()

        insert_count = 4 + len(wide_columns)
        placeholders = ",".join("?" for _ in range(insert_count))

        insert_sql = (
            f"INSERT INTO {BUILD} VALUES ({placeholders})"
        )

        print("========== PHASE 7.1 DATASET ==========")
        print("Source mode: READ ONLY")
        print("Output:", OUTPUT)
        print("Features per boat:", len(feature_names))
        print("Features per race:", len(wide_columns))
        print("Start:", args.start or "ALL")
        print("End:", args.end or "ALL")
        print("Limit races:", args.limit_races or "ALL", flush=True)

        cursor = src.execute(sql, params)

        current_key = None
        race_rows = []
        batch = []
        saved = 0
        scanned = 0
        skipped = Counter()
        labels = Counter()

        def save_race(rows):
            nonlocal saved, scanned, batch

            if not rows:
                return

            scanned += 1
            race_key = tuple(rows[0][:3])

            if len(rows) != 6:
                skipped["not_six_entries"] += 1
                return

            lanes = [row[3] for row in rows]

            if lanes != [1, 2, 3, 4, 5, 6]:
                skipped["invalid_lanes"] += 1
                return

            label = label_from_results(rows)

            if label is None:
                skipped["ambiguous_finish"] += 1
                return

            flat = []

            for row in rows:
                values = list(row[5:-1])

                for i, name in enumerate(feature_names):
                    if name == "p_grade_rank":
                        values[i] = GRADE.get(values[i])

                    elif values[i] is not None:
                        try:
                            values[i] = float(values[i])
                        except (ValueError, TypeError):
                            values[i] = None

                flat.extend(values)

            batch.append((
                *race_key,
                label,
                *flat
            ))

            labels[label] += 1
            saved += 1

            if len(batch) >= 500:
                dst.executemany(insert_sql, batch)
                dst.commit()
                batch.clear()

            if saved % 10000 == 0:
                print(
                    f"Saved races: {saved:,}"
                    f" | Scanned: {scanned:,}",
                    flush=True
                )

        for row in cursor:
            key = tuple(row[:3])

            if current_key is None:
                current_key = key

            if key != current_key:
                save_race(race_rows)
                race_rows = []
                current_key = key

                if (
                    args.limit_races
                    and scanned >= args.limit_races
                ):
                    break

            race_rows.append(row)

        else:
            if race_rows and (
                not args.limit_races
                or scanned < args.limit_races
            ):
                save_race(race_rows)

        if batch:
            dst.executemany(insert_sql, batch)
            dst.commit()

        stored = dst.execute(
            f"SELECT COUNT(*) FROM {BUILD}"
        ).fetchone()[0]

        if stored != saved:
            raise RuntimeError(
                f"Saved row mismatch: {saved} != {stored}"
            )

        # 本番テーブルへの切替
        with dst:
            dst.execute(f"DROP TABLE IF EXISTS {TABLE}")
            dst.execute(
                f"ALTER TABLE {BUILD} RENAME TO {TABLE}"
            )

        print("\n=== SUMMARY ===")
        print("Scanned races:", f"{scanned:,}")
        print("Stored races:", f"{stored:,}")
        print("Skipped races:", f"{sum(skipped.values()):,}")
        print("Features per race:", len(wide_columns))
        print("Unique labels:", len(labels))

        print("\n=== SKIPPED REASONS ===")
        for reason, n in skipped.most_common():
            print(reason, n)

        print("\n=== TOP 10 LABELS ===")
        for label, n in labels.most_common(10):
            print(label, n)

        print("\n=== SAMPLE ===")
        for row in dst.execute(f"""
            SELECT race_date, venue_code, race_no, label
            FROM {TABLE}
            ORDER BY race_date DESC, venue_code, race_no
            LIMIT 10
        """):
            print(row)

        print("\nPHASE 7.1 DATASET COMPLETE")

    finally:
        src.close()
        if dst is not None:
            dst.close()


if __name__ == "__main__":
    main()
