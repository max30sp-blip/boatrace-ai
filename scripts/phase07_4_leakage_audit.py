
import sqlite3
import math
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "train_dataset_v1.db"

FORBIDDEN = {
    "finish_position",
    "result_code",
    "finish_code",
    "payout_yen",
    "combination",
    "refund_yen",
    "target",
    "label",
    "source_file",
}

# 同一選手・同一日で不変のはずの履歴特徴量
PREFIXES = (
    "f0_",  # 過去30/90/180日
    "f5_",  # 直近5/10/20走
    "f6_",  # 今節成績
)

def connect_ro(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True
    )


def equal_value(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        if math.isnan(float(a)) or math.isnan(float(b)):
            return math.isnan(float(a)) and math.isnan(float(b))
        return math.isclose(
            float(a), float(b),
            rel_tol=1e-10,
            abs_tol=1e-10
        )
    return a == b


def main():
    print("=" * 65)
    print("PHASE 7.4 PRE-RACE FEATURE LEAKAGE AUDIT")
    print("READ ONLY")
    print("=" * 65, flush=True)

    con = connect_ro(DB)

    try:
        cols = [
            x[1] for x in con.execute(
                "PRAGMA table_info(train_races)"
            )
        ]

        feature_cols = cols[4:]

        print("\n=== FORBIDDEN COLUMNS ===")

        forbidden = []
        for name in feature_cols:
            # b1_f5_recent_5_avg_finish 等の
            # 過去着順集計は許可する
            suffix = name.split("_", 2)[-1].lower()
            if suffix in FORBIDDEN:
                forbidden.append(name)

        print("Forbidden columns:", forbidden)

        print("\n=== DATASET INTEGRITY ===")

        total, distinct_keys = con.execute("""
            SELECT
                COUNT(*),
                COUNT(DISTINCT
                    race_date || ':' ||
                    venue_code || ':' || race_no)
            FROM train_races
        """).fetchone()

        print("Rows:", f"{total:,}")
        print("Distinct race keys:", f"{distinct_keys:,}")
        print("Duplicate keys:", total - distinct_keys)

        print("\n=== SAME-DAY HISTORY CONSISTENCY ===")

        # 学習データに選手IDが含まれていないため、
        # この検査は保存済み618特徴量だけでは
        # 正確に実施できない。
        # 実施するには元のprogram_entriesと照合する。
        source_path = ROOT / "data" / "boatrace.db"

        con.execute(
            "ATTACH DATABASE ? AS base",
            (source_path.resolve().as_uri() + "?mode=ro",)
        )
        con.execute("PRAGMA query_only=ON")

        # 使用する履歴特徴量のみ選択
        histories = [
            c for c in feature_cols
            if c.startswith("b1_")
            and c[3:].startswith(PREFIXES)
        ]

        print("Audited history fields:", len(histories))

        # 1艇につき同名列が存在することを確認
        missing = []
        for name in histories:
            suffix = name[3:]
            for lane in range(2, 7):
                if f"b{lane}_{suffix}" not in cols:
                    missing.append(f"b{lane}_{suffix}")

        print("Missing counterpart columns:", len(missing))

        # 日付単位でストリーム処理
        select_cols = [
            "t.race_date",
            "t.venue_code",
            "t.race_no",
        ]

        for lane in range(1, 7):
            for name in histories:
                select_cols.append(
                    f't."b{lane}_{name[3:]}"'
                )

        query = f"""
            SELECT {", ".join(select_cols)}
            FROM train_races t
            ORDER BY
                t.race_date, t.venue_code, t.race_no
        """

        current_day = None
        day_entries = {}
        repeats = 0
        differences = Counter()
        examples = []

        cursor = con.execute(query)

        for row in cursor:
            race_date, venue, race_no = row[:3]

            if race_date != current_day:
                day_entries.clear()
                current_day = race_date

            values = row[3:]
            field_count = len(histories)

            # 6艇の選手IDを元の番組表から取得
            racers = list(con.execute("""
                SELECT lane, racer_id
                FROM base.program_entries
                WHERE race_date = ?
                  AND venue_code = ?
                  AND race_no = ?
                ORDER BY lane
            """, (race_date, venue, race_no)))

            if len(racers) != 6:
                continue

            for lane, racer_id in racers:
                start = (lane - 1) * field_count
                end = start + field_count
                features = values[start:end]

                key = (venue, racer_id)

                if key not in day_entries:
                    day_entries[key] = features
                    continue

                repeats += 1
                previous = day_entries[key]

                for name, old, new in zip(
                    histories, previous, features
                ):
                    if not equal_value(old, new):
                        differences[name[3:]] += 1

                        if len(examples) < 15:
                            examples.append((
                                race_date, venue, racer_id,
                                name[3:], old, new
                            ))

            if race_no == 12 and venue == 24:
                pass

        print("Repeated same-day racer entries:", repeats)
        print("Fields with differences:", len(differences))

        print("\n=== DIFFERENCE COUNTS ===")

        for name, count in differences.most_common(30):
            print(name, count)

        print("\n=== EXAMPLES ===")
        for item in examples:
            print(item)

        print("\n=== FINAL ===")

        if forbidden or missing or total != distinct_keys:
            print("STRUCTURAL LEAKAGE AUDIT FAILED")
        elif differences:
            print("SAME-DAY FEATURE CHANGES REQUIRE REVIEW")
        else:
            print("SAMPLED RULE CHECKS PASSED")
            print("Source-code temporal audit still required")

        print("\nPHASE 7.4 AUDIT COMPLETE")

    finally:
        con.close()


if __name__ == "__main__":
    main()
