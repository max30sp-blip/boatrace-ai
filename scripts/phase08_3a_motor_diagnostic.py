
import argparse
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

BASE_DB = DATA / "boatrace.db"
MOTOR_DB = DATA / "motor_features.db"


def uri(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve().as_uri() + "?mode=ro"


def connect():
    con = sqlite3.connect(":memory:", uri=True, timeout=120)
    con.execute(
        "ATTACH DATABASE ? AS base",
        (uri(BASE_DB),)
    )
    con.execute(
        "ATTACH DATABASE ? AS motor",
        (uri(MOTOR_DB),)
    )
    con.execute("PRAGMA query_only=ON")
    return con


def describe(values):
    if not values:
        return "NO DATA"

    values = sorted(values)
    n = len(values)

    def percentile(p):
        index = min(n - 1, int((n - 1) * p))
        return values[index]

    return (
        f"n={n:,} "
        f"mean={sum(values)/n:.4f} "
        f"p10={percentile(.10):.4f} "
        f"p50={percentile(.50):.4f} "
        f"p90={percentile(.90):.4f}"
    )


def correlation(pairs):
    n = len(pairs)
    if n < 3:
        return None

    sx = sy = sxx = syy = sxy = 0.0

    for x, y in pairs:
        sx += x
        sy += y
        sxx += x*x
        syy += y*y
        sxy += x*y

    vx = n*sxx - sx*sx
    vy = n*syy - sy*sy

    if vx <= 0 or vy <= 0:
        return None

    return (n*sxy - sx*sy) / math.sqrt(vx*vy)


def inspect_years(con):
    print("\n=== YEARLY COVERAGE ===", flush=True)

    sql = """
        SELECT
            SUBSTR(race_date,1,4),
            COUNT(*),
            AVG(CASE
                WHEN motor_starts_30d = 0
                THEN 1.0 ELSE 0.0 END),
            AVG(CASE
                WHEN motor_starts_90d = 0
                THEN 1.0 ELSE 0.0 END),
            AVG(CASE
                WHEN boat_starts_30d = 0
                THEN 1.0 ELSE 0.0 END),
            AVG(CASE
                WHEN boat_starts_90d = 0
                THEN 1.0 ELSE 0.0 END),
            AVG(motor_starts_90d),
            AVG(boat_starts_90d)
        FROM motor.equipment_features_fast
        GROUP BY SUBSTR(race_date,1,4)
        ORDER BY 1
    """

    for row in con.execute(sql):
        year, n, m30, m90, b30, b90, am, ab = row
        print(
            f"{year}: rows={n:,} "
            f"M30_zero={m30:.2%} "
            f"M90_zero={m90:.2%} "
            f"B30_zero={b30:.2%} "
            f"B90_zero={b90:.2%} "
            f"M90_avg={am:.1f} "
            f"B90_avg={ab:.1f}",
            flush=True
        )


def inspect_quality(con):
    print("\n=== FULL DATA QUALITY ===", flush=True)

    sql = """
        SELECT
            COUNT(*),
            SUM(CASE
                WHEN motor_starts_90d < motor_starts_30d
                THEN 1 ELSE 0 END),
            SUM(CASE
                WHEN boat_starts_90d < boat_starts_30d
                THEN 1 ELSE 0 END),
            SUM(CASE
                WHEN motor_starts_30d = 0
                 AND motor_win_rate_30d IS NOT NULL
                THEN 1 ELSE 0 END),
            SUM(CASE
                WHEN boat_starts_30d = 0
                 AND boat_win_rate_30d IS NOT NULL
                THEN 1 ELSE 0 END),
            SUM(CASE
                WHEN motor_starts_90d > 0
                 AND motor_win_rate_90d IS NULL
                THEN 1 ELSE 0 END),
            SUM(CASE
                WHEN boat_starts_90d > 0
                 AND boat_win_rate_90d IS NULL
                THEN 1 ELSE 0 END)
        FROM motor.equipment_features_fast
    """

    result = con.execute(sql).fetchone()

    names = [
        "total_rows",
        "motor_90_lt_30",
        "boat_90_lt_30",
        "motor_zero_with_rate",
        "boat_zero_with_rate",
        "motor_nonzero_without_rate",
        "boat_nonzero_without_rate",
    ]

    for name, value in zip(names, result):
        print(f"{name}: {value:,}", flush=True)


def inspect_finishes(con, start, end):
    print("\n=== SPECIAL FINISH RECORDS ===", flush=True)

    # F3の元コードでは着順不明も
    # 履歴分母に数えるため、その影響を確認
    for table, lane_col in [
        ("race_boat_results_v2", "lane"),
        ("race_boat_results_supplement", "boat_no"),
    ]:
        sql = f"""
            SELECT
                COUNT(*),
                SUM(CASE
                    WHEN finish_position BETWEEN 1 AND 6
                    THEN 1 ELSE 0 END),
                SUM(CASE
                    WHEN finish_position IS NULL
                      OR finish_position NOT BETWEEN 1 AND 6
                    THEN 1 ELSE 0 END)
            FROM base.{table}
            WHERE race_date BETWEEN ? AND ?
        """

        total, valid, other = con.execute(
            sql, (start, end)
        ).fetchone()

        print(
            f"{table}: "
            f"total={total:,} "
            f"valid={valid:,} "
            f"special_or_missing={other:,}",
            flush=True
        )


def inspect_sample(con, start, end, sample_limit):
    print("\n=== MOTOR / PROGRAM COMPARISON ===", flush=True)
    print("Period:", start, "to", end)
    print("Sample limit:", sample_limit)

    sql = """
        SELECT
            p.race_date,
            p.venue_code,
            p.lane,
            p.motor_no,
            p.boat_no,
            p.motor_2rate,
            p.boat_2rate,
            f.motor_starts_30d,
            f.motor_starts_90d,
            f.motor_win_rate_90d,
            f.motor_top3_rate_90d,
            f.boat_starts_30d,
            f.boat_starts_90d,
            f.boat_win_rate_90d,
            f.boat_top3_rate_90d
        FROM base.program_entries p
        JOIN motor.equipment_features_fast f
          ON p.race_date = f.race_date
         AND p.venue_code = f.venue_code
         AND p.race_no = f.race_no
         AND p.lane = f.lane
         AND p.racer_id = f.racer_id
        WHERE p.race_date BETWEEN ? AND ?
        ORDER BY p.race_date, p.venue_code,
                 p.race_no, p.lane
    """

    # 期間内のデータを分散して抽出
    count_sql = """
        SELECT COUNT(*)
        FROM base.program_entries
        WHERE race_date BETWEEN ? AND ?
    """

    total = con.execute(
        count_sql, (start, end)
    ).fetchone()[0]

    if total == 0:
        raise RuntimeError("No entries for period")

    stride = max(1, total // sample_limit)

    samples = 0
    buckets = defaultdict(int)
    values = defaultdict(list)
    pairs = defaultdict(list)
    equipment = {
        "motor": defaultdict(lambda: [None, None, 0]),
        "boat": defaultdict(lambda: [None, None, 0]),
    }

    for index, row in enumerate(
        con.execute(sql, (start, end))
    ):
        if index % stride:
            continue

        if samples >= sample_limit:
            break

        (
            rd, venue, lane, motor, boat,
            motor_2rate, boat_2rate,
            m30, m90, mw90, mt90,
            b30, b90, bw90, bt90
        ) = row

        samples += 1

        for prefix, count30, count90, win90, top90 in [
            ("motor", m30, m90, mw90, mt90),
            ("boat", b30, b90, bw90, bt90),
        ]:
            values[f"{prefix}_starts30"].append(count30)
            values[f"{prefix}_starts90"].append(count90)

            if count90 == 0:
                buckets[f"{prefix}_zero90"] += 1
            elif count90 < 5:
                buckets[f"{prefix}_under5"] += 1
            elif count90 < 15:
                buckets[f"{prefix}_under15"] += 1
            elif count90 < 30:
                buckets[f"{prefix}_under30"] += 1
            else:
                buckets[f"{prefix}_30plus"] += 1

            if (
                count90 > 0
                and win90 is not None
                and (win90 == 0 or win90 == 1)
            ):
                buckets[f"{prefix}_extreme_winrate"] += 1

            if top90 is not None:
                values[f"{prefix}_top3_90d"].append(top90)

        # 番組表の2連対率は0～100の数値を想定。
        # もし0～1で保存されていれば自動推測はせず、
        # 下の分布を確認してから解釈する。
        for name, v in [
            ("program_motor_2rate", motor_2rate),
            ("program_boat_2rate", boat_2rate),
        ]:
            if v is not None:
                values[name].append(float(v))

        if motor_2rate is not None and mt90 is not None:
            pairs["motor_program_vs_top3"].append(
                (float(motor_2rate), float(mt90))
            )

        if boat_2rate is not None and bt90 is not None:
            pairs["boat_program_vs_top3"].append(
                (float(boat_2rate), float(bt90))
            )

        # 同じ番号の使用日の広がりを確認する。
        # 交換日そのものはこれでは確定できない。
        for kind, number in [
            ("motor", motor),
            ("boat", boat),
        ]:
            if number is None:
                continue

            item = equipment[kind][(venue, number)]
            if item[0] is None or rd < item[0]:
                item[0] = rd
            if item[1] is None or rd > item[1]:
                item[1] = rd
            item[2] += 1

    print("Sampled entries:", f"{samples:,}")

    print("\n--- HISTORY VOLUME ---")
    for name in [
        "motor_starts30",
        "motor_starts90",
        "boat_starts30",
        "boat_starts90",
    ]:
        print(name, describe(values[name]))

    print("\n--- HISTORY BUCKETS ---")
    for kind in ("motor", "boat"):
        print(f"\n{kind.upper()}")
        for label in [
            "zero90", "under5", "under15",
            "under30", "30plus",
            "extreme_winrate"
        ]:
            n = buckets[f"{kind}_{label}"]
            print(
                f"{label:20s}: {n:7,d} "
                f"({n/max(samples,1):.2%})"
            )

    print("\n--- PROGRAM PUBLISHED RATES ---")
    for name in [
        "program_motor_2rate",
        "program_boat_2rate",
    ]:
        print(name, describe(values[name]))

    print("\n--- CORRELATIONS (REFERENCE ONLY) ---")
    for name in [
        "motor_program_vs_top3",
        "boat_program_vs_top3",
    ]:
        r = correlation(pairs[name])
        print(
            name,
            f"r={r:.4f}" if r is not None else "N/A",
            f"n={len(pairs[name]):,}"
        )

    print(
        "\nNOTE: program 2rate and historical "
        "top3 rate have different definitions."
    )
    print(
        "Correlation here is descriptive only, "
        "not a validity or predictive-power test."
    )

    print("\n--- REUSED EQUIPMENT NUMBERS ---")
    for kind in ("motor", "boat"):
        spans = []
        long_examples = []

        for (venue, number), (first, last, n) in (
            equipment[kind].items()
        ):
            from datetime import date
            span = (
                date.fromisoformat(last)
                - date.fromisoformat(first)
            ).days

            spans.append(span)

            if span >= 365:
                long_examples.append(
                    (span, venue, number, first, last, n)
                )

        print(
            kind,
            "unique venue+number:",
            len(spans),
            "observed span:",
            describe(spans)
        )

        print("Observed span >=365 days:",
              len(long_examples))

        for item in sorted(
            long_examples, reverse=True
        )[:10]:
            print(" ", item)

    print(
        "\nNOTE: A long span is evidence of "
        "number reuse or long-term records, "
        "not proof of a replacement date."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start", default="2022-01-01"
    )
    parser.add_argument(
        "--end", default="2024-12-31"
    )
    parser.add_argument(
        "--sample", type=int, default=30000
    )

    args = parser.parse_args()

    from datetime import date
    date.fromisoformat(args.start)
    date.fromisoformat(args.end)

    if args.start > args.end or args.sample < 1:
        raise ValueError("Invalid date range or sample size")

    print("=" * 68)
    print("PHASE 8.3-A MOTOR / BOAT FEATURE DIAGNOSTIC")
    print("ALL DATABASES READ ONLY")
    print("=" * 68, flush=True)

    con = connect()

    try:
        inspect_quality(con)
        inspect_years(con)
        inspect_finishes(
            con, args.start, args.end
        )
        inspect_sample(
            con, args.start, args.end, args.sample
        )

        print("\nPHASE 8.3-A DIAGNOSTIC COMPLETE")

    finally:
        con.close()


if __name__ == "__main__":
    main()
