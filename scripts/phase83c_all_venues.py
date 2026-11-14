
import csv
import sqlite3
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict, deque
from itertools import groupby

ROOT = Path(r"C:\boatrace_ai")
DB = ROOT / "data" / "boatrace.db"
BASE = ROOT / "outputs" / "phase83b_f3_final_preview.csv"

REGISTRY = ROOT / "data" / "motor_reset_candidates_2026.csv"
OUTPUT = ROOT / "outputs" / "phase83c_all_venues_preview.csv"
REPORT = ROOT / "outputs" / "phase83c_venue_impact.csv"

TARGET_START = date(2026, 10, 23)
TARGET_END = date(2026, 10, 29)
WINDOWS = (30, 90, 180)

ALPHA = 20
PRIOR = 1 / 3

# 2026年10月時点で使用中のモーターの開始日候補
# 日付は二次資料による候補。公式確認前は学習に採用しない。
CANDIDATES = {
    1: ("桐生", "2025-12-27"),
    2: ("戸田", "2026-08-06"),
    3: ("江戸川", "2026-05-11"),
    4: ("平和島", "2026-06-18"),
    5: ("多摩川", "2026-04-18"),
    6: ("浜名湖", "2026-04-09"),
    7: ("蒲郡", "2026-08-11"),
    8: ("常滑", "2026-10-06"),
    9: ("津", "2025-12-22"),
    10: ("三国", "2026-03-07"),
    11: ("びわこ", "2026-04-08"),
    12: ("住之江", "2026-03-23"),
    13: ("尼崎", "2026-04-17"),
    14: ("鳴門", "2026-04-11"),
    15: ("丸亀", "2026-09-17"),
    16: ("児島", "2025-12-17"),
    17: ("宮島", "2026-10-20"),
    18: ("徳山", "2026-04-20"),
    19: ("下関", "2026-04-29"),
    20: ("若松", "2025-11-26"),
    21: ("芦屋", "2026-04-16"),
    22: ("福岡", "2026-02-18"),
    23: ("唐津", "2026-09-16"),
    24: ("大村", "2026-05-24"),
}

# 現時点で公式サイトまたは公式発表に
# 使用開始日を裏付ける情報がある場。
# 予定日については別途実施状況を確認する。
OFFICIAL_EVIDENCE = {5, 6, 13, 18, 19, 21, 23}

KEYS = ("race_date", "venue_code", "race_no", "lane")


def row_key(row):
    return tuple(str(row[x]) for x in KEYS)


def calculate(history, day, days):
    values = [
        finish
        for previous_day, finish in history
        if 0 < (day - previous_day).days < days
    ]

    n = len(values)
    wins = sum(f == 1 for f in values)
    top2 = sum(f <= 2 for f in values)
    top3 = sum(f <= 3 for f in values)

    return {
        "starts": n,
        "win_rate": wins / n if n else "",
        "top2_rate": top2 / n if n else "",
        "top3_rate": top3 / n if n else "",
        "top2_smoothed": (
            top2 + ALPHA * PRIOR
        ) / (n + ALPHA),
    }


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # 候補日の台帳。既存の確認済み台帳は上書きしない。
    with REGISTRY.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "venue_code", "venue_name",
            "motor_start_date", "evidence_status"
        ])

        for venue, (name, start) in sorted(CANDIDATES.items()):
            status = (
                "official_evidence"
                if venue in OFFICIAL_EVIDENCE
                else "secondary_candidate"
            )
            writer.writerow([venue, name, start, status])

    with BASE.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)
        base_rows = list(reader)
        base_columns = reader.fieldnames or []

    if len({row_key(r) for r in base_rows}) != len(base_rows):
        raise ValueError("Duplicate base keys")

    conn = sqlite3.connect(
        DB.as_uri() + "?mode=ro",
        uri=True
    )
    conn.execute("PRAGMA query_only=ON")

    corrections = {}

    for venue, (name, start_string) in sorted(CANDIDATES.items()):
        start = date.fromisoformat(start_string)

        # 180日の窓を超えた履歴は不要
        query_start = max(
            start,
            TARGET_START - timedelta(days=max(WINDOWS))
        )

        records = conn.execute("""
            SELECT race_date, race_no, lane,
                   motor_no, finish_position
            FROM race_boat_results_v2
            WHERE venue_code = ?
              AND race_date BETWEEN ? AND ?
            ORDER BY race_date, race_no, lane
        """, (
            venue, query_start.isoformat(),
            TARGET_END.isoformat()
        ))

        history = defaultdict(deque)
        count = 0

        for day_string, group in groupby(
            records, key=lambda r: r[0]
        ):
            day = date.fromisoformat(day_string)
            day_rows = list(group)

            # 当日の着順は当日の予測に使わない
            if TARGET_START <= day <= TARGET_END:
                for _, race_no, lane, motor, finish in day_rows:
                    if motor is None or motor <= 0:
                        continue

                    k = (
                        day_string, str(venue),
                        str(race_no), str(lane)
                    )

                    features = {}

                    for window in WINDOWS:
                        stats = calculate(
                            history[motor], day, window
                        )

                        for metric, value in stats.items():
                            features[
                                f"reset_motor_{window}_{metric}"
                            ] = value

                    corrections[k] = features
                    count += 1

            # 特徴量計算後に当日結果を履歴に追加
            for _, _, _, motor, finish in day_rows:
                if (
                    motor is not None
                    and motor > 0
                    and finish is not None
                    and 1 <= finish <= 6
                ):
                    history[motor].append((day, finish))

        print(
            f"Venue {venue:02d} {name}: "
            f"{count} rows, start={start_string}"
        )

    conn.close()

    metric_names = (
        "starts", "win_rate", "top2_rate",
        "top3_rate", "top2_smoothed"
    )

    reset_columns = [
        f"reset_motor_{w}_{metric}"
        for w in WINDOWS
        for metric in metric_names
    ]

    extra_columns = [
        "reset_motor_start_date",
        "reset_evidence_status"
    ] + reset_columns

    impact = defaultdict(
        lambda: {"rows": 0, "changed": 0, "sum_diff": 0.0}
    )

    with OUTPUT.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=base_columns + extra_columns
        )
        writer.writeheader()

        for base in base_rows:
            r = dict(base)
            venue = int(r["venue_code"])
            candidate = CANDIDATES.get(venue)

            if candidate is None:
                raise ValueError(f"Unknown venue: {venue}")

            correction = corrections.get(row_key(r))
            if correction is None:
                raise ValueError(
                    f"Missing correction: {row_key(r)}"
                )

            r["reset_motor_start_date"] = candidate[1]
            r["reset_evidence_status"] = (
                "official_evidence"
                if venue in OFFICIAL_EVIDENCE
                else "secondary_candidate"
            )

            r.update(correction)
            writer.writerow(r)

            old = float(r["motor_180_top2_smoothed"])
            new = float(
                r["reset_motor_180_top2_smoothed"]
            )

            stats = impact[venue]
            stats["rows"] += 1
            stats["sum_diff"] += abs(new - old)

            if abs(new - old) > 1e-9:
                stats["changed"] += 1

    with REPORT.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "venue_code", "venue_name",
            "rows", "changed_rows",
            "mean_absolute_top2_difference",
            "evidence_status"
        ])

        for venue, stats in sorted(impact.items()):
            writer.writerow([
                venue,
                CANDIDATES[venue][0],
                stats["rows"],
                stats["changed"],
                stats["sum_diff"] / stats["rows"],
                (
                    "official_evidence"
                    if venue in OFFICIAL_EVIDENCE
                    else "secondary_candidate"
                ),
            ])

    print("\n=== PHASE 8.3-C ALL VENUES COMPLETE ===")
    print("Total rows:", len(base_rows))
    print("Corrected rows:", len(corrections))
    print("Output columns:", len(base_columns + extra_columns))
    print("Official-evidence venues:", len(OFFICIAL_EVIDENCE))
    print("Candidate-only venues:", 24 - len(OFFICIAL_EVIDENCE))
    print("Registry:", REGISTRY)
    print("Output:", OUTPUT)
    print("Impact report:", REPORT)


if __name__ == "__main__":
    main()
