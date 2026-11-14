
import csv
import sqlite3
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict
from itertools import groupby

ROOT = Path(r"C:\boatrace_ai")
DB = ROOT / "data" / "boatrace.db"
BASE = ROOT / "outputs" / "phase83b_f3_final_preview.csv"
OUT = ROOT / "outputs" / "phase83c_reset_f3_preview.csv"
REGISTRY = ROOT / "data" / "motor_reset_registry.csv"

# 公式情報で確認できたモーター使用開始日
# 他の場は確認後に追加する
CONFIRMED_RESETS = {
    23: date(2026, 9, 16),  # 唐津
}

WINDOWS = (30, 90, 180)
ALPHA = 20
PRIOR = 1 / 3

TARGET_START = date(2026, 10, 23)
TARGET_END = date(2026, 10, 29)

KEYS = ("race_date", "venue_code", "race_no", "lane")


def key(row):
    return tuple(str(row[k]) for k in KEYS)


def calculate(history, day, window):
    valid = [
        finish
        for past_day, finish in history
        if 0 < (day - past_day).days <= window
    ]

    n = len(valid)
    wins = sum(f == 1 for f in valid)
    top2 = sum(f <= 2 for f in valid)
    top3 = sum(f <= 3 for f in valid)

    return {
        "starts": n,
        "win_rate": wins / n if n else "",
        "top2_rate": top2 / n if n else "",
        "top3_rate": top3 / n if n else "",
        "top2_smoothed": (
            (top2 + ALPHA * PRIOR) / (n + ALPHA)
        )
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # 交換日管理表を保存
    with REGISTRY.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "venue_code",
            "motor_start_date",
            "verification"
        ])
        for venue, start in sorted(CONFIRMED_RESETS.items()):
            writer.writerow([
                venue, start.isoformat(), "official"
            ])

    with BASE.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)
        base_rows = list(reader)
        base_columns = list(reader.fieldnames or [])

    if not base_rows:
        raise ValueError("Base CSV is empty")

    if len({key(r) for r in base_rows}) != len(base_rows):
        raise ValueError("Duplicate race keys in base CSV")

    conn = sqlite3.connect(
        DB.as_uri() + "?mode=ro", uri=True
    )
    conn.execute("PRAGMA query_only=ON")

    corrections = {}

    for venue, start in sorted(CONFIRMED_RESETS.items()):
        sql = """
        SELECT race_date, race_no, lane,
               motor_no, finish_position
        FROM race_boat_results_v2
        WHERE venue_code = ?
          AND race_date BETWEEN ? AND ?
        ORDER BY race_date, race_no, lane
        """

        cursor = conn.execute(
            sql,
            (
                venue,
                start.isoformat(),
                TARGET_END.isoformat()
            )
        )

        histories = defaultdict(list)

        for day_string, grouped in groupby(
            cursor, key=lambda x: x[0]
        ):
            day = date.fromisoformat(day_string)
            day_rows = list(grouped)

            if TARGET_START <= day <= TARGET_END:
                for _, race_no, lane, motor, finish in day_rows:
                    if motor is None or motor <= 0:
                        continue

                    row_key = (
                        day_string,
                        str(venue),
                        str(race_no),
                        str(lane)
                    )

                    features = {}

                    for window in WINDOWS:
                        stats = calculate(
                            histories[motor], day, window
                        )

                        for name, value in stats.items():
                            features[
                                f"motor_reset_{window}_{name}"
                            ] = value

                    corrections[row_key] = features

            # 当日全レースの特徴量を計算後に更新
            for _, race_no, lane, motor, finish in day_rows:
                if (
                    motor is not None
                    and motor > 0
                    and finish is not None
                    and 1 <= finish <= 6
                ):
                    histories[motor].append((day, finish))

        print(
            "Processed venue:", venue,
            "motor start:", start
        )

    conn.close()

    corrected_columns = [
        f"motor_reset_{w}_{name}"
        for w in WINDOWS
        for name in (
            "starts", "win_rate", "top2_rate",
            "top3_rate", "top2_smoothed"
        )
    ]

    extra_columns = [
        "motor_reset_status",
        "motor_reset_start_date"
    ] + corrected_columns

    counts = defaultdict(int)

    with OUT.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=base_columns + extra_columns
        )
        writer.writeheader()

        for row in base_rows:
            venue = int(row["venue_code"])
            row_key = key(row)

            row["motor_reset_status"] = "unverified"
            row["motor_reset_start_date"] = ""

            for column in corrected_columns:
                row[column] = ""

            if venue in CONFIRMED_RESETS:
                corrected = corrections.get(row_key)

                if corrected is None:
                    raise ValueError(
                        f"Missing correction: {row_key}"
                    )

                row["motor_reset_status"] = "verified"
                row["motor_reset_start_date"] = (
                    CONFIRMED_RESETS[venue].isoformat()
                )
                row.update(corrected)
                counts["verified"] += 1
            else:
                counts["unverified"] += 1

            writer.writerow(row)

    print("\n=== PHASE 8.3-C RESET F3 COMPLETE ===")
    print("Total rows:", len(base_rows))
    print("Verified rows:", counts["verified"])
    print("Unverified rows:", counts["unverified"])
    print("Output columns:", len(base_columns + extra_columns))
    print("Registry:", REGISTRY)
    print("Output:", OUT)


if __name__ == "__main__":
    main()
