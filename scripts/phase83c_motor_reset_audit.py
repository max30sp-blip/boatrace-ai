
import csv
import sqlite3
from pathlib import Path
from datetime import date
from collections import defaultdict

ROOT = Path(r"C:\boatrace_ai")
DB = ROOT / "data" / "boatrace.db"
BASE = ROOT / "outputs" / "phase83b_f3_final_preview.csv"
OUT = ROOT / "outputs" / "phase83c_motor_reset_audit.csv"

# 唐津：公式に確認できた2026年モーター使用開始日
VENUE = 23
MOTOR_START = date(2026, 9, 16)

TARGET_START = date(2026, 10, 23)
TARGET_END = date(2026, 10, 29)

ALPHA = 20
PRIOR = 1 / 3


def load_previous():
    with BASE.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        return [
            r for r in csv.DictReader(f)
            if int(r["venue_code"]) == VENUE
        ]


def main():
    previous = load_previous()
    print("=== PHASE 8.3-C MOTOR RESET AUDIT ===")
    print("Venue:", VENUE)
    print("Motor start:", MOTOR_START)
    print("Target rows:", len(previous))

    if not previous:
        print("No target rows. Nothing to compare.")
        return

    conn = sqlite3.connect(
        DB.as_uri() + "?mode=ro", uri=True
    )
    conn.execute("PRAGMA query_only=ON")

    sql = """
        SELECT
            race_date, race_no, lane,
            motor_no, finish_position
        FROM race_boat_results_v2
        WHERE venue_code = ?
          AND race_date BETWEEN ? AND ?
        ORDER BY race_date, race_no, lane
    """

    records = conn.execute(
        sql,
        (
            VENUE,
            MOTOR_START.isoformat(),
            TARGET_END.isoformat()
        )
    ).fetchall()

    conn.close()

    # motor_no -> 過去の着順履歴
    history = defaultdict(list)
    recalculated = {}

    from itertools import groupby

    for day_string, group in groupby(
        records, key=lambda r: r[0]
    ):
        day = date.fromisoformat(day_string)
        day_rows = list(group)

        # 当日の結果を追加する前に計算
        if TARGET_START <= day <= TARGET_END:
            for _, race_no, lane, motor, finish in day_rows:
                if motor is None or motor <= 0:
                    continue

                past = history[motor]

                n = len(past)
                top2 = sum(x <= 2 for x in past)
                top3 = sum(x <= 3 for x in past)
                wins = sum(x == 1 for x in past)

                key = (
                    day_string,
                    str(race_no),
                    str(lane)
                )

                recalculated[key] = {
                    "new_starts": n,
                    "new_win_rate": wins / n if n else "",
                    "new_top2_rate": top2 / n if n else "",
                    "new_top3_rate": top3 / n if n else "",
                    "new_top2_smoothed": (
                        top2 + ALPHA * PRIOR
                    ) / (n + ALPHA)
                }

        # 当日の予測処理が終わった後に履歴追加
        for _, race_no, lane, motor, finish in day_rows:
            if (
                motor is not None
                and motor > 0
                and finish is not None
                and 1 <= finish <= 6
            ):
                history[motor].append(finish)

    output = []
    missing = 0

    for row in previous:
        key = (
            row["race_date"],
            row["race_no"],
            row["lane"]
        )

        new = recalculated.get(key)
        if new is None:
            missing += 1
            continue

        old_starts = int(row["motor_180_starts"])
        new_starts = new["new_starts"]

        old_rate = float(
            row["motor_180_top2_smoothed"]
        )
        new_rate = new["new_top2_smoothed"]

        output.append({
            "race_date": row["race_date"],
            "venue_code": VENUE,
            "race_no": row["race_no"],
            "lane": row["lane"],
            "motor_no": row["motor_no"],
            "old_180_starts": old_starts,
            "new_starts": new_starts,
            "excluded_old_starts": old_starts - new_starts,
            "old_top2_smoothed": old_rate,
            "new_top2_smoothed": new_rate,
            "top2_difference": new_rate - old_rate
        })

    if not output:
        print("No matched rows.")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(output[0].keys())
        )
        writer.writeheader()
        writer.writerows(output)

    affected = [
        r for r in output
        if r["excluded_old_starts"] > 0
    ]

    differences = [
        abs(r["top2_difference"])
        for r in output
    ]

    print("\n=== COMPARISON ===")
    print("Matched rows:", len(output))
    print("Unmatched rows:", missing)
    print("Affected rows:", len(affected))

    print(
        "Average excluded starts:",
        round(
            sum(r["excluded_old_starts"] for r in output)
            / len(output), 3
        )
    )

    print(
        "Average absolute top2 difference:",
        round(sum(differences) / len(differences), 6)
    )

    print(
        "Maximum absolute difference:",
        round(max(differences), 6)
    )

    print("\n=== TOP 10 CHANGES ===")

    ranked = sorted(
        output,
        key=lambda r: abs(r["top2_difference"]),
        reverse=True
    )

    for r in ranked[:10]:
        print(
            r["race_date"],
            "R", r["race_no"],
            "lane", r["lane"],
            "motor", r["motor_no"],
            "old_n", r["old_180_starts"],
            "new_n", r["new_starts"],
            "difference", round(r["top2_difference"], 5)
        )

    print("\n=== PHASE 8.3-C AUDIT COMPLETE ===")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
