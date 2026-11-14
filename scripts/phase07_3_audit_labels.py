
import sqlite3
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

TRAIN_DB = DATA / "train_dataset_v1.db"
BASE_DB = DATA / "boatrace.db"


def ro_uri(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve().as_uri() + "?mode=ro"


def main():
    print("=" * 65)
    print("PHASE 7.3 OFFICIAL TRIFECTA LABEL AUDIT")
    print("READ ONLY")
    print("=" * 65, flush=True)

    con = sqlite3.connect(
        ro_uri(TRAIN_DB),
        uri=True,
        timeout=120
    )

    try:
        con.execute(
            "ATTACH DATABASE ? AS base",
            (ro_uri(BASE_DB),)
        )
        con.execute("PRAGMA query_only=ON")

        sql = """
        WITH all_payouts AS (
            SELECT
                race_date,
                venue_code,
                race_no,
                TRIM(combination) AS combo,
                'normal' AS source_type
            FROM base.trifecta_results

            UNION ALL

            SELECT
                race_date,
                venue_code,
                race_no,
                TRIM(combination) AS combo,
                'supplement' AS source_type
            FROM base.trifecta_payouts_supplement
        ),
        payouts AS (
            SELECT
                race_date,
                venue_code,
                race_no,
                COUNT(*) AS payout_rows,
                COUNT(DISTINCT combo) AS distinct_combos,
                MIN(combo) AS single_combo,
                GROUP_CONCAT(DISTINCT combo) AS combinations,
                SUM(
                    CASE WHEN source_type = 'normal'
                    THEN 1 ELSE 0 END
                ) AS normal_rows,
                SUM(
                    CASE WHEN source_type = 'supplement'
                    THEN 1 ELSE 0 END
                ) AS supplement_rows
            FROM all_payouts
            GROUP BY
                race_date, venue_code, race_no
        )
        SELECT
            t.race_date,
            t.venue_code,
            t.race_no,
            t.label,
            p.payout_rows,
            p.distinct_combos,
            p.single_combo,
            p.combinations,
            p.normal_rows,
            p.supplement_rows
        FROM train_races t
        LEFT JOIN payouts p
          ON p.race_date = t.race_date
         AND p.venue_code = t.venue_code
         AND p.race_no = t.race_no
        ORDER BY t.race_date, t.venue_code, t.race_no
        """

        summary = Counter()
        examples = {}
        yearly = Counter()
        processed = 0

        for row in con.execute(sql):
            (
                race_date,
                venue_code,
                race_no,
                label,
                payout_rows,
                distinct_combos,
                single_combo,
                combinations,
                normal_rows,
                supplement_rows
            ) = row

            processed += 1

            if distinct_combos is None:
                status = "NO_OFFICIAL_PAYOUT"

            elif distinct_combos == 0:
                status = "EMPTY_COMBINATION"

            elif distinct_combos > 1:
                status = "MULTIPLE_PAYOUTS"

            elif label != single_combo:
                status = "LABEL_MISMATCH"

            else:
                status = "EXACT_MATCH"

            summary[status] += 1

            if normal_rows and supplement_rows:
                summary["BOTH_PAYOUT_SOURCES"] += 1

            if payout_rows and distinct_combos:
                if payout_rows > distinct_combos:
                    summary["DUPLICATED_PAYOUT_ROWS"] += 1

            if status != "EXACT_MATCH":
                yearly[(race_date[:4], status)] += 1

                if len(examples.get(status, [])) < 10:
                    examples.setdefault(status, []).append({
                        "date": race_date,
                        "venue": venue_code,
                        "race": race_no,
                        "label": label,
                        "payouts": combinations,
                        "normal": normal_rows,
                        "supplement": supplement_rows,
                    })

            if processed % 100000 == 0:
                print(
                    f"Audited: {processed:,}",
                    flush=True
                )

        print("\n=== SUMMARY ===")
        print("Audited races:", f"{processed:,}")

        for status in (
            "EXACT_MATCH",
            "MULTIPLE_PAYOUTS",
            "LABEL_MISMATCH",
            "NO_OFFICIAL_PAYOUT",
            "EMPTY_COMBINATION",
        ):
            print(
                f"{status:24s}",
                f"{summary[status]:,}"
            )

        print(
            "BOTH_PAYOUT_SOURCES:",
            f"{summary['BOTH_PAYOUT_SOURCES']:,}"
        )
        print(
            "DUPLICATED_PAYOUT_ROWS:",
            f"{summary['DUPLICATED_PAYOUT_ROWS']:,}"
        )

        accounted = sum(
            summary[x]
            for x in (
                "EXACT_MATCH",
                "MULTIPLE_PAYOUTS",
                "LABEL_MISMATCH",
                "NO_OFFICIAL_PAYOUT",
                "EMPTY_COMBINATION",
            )
        )

        print("\nAccounted for:", f"{accounted:,}")
        print("Accounting OK:", accounted == processed)

        pct = 100 * summary["EXACT_MATCH"] / max(processed, 1)
        print(f"Exact match rate: {pct:.4f}%")

        print("\n=== PROBLEMS BY YEAR ===")
        for (year, status), count in sorted(yearly.items()):
            print(year, status, count)

        print("\n=== EXAMPLES ===")
        for status, rows in examples.items():
            print(f"\n[{status}]")
            for item in rows:
                print(item)

        print("\n=== FINAL DECISION ===")

        bad = (
            summary["LABEL_MISMATCH"]
            + summary["NO_OFFICIAL_PAYOUT"]
            + summary["EMPTY_COMBINATION"]
        )

        if bad:
            print("LABEL CONSISTENCY REVIEW REQUIRED")
        elif summary["MULTIPLE_PAYOUTS"]:
            print("MULTIPLE-PAYOUT RACES REQUIRE FILTERING")
        else:
            print("OFFICIAL LABEL AUDIT PASSED")

        print("\nPHASE 7.3 AUDIT COMPLETE")

    finally:
        con.close()


if __name__ == "__main__":
    main()
