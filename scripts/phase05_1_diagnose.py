
import re
import sqlite3
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "boatrace.db"
RESULTS = ROOT / "data" / "extracted" / "results"


def normalize(s):
    return unicodedata.normalize("NFKC", s)


with sqlite3.connect(DB) as conn:
    missing = conn.execute("""
        SELECT
            r.race_date,
            r.venue_code,
            r.race_no,
            r.combination
        FROM trifecta_results r
        LEFT JOIN (
            SELECT race_date, venue_code, race_no
            FROM race_boat_results
            GROUP BY race_date, venue_code, race_no
        ) b
        ON r.race_date = b.race_date
        AND r.venue_code = b.venue_code
        AND r.race_no = b.race_no
        WHERE b.race_no IS NULL
        ORDER BY r.race_date DESC, r.venue_code, r.race_no
    """).fetchall()

print("Missing races with trifecta results:", len(missing))

for day, venue, race_no, combo in missing[:8]:
    filename = "k" + day[2:4] + day[5:7] + day[8:10] + ".txt"
    path = RESULTS / day[:4] / filename

    if not path.exists():
        print("File missing:", path)
        continue

    lines = path.read_text(encoding="cp932").splitlines()
    current_venue = None
    start = None

    for i, raw in enumerate(lines):
        line = normalize(raw)

        m = re.match(r"^\s*(\d{2})KBGN", line)
        if m:
            current_venue = int(m.group(1))

        m = re.match(
            r"^\s*(\d{1,2})R\s+.*?\bH\s*\d{3,4}\s*m\b",
            line,
            re.I
        )

        if (
            m
            and current_venue == venue
            and int(m.group(1)) == race_no
        ):
            start = i
            break

    print(
        f"\n===== {day} Venue {venue} "
        f"Race {race_no} / {combo} ====="
    )

    if start is None:
        print("Race header not found")
        continue

    for j in range(start, min(start + 14, len(lines))):
        print(f"{j+1:04d}: {normalize(lines[j])}")
