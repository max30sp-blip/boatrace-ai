
import sqlite3
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--db", required=True)
args = parser.parse_args()

db_path = Path(args.db).resolve()

if not db_path.is_file():
    raise FileNotFoundError(db_path)

# 読み取り専用でDBを開く
conn = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)

tables = conn.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name
""").fetchall()

keywords = [
    "motor", "boat", "feature",
    "entry", "race", "result",
    "history", "exchange", "part"
]

print("=== PHASE 8.3-B DB DIAGNOSIS ===")
print("Database:", db_path)
print("Total tables:", len(tables))

for (table_name,) in tables:
    if not any(k in table_name.lower() for k in keywords):
        continue

    print("\n" + "=" * 60)
    print("TABLE:", table_name)

    safe_name = table_name.replace('"', '""')

    columns = conn.execute(
        f'PRAGMA table_info("{safe_name}")'
    ).fetchall()

    for col in columns:
        print(f"  {col[1]:35s} {col[2]}")

    print("\nSAMPLE:")
    try:
        rows = conn.execute(
            f'SELECT * FROM "{safe_name}" LIMIT 2'
        ).fetchall()

        column_names = [col[1] for col in columns]

        for row in rows:
            print(dict(zip(column_names, row)))

    except sqlite3.Error as e:
        print("Sample error:", e)

conn.close()
print("\n=== DIAGNOSIS COMPLETE ===")
