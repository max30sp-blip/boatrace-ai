
import argparse
import hashlib
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ========================================
# PHASE 01 : BOAT RACE DATA COLLECTION
# ========================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "boatrace.db"

BASE_URL = "https://www1.mbrace.or.jp/od2"

DATA_TYPES = {
    "B": "programs",
    "K": "results",
}


def init_database():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_date TEXT NOT NULL,
                data_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                source_url TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                downloaded_at TEXT NOT NULL,
                UNIQUE(race_date, data_type)
            )
        """)

    print("[OK] Database initialized:", DB_PATH)


def date_range(start, end):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_url(day, data_type):
    year_month = day.strftime("%Y%m")
    short_date = day.strftime("%y%m%d")
    filename = f"{data_type.lower()}{short_date}.lzh"

    url = (
        f"{BASE_URL}/{data_type}/"
        f"{year_month}/{filename}"
    )

    return url, filename


def valid_lzh(content):
    # LZHの圧縮方式ヘッダを簡易検査
    if len(content) < 16:
        return False

    method = content[2:7]
    return (
        method.startswith(b"-lh")
        or method.startswith(b"-lz")
    )


def download_file(url, destination, retries=3):
    if destination.exists():
        data = destination.read_bytes()
        if valid_lzh(data):
            return data, "cached"
        destination.unlink()

    for attempt in range(1, retries + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            with urlopen(request, timeout=30) as response:
                content = response.read()

            if not valid_lzh(content):
                raise ValueError(
                    "Response is not a valid LZH header"
                )

            # 途中終了でも壊れたファイルを残さない
            temp_path = destination.with_suffix(".tmp")
            temp_path.write_bytes(content)
            temp_path.replace(destination)

            return content, "downloaded"

        except HTTPError as exc:
            if exc.code == 404:
                return None, "not_found"

            print(
                f"[WARN] HTTP {exc.code} "
                f"attempt {attempt}/{retries}"
            )

        except (URLError, OSError, ValueError) as exc:
            print(
                f"[WARN] {exc} "
                f"attempt {attempt}/{retries}"
            )

        if attempt < retries:
            time.sleep(3 * attempt)

    return None, "failed"


def register_file(day, data_type, path, url, content):
    file_hash = hashlib.sha256(content).hexdigest()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO raw_files (
                race_date,
                data_type,
                file_path,
                source_url,
                file_size,
                sha256,
                downloaded_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(race_date, data_type)
            DO UPDATE SET
                file_path=excluded.file_path,
                source_url=excluded.source_url,
                file_size=excluded.file_size,
                sha256=excluded.sha256,
                downloaded_at=excluded.downloaded_at
        """, (
            day.isoformat(),
            data_type,
            str(path.relative_to(ROOT)),
            url,
            len(content),
            file_hash,
            datetime.now(timezone.utc).isoformat()
        ))


def collect(start, end, delay):
    init_database()

    stats = {
        "downloaded": 0,
        "cached": 0,
        "not_found": 0,
        "failed": 0,
    }

    for day in date_range(start, end):
        print(f"\n=== {day.isoformat()} ===")

        for data_type, folder_name in DATA_TYPES.items():
            url, filename = build_url(day, data_type)

            folder = RAW_DIR / folder_name / str(day.year)
            folder.mkdir(parents=True, exist_ok=True)

            destination = folder / filename

            content, status = download_file(
                url, destination
            )

            stats[status] += 1

            if content is not None:
                register_file(
                    day, data_type, destination,
                    url, content
                )

                print(
                    f"[OK] {data_type} | {status} "
                    f"| {len(content):,} bytes"
                )
            else:
                print(f"[SKIP] {data_type} | {status}")

            if status != "cached":
                time.sleep(delay)

    print("\n========== SUMMARY ==========")

    for key, value in stats.items():
        print(f"{key}: {value}")

    with sqlite3.connect(DB_PATH) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM raw_files"
        ).fetchone()[0]

        total_size = conn.execute(
            "SELECT COALESCE(SUM(file_size), 0) "
            "FROM raw_files"
        ).fetchone()[0]

    print(f"Registered files: {total:,}")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    print("Database:", DB_PATH)


def main():
    parser = argparse.ArgumentParser(
        description="BOAT RACE official data collector"
    )

    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=date(2026, 10, 23)
    )

    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date(2026, 10, 29)
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.5
    )

    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start must be before --end")

    if args.end > date.today():
        parser.error("Future dates are not allowed")

    if args.delay < 1:
        parser.error("--delay must be at least 1")

    collect(args.start, args.end, args.delay)


if __name__ == "__main__":
    main()
