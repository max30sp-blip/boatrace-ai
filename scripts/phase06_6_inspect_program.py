
from pathlib import Path
from collections import Counter
import argparse
import re

ROOT = Path(__file__).resolve().parent.parent
PROGRAM_DIR = ROOT / "data" / "extracted" / "programs"


def decode_file(path):
    raw = path.read_bytes()

    for encoding in ("cp932", "shift_jis"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            pass

    return raw.decode("cp932", errors="replace"), "cp932-replace"


def inspect_file(path, max_lines=100):
    print("\n" + "=" * 72)
    print("FILE:", path)
    print("=" * 72)

    text, encoding = decode_file(path)
    lines = text.splitlines()

    print("Encoding:", encoding)
    print("Total lines:", len(lines))

    print("\n=== FIRST LINES ===")

    for i, line in enumerate(lines[:max_lines], 1):
        print(f"{i:04d} | {line[:220]}")

    keywords = (
        "今節", "成績", "勝率", "選手",
        "スタート", "ＳＴ", "ST",
        "開催", "日目", "初日", "最終日",
        "節間", "出走表"
    )

    print("\n=== KEYWORD MATCHES ===")

    matches = []

    for i, line in enumerate(lines, 1):
        if any(word in line for word in keywords):
            matches.append((i, line))

    print("Matching lines:", len(matches))

    for i, line in matches[:60]:
        print(f"{i:04d} | {line[:220]}")

    print("\n=== LINE LENGTH DISTRIBUTION ===")

    lengths = Counter(len(line) for line in lines)

    for length, count in lengths.most_common(15):
        print(f"Length {length}: {count} lines")

    print("\n=== NUMERIC ROW SAMPLES ===")

    count = 0

    for i, line in enumerate(lines, 1):
        if re.match(r"^\s*[1-6]\s+\d{4}", line):
            print(f"{i:04d} | {line[:240]}")
            count += 1

        if count >= 12:
            break

    print("Numeric samples found:", count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dates",
        nargs="+",
        default=[
            "20160131",
            "20261029"
        ]
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=100
    )

    args = parser.parse_args()

    print("========== PHASE 6.6 INSPECTION ==========")
    print("Program directory:", PROGRAM_DIR)
    print("Mode: READ ONLY")

    for day in args.dates:
        if not re.fullmatch(r"\d{8}", day):
            print("Invalid date:", day)
            continue

        yy = day[2:4]
        mm = day[4:6]
        dd = day[6:8]
        year = day[:4]

        filename = f"b{yy}{mm}{dd}.txt"
        path = PROGRAM_DIR / year / filename

        if not path.exists():
            print("\nFILE NOT FOUND:", path)
            continue

        inspect_file(path, args.max_lines)

    print("\nPHASE 6.6 INSPECTION COMPLETE")


if __name__ == "__main__":
    main()
