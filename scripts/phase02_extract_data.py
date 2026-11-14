
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "extracted"


def find_7zip():
    candidates = [
        shutil.which("7z"),
        shutil.which("7za"),
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)

    raise RuntimeError(
        "7-Zip が見つかりません。\n"
        "winget install --id 7zip.7zip -e"
    )


def find_archives():
    if not RAW_DIR.exists():
        return []

    files = list(RAW_DIR.rglob("*.lzh"))

    # 最新のファイルから確認する
    return sorted(
        files,
        key=lambda p: p.name[1:7],
        reverse=True
    )


def detect_encoding(content):
    for encoding in ("cp932", "shift_jis"):
        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    return content.decode("cp932", errors="replace"), "cp932(replace)"


def extract_one(archive, seven_zip):
    relative = archive.relative_to(RAW_DIR)
    destination = OUTPUT_DIR / relative.parent
    destination.mkdir(parents=True, exist_ok=True)

    # 圧縮ファイル1個につき展開テキスト1個を想定
    output_path = destination / (archive.stem + ".txt")

    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path, "cached"

    with tempfile.TemporaryDirectory() as tmp:
        command = [
            seven_zip,
            "x",
            "-y",
            f"-o{tmp}",
            str(archive)
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"7-Zip error {result.returncode}: "
                f"{result.stderr[-500:]} "
                f"{result.stdout[-500:]}"
            )

        extracted = [
            p for p in Path(tmp).rglob("*")
            if p.is_file()
        ]

        if len(extracted) != 1:
            raise RuntimeError(
                f"Expected 1 extracted file, got {len(extracted)}"
            )

        data = extracted[0].read_bytes()

        if not data:
            raise RuntimeError("Extracted file is empty")

        # 同一フォルダ内の一時ファイルを最後に置き換える
        temporary = output_path.with_suffix(".partial")
        temporary.write_bytes(data)
        temporary.replace(output_path)

    return output_path, "extracted"


def inspect_file(path, show_lines):
    content = path.read_bytes()
    text, encoding = detect_encoding(content)
    lines = text.splitlines()

    nonempty = [line for line in lines if line.strip()]

    print(f"Encoding: {encoding}")
    print(f"Size: {len(content):,} bytes")
    print(f"Total lines: {len(lines):,}")
    print(f"Nonempty lines: {len(nonempty):,}")

    print("----- SAMPLE START -----")

    for i, line in enumerate(lines[:show_lines], 1):
        print(f"{i:03d}: {line[:180]}")

    print("----- SAMPLE END -----")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit", type=int, default=14
    )

    parser.add_argument(
        "--show", type=int, default=12
    )

    args = parser.parse_args()

    if args.limit < 1 or args.show < 0:
        parser.error("--limit must be >= 1 and --show >= 0")

    seven_zip = find_7zip()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    archives = find_archives()[:args.limit]

    if not archives:
        print("LZH files not found:", RAW_DIR)
        return

    print("PHASE 02 - EXTRACT OFFICIAL DATA")
    print("7-Zip:", seven_zip)
    print("Archives:", len(archives))

    success = 0
    failures = 0

    # BとKそれぞれの構造を最初の1件で確認する
    inspected_types = set()

    for archive in archives:
        print(f"\n=== {archive.name} ===")

        try:
            output, status = extract_one(
                archive, seven_zip
            )

            print("Status:", status)
            print("Output:", output.relative_to(ROOT))

            success += 1

            data_type = archive.stem[0].lower()

            if data_type not in inspected_types:
                inspect_file(output, args.show)
                inspected_types.add(data_type)

        except Exception as exc:
            failures += 1
            print("[ERROR]", exc)

    print("\n========== SUMMARY ==========")
    print("Processed:", len(archives))
    print("Success:", success)
    print("Failed:", failures)
    print("Inspected types:", sorted(inspected_types))


if __name__ == "__main__":
    main()
