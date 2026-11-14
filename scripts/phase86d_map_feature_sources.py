
import ast
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"C:\boatrace_ai")
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SCRIPT_DIR = ROOT / "scripts"

REPORT = ROOT / "models/phase08_1_b_report.json"
TRAIN_DB = DATA_DIR / "train_dataset_v1.db"

OUT = OUTPUT_DIR / "phase86d_source_map.json"
TARGET = "2026-10-29"

GROUPS = ("f1", "f2", "f3", "f4", "f5", "f6")


def readonly(path):
    conn = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True
    )
    conn.execute("PRAGMA query_only=ON")
    return conn


def quote(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def source_database_paths():
    paths = set(DATA_DIR.glob("*.db"))
    paths.update(OUTPUT_DIR.glob("*.db"))

    # 学習DBは正解値の参照専用。
    paths.discard(TRAIN_DB)

    return sorted(
        (p for p in paths if p.is_file()),
        key=lambda p: str(p)
    )


def get_table_schemas(path):
    conn = readonly(path)
    schemas = []

    try:
        tables = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """).fetchall()

        for (table,) in tables:
            columns = [
                item[1]
                for item in conn.execute(
                    f"PRAGMA table_info({quote(table)})"
                )
            ]

            schemas.append({
                "db": str(path),
                "table": table,
                "columns": columns,
            })

    finally:
        conn.close()

    return schemas


def load_model_features():
    if not REPORT.is_file():
        raise FileNotFoundError(REPORT)

    with REPORT.open(encoding="utf-8-sig") as f:
        report = json.load(f)

    features = report["feature_names"]

    if len(features) != 618:
        raise ValueError(
            f"Expected 618 features, got {len(features)}"
        )

    groups = defaultdict(list)

    for name in features:
        m = re.fullmatch(
            r"b([1-6])_(f[0-6]|p)_(.+)",
            name
        )

        if m is None:
            raise ValueError(
                f"Unknown feature naming: {name}"
            )

        lane, group, suffix = m.groups()

        groups[group].append({
            "full_name": name,
            "lane": int(lane),
            "suffix": suffix,
        })

    return groups


def map_table(group, features, schema):
    columns = set(schema["columns"])

    required_keys = {
        "race_date",
        "venue_code",
        "race_no",
    }

    if not required_keys.issubset(columns):
        return None

    if "lane" in columns:
        boat_key = "lane"
    elif "boat_no" in columns:
        boat_key = "boat_no"
    else:
        return None

    # 各グループの1号艇分だけから、
    # 元のデータ列名を推定する。
    one_boat = [
        item["suffix"]
        for item in features
        if item["lane"] == 1
    ]

    mapping = {}

    for suffix in one_boat:
        # 完全一致を最優先。
        candidates = (
            suffix,
            f"{group}_{suffix}",
        )

        matched = next(
            (name for name in candidates if name in columns),
            None
        )

        if matched is not None:
            mapping[suffix] = matched

    if not mapping:
        return None

    result = {
        "db": schema["db"],
        "table": schema["table"],
        "boat_key": boat_key,
        "mapped_count_per_boat": len(mapping),
        "total_per_boat": len(one_boat),
        "mapping": mapping,
    }

    # 同じ日付のデータがあるかも確認する。
    try:
        conn = readonly(Path(schema["db"]))

        query = (
            f"SELECT COUNT(*) FROM {quote(schema['table'])}"
            " WHERE race_date = ?"
        )

        result["target_date_rows"] = conn.execute(
            query, (TARGET,)
        ).fetchone()[0]

        conn.close()

    except sqlite3.Error as exc:
        result["target_date_error"] = str(exc)

    return result


def analyze_python_sources(groups):
    scripts = []

    for path in sorted(SCRIPT_DIR.glob("*.py")):
        try:
            content = path.read_text(
                encoding="utf-8-sig",
                errors="replace"
            )

            tree = ast.parse(content)

        except (OSError, SyntaxError, UnicodeError):
            continue

        lines = content.splitlines()

        # 対象グループの特徴量の接尾辞が
        # ソースコードに実際に現れるかを確認。
        hits = defaultdict(list)

        for group in GROUPS:
            suffixes = {
                item["suffix"]
                for item in groups[group]
                if item["lane"] == 1
            }

            for number, line in enumerate(lines, 1):
                stripped = line.strip()

                if not stripped or stripped.startswith("#"):
                    continue

                found = [
                    suffix for suffix in suffixes
                    if suffix in line
                ]

                if found:
                    hits[group].append({
                        "line": number,
                        "text": stripped[:180],
                        "features": sorted(found)[:6],
                    })

        if not hits:
            continue

        functions = []

        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                functions.append({
                    "name": node.name,
                    "start": node.lineno,
                    "end": node.end_lineno,
                })

        scripts.append({
            "path": str(path),
            "line_count": len(lines),
            "hits": {
                group: items[:15]
                for group, items in hits.items()
            },
            "functions": functions,
        })

    return scripts


def main():
    print("=== PHASE 8.6-D SOURCE MAPPING ===")

    groups = load_model_features()

    print("\n=== MODEL GROUPS ===")

    for group in GROUPS:
        count = len(groups[group])
        print(
            f"{group}: {count} columns, "
            f"{count // 6} per boat"
        )

    print("\n=== DATABASE DISCOVERY ===")

    paths = source_database_paths()
    schemas = []

    for path in paths:
        try:
            found = get_table_schemas(path)
            schemas.extend(found)

        except sqlite3.Error as exc:
            print("Skipped DB:", path, exc)

    print("Databases:", len(paths))
    print("Tables inspected:", len(schemas))

    mappings = {}

    for group in GROUPS:
        candidates = []

        for schema in schemas:
            candidate = map_table(
                group, groups[group], schema
            )

            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(
            key=lambda x: (
                -x["mapped_count_per_boat"],
                -x.get("target_date_rows", 0),
                x["table"]
            )
        )

        mappings[group] = candidates

        print(f"\n=== {group.upper()} CANDIDATES ===")

        if not candidates:
            print("No direct column-name match")
            continue

        for item in candidates[:8]:
            print(
                f"{Path(item['db']).name} / "
                f"{item['table']}: "
                f"{item['mapped_count_per_boat']}"
                f"/{item['total_per_boat']} fields, "
                f"target rows="
                f"{item.get('target_date_rows', '?')}"
            )

            print(
                "  Example mapping:",
                list(item["mapping"].items())[:4]
            )

    print("\n=== PYTHON SOURCE DISCOVERY ===")

    scripts = analyze_python_sources(groups)

    print("Scripts containing feature references:", len(scripts))

    for group in GROUPS:
        ranked = []

        for script in scripts:
            hits = script["hits"].get(group, [])

            if hits:
                ranked.append((
                    len(hits),
                    Path(script["path"]).name,
                    script["path"]
                ))

        ranked.sort(reverse=True)

        print(f"\n{group.upper()} source candidates:")

        for count, filename, path in ranked[:8]:
            print(
                f"  {filename}: "
                f"{count} matched lines"
            )

    # 学習DBの当該日付の列がどれほど
    # 埋まっているかも集計する。
    print("\n=== TRAINING REFERENCE COVERAGE ===")

    conn = readonly(TRAIN_DB)

    reference = {}

    for group in GROUPS:
        columns = [
            item["full_name"]
            for item in groups[group]
        ]

        expressions = [
            f"SUM(CASE WHEN {quote(col)} IS NULL "
            "THEN 1 ELSE 0 END)"
            for col in columns
        ]

        result = conn.execute(
            "SELECT COUNT(*), "
            + ", ".join(expressions)
            + " FROM train_races WHERE race_date = ?",
            (TARGET,)
        ).fetchone()

        races = result[0]
        nulls = sum(
            int(value or 0)
            for value in result[1:]
        )

        total = races * len(columns)

        reference[group] = {
            "races": races,
            "columns": len(columns),
            "cells": total,
            "nulls": nulls,
            "non_null_rate": (
                (total - nulls) / total
                if total else None
            )
        }

        print(
            f"{group}: races={races}, "
            f"columns={len(columns)}, "
            f"nulls={nulls}, "
            f"coverage="
            f"{100 * (total - nulls) / total:.2f}%"
            if total else f"{group}: no reference races"
        )

    conn.close()

    output = {
        "target_date": TARGET,
        "model_feature_count": 618,
        "groups": {
            group: {
                "feature_names": [
                    item["full_name"]
                    for item in groups[group]
                ],
                "source_tables": mappings[group],
                "reference_coverage": reference[group],
            }
            for group in GROUPS
        },
        "source_scripts": scripts,
        "notes": [
            "Column-name matches are candidates, not verified feature implementations.",
            "Historical feature rows cannot automatically be used for live predictions.",
            "Source availability before race start must be verified.",
            "Never use the target race's result as a prediction input."
        ]
    }

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 調査レポートのみ更新可能。
    # 学習DBと学習済みモデルには書き込まない。
    with OUT.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n=== PHASE 8.6-D SOURCE MAPPING COMPLETE ===")
    print("Report:", OUT)


if __name__ == "__main__":
    main()
