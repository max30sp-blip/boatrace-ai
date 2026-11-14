
import argparse
import json
import requests


def inspect(obj, path="$", depth=0):
    """JSON内のオッズ関連キーと構造を調べる"""
    if depth > 8:
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            p = f"{path}.{key}"
            key_lower = str(key).lower()

            if any(word in key_lower for word in (
                "odds", "trifecta", "stadium", "race"
            )):
                print(
                    f"{p}: {type(value).__name__}",
                    f"keys={list(value)[:8]}"
                    if isinstance(value, dict)
                    else ""
                )

            if isinstance(value, (dict, list)):
                inspect(value, p, depth + 1)

    elif isinstance(obj, list):
        for i, value in enumerate(obj[:2]):
            if isinstance(value, (dict, list)):
                inspect(value, f"{path}[{i}]", depth + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--venue", required=True, type=int)
    parser.add_argument("--race", required=True, type=int)
    args = parser.parse_args()

    date = args.date.replace("-", "")
    year = date[:4]

    url = (
        "https://lamrongol.github.io/BoatraceOdds"
        f"/v3/{year}/{date}.json"
    )

    print("=== WEB-11 ODDS API DIAGNOSIS ===")
    print("URL:", url)
    print("Venue:", args.venue, "Race:", args.race)

    try:
        response = requests.get(url, timeout=30)
        print("HTTP:", response.status_code)
        print("Content-Type:", response.headers.get(
            "Content-Type"
        ))
        print("Response bytes:", len(response.content))
        response.raise_for_status()
        data = response.json()

    except Exception as exc:
        print("DOWNLOAD ERROR:", repr(exc))
        return

    print("\n=== ROOT ===")
    print("Type:", type(data).__name__)

    if isinstance(data, dict):
        print("Root keys:", list(data)[:30])

    print("\n=== JSON STRUCTURE ===")
    inspect(data)

    print("\n=== TARGET RACE CHECK ===")
    for root_name in ("programs", "odds", None):
        root = (
            data.get(root_name)
            if root_name and isinstance(data, dict)
            else data
        )

        if not isinstance(root, dict):
            continue

        stadiums = root.get("stadiums", root)

        if not isinstance(stadiums, dict):
            continue

        venue = (
            stadiums.get(str(args.venue))
            or stadiums.get(f"{args.venue:02d}")
        )

        if not isinstance(venue, dict):
            continue

        races = venue.get("races", venue)

        if not isinstance(races, dict):
            continue

        race = races.get(str(args.race))

        if isinstance(race, dict):
            print("Matched root:", root_name)
            print("Race keys:", list(race))

            for key, value in race.items():
                if "odd" in key.lower() or "trifecta" in key.lower():
                    print(
                        key,
                        json.dumps(
                            value,
                            ensure_ascii=False
                        )[:800]
                    )


if __name__ == "__main__":
    main()
