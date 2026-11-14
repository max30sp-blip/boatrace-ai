
import requests
from bs4 import BeautifulSoup

url = "https://www.boatrace.jp/owpc/pc/race/odds3t"

params = {
    "hd": "20261105",
    "jcd": "15",
    "rno": "8",
}

response = requests.get(
    url,
    params=params,
    timeout=20,
    headers={
        "User-Agent": "BOAT-RACE-AI-Research/1.0"
    },
)

print("HTTP:", response.status_code)
print("URL:", response.url)
print("Bytes:", len(response.content))

response.raise_for_status()
response.encoding = response.apparent_encoding

soup = BeautifulSoup(response.text, "html.parser")

print("Title:", soup.title.get_text(" ", strip=True)
      if soup.title else "なし")

tables = soup.find_all("table")
print("Tables:", len(tables))

for i, table in enumerate(tables):
    rows = table.find_all("tr")
    text = table.get_text(" ", strip=True)
    print(
        f"Table {i}: rows={len(rows)}, "
        f"sample={text[:180]}"
    )

print("Contains odds heading:",
      "3連単オッズ" in soup.get_text(" ", strip=True))
