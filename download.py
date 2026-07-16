#!/usr/bin/env python
"""Download all Poker44 benchmark releases (one raw JSON file per sourceDate)."""
import json
import os
import sys
import time

import requests

BASE = "https://api.poker44.net/api/v1"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def get(url, params=None, retries=4):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def download_date(source_date, split=None):
    """Paginate through chunks for one date. Returns list of chunk records."""
    records = []
    cursor = None
    while True:
        params = {"sourceDate": source_date, "limit": 48}
        if split:
            params["split"] = split
        if cursor:
            params["cursor"] = cursor
        resp = get(f"{BASE}/benchmark/chunks", params)
        data = resp.get("data", {})
        chunks = data.get("chunks", [])
        if not chunks:
            break
        records.extend(chunks)
        cursor = data.get("nextCursor")
        if not cursor:
            break
    return records


def main():
    info = get(f"{BASE}/benchmark")["data"]
    print("latestSourceDate:", info.get("latestSourceDate"))
    rel = get(f"{BASE}/benchmark/releases", {"limit": 30})["data"]["releases"]
    dates = sorted({r["sourceDate"] for r in rel})
    print(f"{len(dates)} release dates: {dates[0]} .. {dates[-1]}")
    for d in dates:
        path = os.path.join(OUT_DIR, f"raw_{d}.json")
        if os.path.exists(path):
            print(f"{d}: already downloaded, skipping")
            continue
        recs = download_date(d)
        n_sub = sum(len(r.get("chunks", [])) for r in recs)
        n_hands = sum(r.get("handCount", 0) for r in recs)
        with open(path, "w") as f:
            json.dump(recs, f)
        print(f"{d}: {len(recs)} chunk records, {n_sub} sub-chunks (examples), {n_hands} hands")
    return dates


if __name__ == "__main__":
    main()
