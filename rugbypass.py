#!/usr/bin/env python3
"""
RugbyPass Player Stats Scraper
==============================
Loads the player page, then reads stats directly from the two hidden data
divs the site embeds in its HTML:

  #app-comp-stats   → per-competition / per-season aggregate stats (JSON array)
  #app-competitions → per-game match log (JSON array)
  .player-details   → bio (name, age, position, height, weight)

Requirements:
    pip install selenium webdriver-manager beautifulsoup4

Usage:
    python rugbypass.py
    python rugbypass.py --player bundee-aki
    python rugbypass.py --debug    # visible browser window
"""

import argparse
import json
import re
import time
from pathlib import Path

from bs4 import BeautifulSoup

PLAYER_SLUG = "stuart-mccloskey"


def make_driver(debug=False):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    options = Options()
    if not debug:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    return driver


def extract_bio(soup):
    """Parse player bio from .player-details."""
    bio = {}
    bio_div = soup.find("div", class_="player-details")
    if not bio_div:
        return bio
    for detail in bio_div.find_all("div", class_="detail"):
        label = detail.find("h3")
        img = detail.find("img")
        value_div = detail.find("div")
        if label:
            key = label.get_text(strip=True)
            if img:
                bio[key] = img.get("alt", "")
            elif value_div:
                bio[key] = value_div.get_text(strip=True)
    return bio


def extract_comp_stats(soup):
    """
    Parse #app-comp-stats → list of per-competition season aggregates.
    Returns a cleaned list of stat dicts.
    """
    div = soup.find("div", id="app-comp-stats")
    if not div:
        return []

    raw = json.loads(div.get_text())
    rows = []
    for block in raw:
        comp = block.get("competition", {})
        s = comp.get("stats", {}).get("stats", {})
        season = comp.get("season", {})
        rows.append({
            "competition": comp.get("name"),
            "season": season.get("label"),
            "games": s.get("total_games"),
            "minutes": s.get("minutes_played_total"),
            "tries": s.get("tries", 0),
            "try_assists": s.get("try_assist", 0),
            "points": s.get("points", 0),
            "carries": s.get("carries"),
            "metres": s.get("metres"),
            "av_gain": s.get("av_gain"),
            "post_contact_metres": s.get("post_contact_metres"),
            "defenders_beaten": s.get("defenders_beaten"),
            "clean_breaks": s.get("clean_breaks", 0),
            "offloads": s.get("offloads", 0),
            "handling_errors": s.get("handling_error", 0),
            "passes": s.get("passes"),
            "bad_passes": s.get("bad_passes", 0),
            "tackles": s.get("tackles"),
            "missed_tackles": s.get("missed_tackles"),
            "tackles_success_pct": s.get("tackles_success"),
            "dominant_tackles": s.get("dominant_tackles", 0),
            "turnovers_won": s.get("turnovers_won", 0),
            "turnovers_conceded": s.get("turnovers_conceded", 0),
            "kicks_from_hand": s.get("kicks_from_hand", 0),
            "penalties_conceded": s.get("penalties_conceded", 0),
            "yellow_cards": s.get("yellow_cards", 0),
            "red_cards": s.get("red_cards", 0),
        })
    return rows


def extract_match_log(soup):
    """
    Parse #app-competitions → flat list of per-game records.
    """
    div = soup.find("div", id="app-competitions")
    if not div:
        return []

    raw = json.loads(div.get_text())
    games = []
    for comp_block in raw:
        comp_name = comp_block.get("title")
        for g in comp_block.get("games", []):
            s = g.get("stats", {})
            games.append({
                "competition": comp_name,
                "match": g.get("title"),
                "date": g.get("date"),
                "opposition": g.get("opposition", {}).get("name"),
                "win": g.get("win"),
                "draw": g.get("draw"),
                "mins": s.get("mins"),
                "tries": s.get("tries", 0),
                "yellow_cards": s.get("yellow_cards", 0),
                "red_cards": s.get("red_cards", 0),
                "conversions": s.get("conversions", 0),
            })
    return games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", default=PLAYER_SLUG)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    slug = args.player
    url = f"https://www.rugbypass.com/players/{slug}/"
    stem = re.sub(r"[^a-z0-9]+", "_", slug.lower())

    print(f"\n🏉  RugbyPass Scraper — {slug}")
    print(f"    {url}\n")

    driver = make_driver(debug=args.debug)
    try:
        print("① Loading page…")
        driver.get(url)
        time.sleep(8)

        title = driver.title
        print(f"   Title: {title}")
        if "404" in title or "not found" in title.lower():
            print("   ⚠️  Page not found — check player slug")
            return

        html = driver.page_source
    finally:
        driver.quit()

    soup = BeautifulSoup(html, "html.parser")

    print("② Extracting bio…")
    bio = extract_bio(soup)
    print(f"   {bio}")

    print("③ Extracting competition stats…")
    comp_stats = extract_comp_stats(soup)
    print(f"   {len(comp_stats)} competition/season rows")
    for r in comp_stats:
        print(f"   {r['competition']} {r['season']}: "
              f"{r['games']}g  {r['tries']}T  {r['carries']}car  "
              f"{r['metres']}m  {r['tackles']}tkl")

    print("④ Extracting match log…")
    match_log = extract_match_log(soup)
    print(f"   {len(match_log)} matches")

    # Save clean JSON
    out = {
        "player": slug,
        "bio": bio,
        "competition_stats": comp_stats,
        "match_log": match_log,
    }
    json_path = Path(f"rugbypass_{stem}.json")
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\n💾  Saved → {json_path}")


if __name__ == "__main__":
    main()