#!/usr/bin/env python3
"""
RugbyPass Batch Scraper
=======================
Reads all unique players from the Six Nations fantasy CSVs, resolves each to a
RugbyPass URL slug, scrapes their stats in a single browser session, and saves:

  • rugbypass_{slug}.json        — individual player file (same format as rugbypass.py)
  • rugbypass_stats_6n.csv       — Six Nations rows from every player's comp_stats
  • rugbypass_unknown_players.txt — players whose slug could not be resolved

Usage:
    python rugbypass_batch.py
    python rugbypass_batch.py --years 2025 2026
    python rugbypass_batch.py --dry-run
    python rugbypass_batch.py --debug      # visible browser
    python rugbypass_batch.py --delay 5    # seconds between pages
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Player name → RugbyPass slug mapping
# Key format: "Initial. Surname" exactly as it appears in the CSV
# ---------------------------------------------------------------------------
NAME_TO_SLUG: dict[str, str] = {

    # ── France ──────────────────────────────────────────────────────────────
    "A. Dupont":           "antoine-dupont",
    "T. Ramos":            "thomas-ramos",
    "M. Jalibert":         "matthieu-jalibert",
    "L. Bielle-Biarrey":   "louis-bielle-biarrey",
    "E. Gailleton":        "emilien-gailleton",
    "G. Alldritt":         "gregory-alldritt",
    "Y. Moefana":          "yoram-moefana",
    "G. Fickou":           "gael-fickou",
    "D. Penaud":           "damian-penaud",
    "R. Ntamack":          "romain-ntamack",
    "F. Cros":             "francois-cros",
    "N. Depoortere":       "nicolas-depoortere",
    "E. Meafou":           "emmanuel-meafou",
    "T. Flament":          "thibaud-flament",
    "C. Ollivon":          "charles-ollivon",
    "J. Marchand":         "julien-marchand",
    "P. Mauvaka":          "peato-mauvaka",
    "T. Attissogbe":       "theo-attissogbe",
    "M. Guillard":         "mickael-guillard",
    "A. Jelonch":          "anthony-jelonch",
    "O. Jegou":            "oscar-jegou",
    "P. Barassi":          "pierre-louis-barassi",
    "U. Atonio":           "uini-atonio",
    "C. Baille":           "cyril-baille",
    "B. Serin":            "baptiste-serin",
    "N. Le Garrec":        "nolann-le-garrec",
    "N. le Garrec":        "nolann-le-garrec",
    "M. Lucu":             "maxime-lucu",
    "F. Brau-Boirie":      "florian-brau-boirie",
    "F. Brau Boirie":      "florian-brau-boirie",
    "G. Villiere":         "gabin-villiere",
    "C. Woki":             "cameron-woki",
    "P. Bourgarit":        "pierre-bourgarit",
    "D. Aldegheri":        "dino-aldegheri",
    "P. Boudehent":        "pierre-boudehent",
    "G. Drean":            "gael-drean",
    "A. Roumat":           "alexandre-roumat",
    "J. Gros":             "jean-baptiste-gros",
    "L. Nouchi":           "lenni-nouchi",
    "L. Barre":            "leo-barre",
    "R. Taofifenua":       "romain-taofifenua",
    "H. Auradou":          "hugo-auradou",
    "J. Brennan":          "joshua-brennan",
    "D. Cretin":           "dylan-cretin",
    "R. Neti":             "rodrigue-neti",
    "K. Gourgues":         "kalvin-gourgues",
    "G. Arfeuil":          "gregoire-arfeuil",
    "T. Matiu":            "temo-matiu",
    "G. Colombe":          "georges-henri-colombe",
    "H. Reus":             "hugo-reus",
    "T. Staniforth":       "thomas-staniforth",
    "D. Bamba":            "demba-bamba",
    "T. Laclayat":         "thomas-laclayat",
    "D. Priso":            "dany-priso",
    "M. Gazzotti":         "marko-gazzotti",
    "T. Millet":           "theo-millet",
    "R. Montagne":         "regis-montagne",
    "M. Domon":            "marius-domon",
    "R. Wardi":            "reda-wardi",
    "U. Seunes":           "ugo-seunes",
    "M. Lamothe":          "maxime-lamothe",

    # ── Ireland ─────────────────────────────────────────────────────────────
    "J. Gibson-Park":      "jamison-gibson-park",
    "J. Conan":            "jack-conan",
    "J. van der Flier":    "josh-van-der-flier",
    "C. Doris":            "caelan-doris",
    "T. Beirne":           "tadgh-beirne",
    "S. McCloskey":        "stuart-mccloskey",
    "D. Sheehan":          "dan-sheehan",
    "P. O'Mahony":         "peter-omahony",
    "R. Henshaw":          "robbie-henshaw",
    "B. Aki":              "bundee-aki",
    "H. Keenan":           "hugo-keenan",
    "J. Lowe":             "james-lowe",
    "G. Ringrose":         "garry-ringrose",
    "J. Ryan":             "james-ryan",
    "R. Baloucoune":       "robert-baloucoune",
    "J. Osborne":          "jamie-osborne",
    "S. Prendergast":      "sam-prendergast",
    "C. Prendergast":      "cian-prendergast",
    "J. Stockdale":        "jacob-stockdale",
    "T. Furlong":          "tadhg-furlong",
    "A. Porter":           "andrew-porter",
    "R. Kelleher":         "ronan-kelleher",
    "J. Crowley":          "jack-crowley",
    "J. McCarthy":         "joe-mccarthy",
    "C. Murray":           "conor-murray",
    "I. Henderson":        "iain-henderson",
    "C. Healy":            "cian-healy",
    "N. Timoney":          "nick-timoney",
    "R. Baird":            "ryan-baird",
    "C. Frawley":          "ciaran-frawley",
    "M. Hansen":           "mackenzie-hansen",
    "G. Coombes":          "gavin-coombes",
    "F. Bealham":          "finlay-bealham",
    "J. Boyle":            "jack-boyle",
    "G. McCarthy":         "gus-mccarthy",
    "C. Casey":            "craig-casey",
    "J. Loughman":         "jeremy-loughman",
    "T. O'Toole":          "tom-otoole",
    "T. Clarkson":         "tom-clarkson",
    "H. Byrne":            "harry-byrne",
    "C. Blade":            "caolin-blade",
    "T. Farrell":          "thomas-farrell",
    "T. Stewart":          "tom-stewart",
    "N. Doak":             "nathan-doak",
    "J. Postlethwaite":    "jude-postlethwaite",
    "E. Edogbo":           "edwin-edogbo",
    "C. Izuchukwu":        "cormac-izuchukwu",
    "M. Milne":            "michael-milne",
    "T. O'Brien":          "tom-obrien",
    "J. O'Brien":          "jimmy-obrien",
    "R. Herring":          "rory-herring",
    "M. Gallagher":        "michael-gallagher",
    "B. Ward":             "bryn-ward",
    "T. Stewart":          "tom-stewart",
    "I. Henderson":        "iain-henderson",
    "J. Stockdale":        "jacob-stockdale",
    "H. Byrne":            "harry-byrne",

    # ── England ─────────────────────────────────────────────────────────────
    "G. Ford":             "george-ford",
    "B. Earl":             "ben-earl",
    "O. Lawrence":         "ollie-lawrence",
    "T. Freeman":          "tommy-freeman",
    "O. Chessum":          "oliver-chessum",
    "H. Pollock":          "henry-pollock",
    "M. Smith":            "marcus-smith",
    "M. Itoje":            "maro-itoje",
    "T. Curry":            "tom-curry",
    "B. Curry":            "ben-curry",
    "E. Genge":            "ellis-genge",
    "J. George":           "jamie-george",
    "A. Mitchell":         "alex-mitchell",
    "F. Steward":          "freddie-steward",
    "H. Arundell":         "henry-arundell",
    "A. Coles":            "alex-coles",
    "L. Cowan-Dickie":     "luke-cowan-dickie",
    "J. van Poortvliet":   "jack-van-poortvliet",
    "S. Underhill":        "sam-underhill",
    "E. Daly":             "elliot-daly",
    "H. Slade":            "henry-slade",
    "W. Stuart":           "will-stuart",
    "C. Cunningham-South": "chandler-cunningham-south",
    "T. Willis":           "ted-willis",
    "F. Dingwall":         "fraser-dingwall",
    "J. Heyes":            "joe-heyes",
    "T. Roebuck":          "tommy-roebuck",
    "G. Martin":           "george-martin",
    "O. Sleightholme":     "ollie-sleightholme",
    "F. Smith":            "finlay-smith",
    "G. Furbank":          "george-furbank",
    "H. Randall":          "harry-randall",
    "G. Pepper":           "guy-pepper",
    "O. Beard":            "oscar-beard",
    "C. Murley":           "cadan-murley",
    "B. Spencer":          "ben-spencer",
    "R. Quirke":           "rafi-quirke",
    "M. Ojomoh":           "manny-ojomoh",
    "E. Iyogun":           "emmanuel-iyogun",
    "G. Kloska":           "george-kloska",
    "T. Davison":          "trevor-davison",
    "S. Atkinson":         "sebastien-atkinson",
    "J. Kenningham":       "jack-kenningham",
    "G. Fisilau":          "greg-fisilau",
    "B. Rodd":             "bevan-rodd",
    "T. Dan":              "theo-dan",
    "F. Baxter":           "fin-baxter",
    "W. Rowlands":         "will-rowlands",
    "T. Hill":             "tom-hill",
    "B. Muncaster":        "ben-muncaster",
    "D. Edwards":          "daniel-edwards",
    "G. Kloska":           "george-kloska",
    "A. Clark":            "arthur-clark",
    "T. Staniforth":       "thomas-staniforth",
    "J. Roberts":          "joe-roberts",
    "G. Furbank":          "george-furbank",

    # ── Scotland ────────────────────────────────────────────────────────────
    "F. Russell":          "finn-russell",
    "R. Darge":            "rory-darge",
    "K. Steyn":            "kyle-steyn",
    "J. Dempsey":          "jack-dempsey",
    "H. Jones":            "huw-jones",
    "B. White":            "ben-white",
    "B. Kinghorn":         "blair-kinghorn",
    "M. Fagerson":         "matt-fagerson",
    "Z. Fagerson":         "zander-fagerson",
    "D. van der Merwe":    "duhan-van-der-merwe",
    "D. Graham":           "darcy-graham",
    "J. Gray":             "jonny-gray",
    "E. Ashman":           "ewan-ashman",
    "P. Schoeman":         "pierre-schoeman",
    "D. Cherry":           "david-cherry",
    "G. Gilchrist":        "grant-gilchrist",
    "G. Horne":            "george-horne",
    "J. Ritchie":          "jamie-ritchie",
    "S. McDowall":         "stafford-mcdowall",
    "S. Tuipulotu":        "sione-tuipulotu",
    "J. Bayliss":          "josh-bayliss",
    "R. Sutherland":       "rory-sutherland",
    "T. Jordan":           "tom-jordan",
    "J. Dobie":            "jamie-dobie",
    "K. Rowe":             "kyle-rowe",
    "M. Bradbury":         "magnus-bradbury",
    "A. Hastings":         "adam-hastings",
    "F. Burke":            "fergus-burke",
    "S. Skinner":          "sam-skinner",
    "J. Bhatti":           "jamie-bhattie",
    "W. Hurd":             "will-hurd",
    "N. McBeth":           "nathan-mcbeth",
    "G. Brown":            "gregor-brown",
    "E. Millar-Mills":     "elliot-millar-mills",
    "M. Williamson":       "max-williamson",
    "S. Cummings":         "scott-cummings",
    "D. Rae":              "darcy-rae",
    "A. Craig":            "alex-craig",
    "F. Douglas":          "freddie-douglas",
    "G. Warr":             "gus-warr",
    "G. Hiddleston":       "gregor-hiddleston",
    "O. Smith":            "ollie-smith",
    "R. Hutchinson":       "rory-hutchinson",
    "G. Turner":           "george-turner",
    "M. Walker":           "murphy-walker",
    "S. Stephen":          "seb-stephen",
    "G. Warr":             "gus-warr",
    "J. Mann":             "jack-mann",
    "P. Harrison":         "patrick-harrison",
    "A. Masibaka":         "alex-masibaka",
    "M. Sykes":            "marshall-sykes",
    "J. Dobie":            "jamie-dobie",
    "J. Gray":             "jonny-gray",

    # ── Wales ───────────────────────────────────────────────────────────────
    "E. Mee":              "ellis-mee",
    "J. Evans":            "jarrod-evans",
    "T. Faletau":          "taulupe-faletau",
    "B. Murray":           "blair-murray-1",
    "J. Morgan":           "jac-morgan",
    "G. Anscombe":         "gareth-anscombe",
    "B. Thomas":           "ben-thomas",
    "T. Reffell":          "tommy-reffell",
    "T. Rogers":           "tom-rogers",
    "W. John":             "will-john",
    "D. Edwards":          "daniel-edwards",
    "G. Thomas":           "gareth-thomas",
    "N. Tompkins":         "nick-tompkins",
    "J. Roberts":          "joe-roberts",
    "K. Assiratti":        "kieron-assiratti",
    "E. Dee":              "elliott-dee",
    "R. Williams":         "rhodri-williams",
    "C. Tshiunza":         "christ-tshiunza",
    "E. Bevan":            "ellis-bevan",
    "R. Carre":            "rhys-carre",
    "A. Mann":             "alex-mann",
    "S. Costelow":         "sam-costelow",
    "O. Cracknell":        "olly-cracknell",
    "A. Beard":            "adam-beard",
    "M. Grady":            "mason-grady",
    "J. Hawkins":          "joe-hawkins",
    "H. Deaves":           "harri-deaves",
    "B. Carter":           "ben-carter",
    "R. Elias":            "ryan-elias",
    "T. Francis":          "tomas-francis",
    "O. Watkin":           "owen-watkin",
    "K. Hardy":            "kieran-hardy",
    "R. Morgan-Williams":  "reuben-morgan-williams",
    "L. Hennessey":        "louie-hennessey",
    "G. Hamer-Webb":       "gabriel-hamer-webb",
    "L. Belcher":          "liam-belcher",
    "R. Woodman":          "ryan-woodman",
    "S. Wainwright":       "sam-wainwright",
    "T. Plumtree":         "taine-plumtree",
    "H. Thomas":           "harry-thomas",
    "F. Thomas":           "freddie-thomas",
    "W. Rowlands":         "will-rowlands",
    "J. Macleod":          "josh-macleod",
    "E. Lloyd":            "evan-lloyd",
    "J. Hathaway":         "josh-hathaway",
    "A. Griffin":          "archie-griffin",
    "D. Lamb":             "dino-lamb",
    "M. Llewellyn":        "max-llewellyn",
    "C. Nash":             "calvin-nash",
    "E. Johnson":          "ewan-johnson",
    "M. Currie":           "matt-currie",

    # ── Italy ───────────────────────────────────────────────────────────────
    "L. Cannone":          "lorenzo-cannone",
    "M. Zuliani":          "manuelzuliani",
    "T. Menoncello":       "tommaso-menoncello",
    "A. Capuozzo":         "ange-capuozzo",
    "M. Lamaro":           "michele-lamaro",
    "P. Garbisi":          "paolo-garbisi",
    "A. Garbisi":          "alessandro-garbisi",
    "N. Cannone":          "niccolo-cannone",
    "M. Ioane":            "monty-ioane",
    "G. Nicotera":         "giacomo-nicotera",
    "D. Fischetti":        "danilo-fischetti",
    "S. Ferrari":          "simone-ferrari",
    "T. Allan":            "tommaso-allan",
    "F. Ruzza":            "federico-ruzza",
    "R. Vintcent":         "ross-vintcent",
    "M. Page-Relo":        "martin-page-relo",
    "S. Varney":           "stephen-varney",
    "J. Brex":             "juan-ignacio-brex",
    "N. Brex":             "juan-ignacio-brex",
    "P. Odogwu":           "paolo-odogwu",
    "L. Lynagh":           "louis-lynagh",
    "L. Pani":             "lorenzo-pani",
    "G. Lucchesi":         "gianmarcolucchesi",
    "A. Fusco":            "alessandro-fusco",
    "G. Zilocchi":         "giosue-zilocchi",
    "M. Spagnolo":         "mirco-spagnolo",
    "M. Riccioni":         "marco-riccioni",
    "R. Favretto":         "riccardo-favretto",
    "G. Bertaccini":       "giulio-bertaccini",
    "D. Mazza":            "damiano-mazza",
    "S. Locatelli":        "samuele-locatelli",
    "A. Izekor":           "alessandro-izekor",
    "G. da Re":            "giacomo-da-re",
    "P. Dimcheff":         "pablo-dimcheff",
    "T. di Bartolomeo":    "tommaso-di-bartolomeo",
    "M. Hasa":             "muhamed-hasa",
    "L. Marin":            "leonardo-marin",
    "A. Zambonin":         "andrea-zambonin",
    "S. Negri":            "sebastian-negri-da-ollegio",
    "M. Zanon":            "marco-zanon",
    "S. Gesi":             "simone-gesi",
    "J. Trulla":           "jacopo-trulla",
    "G. da Re":            "giacomo-da-re",
    "D. Odiase":           "david-odiase",
    "M. Gallagher":        "matt-gallagher",
    "S. Locatelli":        "samuele-locatelli",
    "G. Bertaccini":       "giulio-bertaccini",
    "D. Mazza":            "damiano-mazza",
}

SIX_NATIONS_KEYWORDS = {"six nations", "six-nations", "6 nations"}


# ---------------------------------------------------------------------------
# Helpers (mirror rugbypass.py without importing it to keep this self-contained)
# ---------------------------------------------------------------------------

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
        service=Service(ChromeDriverManager().install()), options=options
    )
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    return driver


def extract_bio(soup):
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
    div = soup.find("div", id="app-comp-stats")
    if not div:
        return []
    raw = json.loads(div.get_text()) or []
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
    div = soup.find("div", id="app-competitions")
    if not div:
        return []
    raw = json.loads(div.get_text()) or []
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


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def get_unique_players(csv_paths: list[Path]) -> list[str]:
    names: set[str] = set()
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            if "Player" in df.columns:
                names.update(df["Player"].dropna().unique())
        except Exception as e:
            print(f"  ⚠️  Could not read {path}: {e}")
    return sorted(names)


def is_six_nations(comp_name: str | None) -> bool:
    if not comp_name:
        return False
    return any(kw in comp_name.lower() for kw in SIX_NATIONS_KEYWORDS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch-scrape RugbyPass for Six Nations players")
    parser.add_argument("--years", nargs="+", default=["2025", "2026"],
                        help="Which season folders to read CSVs from (default: 2025 2026)")
    parser.add_argument("--debug", action="store_true", help="Show browser window")
    parser.add_argument("--dry-run", action="store_true",
                        help="List players + slugs without scraping")
    parser.add_argument("--delay", type=float, default=8.0,
                        help="Seconds to wait after each page load (default: 8)")
    parser.add_argument("--force", action="store_true",
                        help="Re-scrape even if JSON already exists")
    args = parser.parse_args()

    base = Path(__file__).parent
    out_dir = base / "players"          # per-player JSONs live here
    out_dir.mkdir(exist_ok=True)

    # ── Collect CSV paths ───────────────────────────────────────────────────
    csv_paths: list[Path] = []
    for year in args.years:
        p = base / year / "players.csv"
        if p.exists():
            csv_paths.append(p)
            print(f"  📄  Found {p}")
        else:
            print(f"  ⚠️  Not found: {p}")

    if not csv_paths:
        print("No CSVs found — exiting.")
        sys.exit(1)

    # ── Build player list ───────────────────────────────────────────────────
    all_names = get_unique_players(csv_paths)
    print(f"\n  {len(all_names)} unique player names found across CSVs.\n")

    known: list[tuple[str, str]] = []
    unknown: list[str] = []

    for name in all_names:
        slug = NAME_TO_SLUG.get(name)
        if slug:
            known.append((name, slug))
        else:
            unknown.append(name)

    print(f"  ✅  {len(known)} mapped   ❓  {len(unknown)} unknown\n")

    # Save unknown list
    unknown_path = base / "rugbypass_unknown_players.txt"
    unknown_path.write_text("\n".join(unknown) + "\n")
    if unknown:
        print(f"  Unknown players written to {unknown_path.name}")
        for n in unknown:
            print(f"      {n}")
        print()

    if args.dry_run:
        print("── Dry run — would scrape ──")
        for name, slug in known:
            json_path = out_dir / f"rugbypass_{slug.replace('-', '_')}.json"
            status = "EXISTS" if json_path.exists() else "new"
            print(f"  {name:35s}  {slug:45s}  [{status}]")
        return

    # ── Determine which players still need scraping ─────────────────────────
    to_scrape: list[tuple[str, str]] = []
    for name, slug in known:
        stem = slug.replace("-", "_")
        json_path = out_dir / f"rugbypass_{stem}.json"
        if json_path.exists() and not args.force:
            print(f"  ⏭   Skipping {name} (already have {json_path.name})")
        else:
            to_scrape.append((name, slug))

    if not to_scrape:
        print("\n  All players already scraped. Use --force to re-scrape.")
        return

    print(f"\n  🏉  Scraping {len(to_scrape)} players...\n")

    # ── Single browser session ──────────────────────────────────────────────
    driver = make_driver(debug=args.debug)
    six_nations_rows: list[dict] = []
    failed: list[str] = []

    try:
        for i, (name, slug) in enumerate(to_scrape, 1):
            url = f"https://www.rugbypass.com/players/{slug}/"
            stem = slug.replace("-", "_")
            json_path = out_dir / f"rugbypass_{stem}.json"

            print(f"  [{i:>3}/{len(to_scrape)}]  {name}  →  {url}")

            try:
                driver.get(url)
                time.sleep(args.delay)

                title = driver.title
                if "404" in title or "not found" in title.lower():
                    print(f"           ⚠️  404 — slug may be wrong for '{name}'")
                    failed.append(f"{name} ({slug})")
                    continue

                html = driver.page_source
                soup = BeautifulSoup(html, "html.parser")

                bio = extract_bio(soup)
                comp_stats = extract_comp_stats(soup)
                match_log = extract_match_log(soup)

                # Save individual JSON
                out = {
                    "player": slug,
                    "csv_name": name,
                    "bio": bio,
                    "competition_stats": comp_stats,
                    "match_log": match_log,
                }
                json_path.write_text(json.dumps(out, indent=2))

                # Collect Six Nations rows for combined CSV
                for row in comp_stats:
                    if is_six_nations(row.get("competition")):
                        six_nations_rows.append({"player": name, "slug": slug, **row})

                sn_count = sum(1 for r in comp_stats if is_six_nations(r.get("competition")))
                print(f"           ✅  bio={bool(bio)}  "
                      f"comp_rows={len(comp_stats)}  six_nations_rows={sn_count}  "
                      f"saved→{json_path.name}")

            except Exception as e:
                print(f"           ❌  Error: {e}")
                failed.append(f"{name} ({slug})")

    finally:
        driver.quit()

    # ── Write combined Six Nations CSV ──────────────────────────────────────
    if six_nations_rows:
        out_csv = base / "rugbypass_stats_6n.csv"
        pd.DataFrame(six_nations_rows).to_csv(out_csv, index=False)
        print(f"\n  💾  Six Nations stats  → {out_csv}  ({len(six_nations_rows)} rows)")
    else:
        print("\n  ⚠️  No Six Nations competition rows found in scraped data.")

    # ── Also rebuild from existing JSONs (catches previously scraped players) ─
    all_6n: list[dict] = []
    for json_file in sorted(out_dir.glob("rugbypass_*.json")):
        if json_file.name in ("rugbypass_stats_6n.json",):
            continue
        try:
            data = json.loads(json_file.read_text())
            pname = data.get("csv_name") or data.get("player", "")
            for row in data.get("competition_stats", []):
                if is_six_nations(row.get("competition")):
                    all_6n.append({"player": pname, "slug": data.get("player", ""), **row})
        except Exception:
            pass

    if all_6n:
        out_all = base / "rugbypass_stats_6n_all.csv"
        pd.DataFrame(all_6n).to_csv(out_all, index=False)
        print(f"  💾  All Six Nations stats (incl. existing JSONs) → {out_all}  ({len(all_6n)} rows)")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n  Done.  Scraped: {len(to_scrape) - len(failed)}  Failed: {len(failed)}")
    if failed:
        print("\n  Failed players (slug may need correcting in NAME_TO_SLUG):")
        for f in failed:
            print(f"    {f}")


if __name__ == "__main__":
    main()
