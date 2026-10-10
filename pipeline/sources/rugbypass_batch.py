from __future__ import annotations

import json

NAME_TO_SLUG = {
    "Hame Faiva": "epalahame-faiva", "Oli Kebble": "oliver-kebble", "Noah Nene": "noah-tisie-nene", "A. Dupont": "antoine-dupont", "T. Ramos": "thomas-ramos",
    "M. Jalibert": "matthieu-jalibert", "L. Bielle-Biarrey": "louis-bielle-biarrey", "E. Gailleton": "emilien-gailleton", "G. Alldritt": "gregory-alldritt",
    "Y. Moefana": "yoram-moefana", "G. Fickou": "gael-fickou", "D. Penaud": "damian-penaud", "R. Ntamack": "romain-ntamack", "F. Cros": "francois-cros",
    "N. Depoortere": "nicolas-depoortere", "E. Meafou": "emmanuel-meafou", "T. Flament": "thibaud-flament", "C. Ollivon": "charles-ollivon",
    "J. Marchand": "julien-marchand", "P. Mauvaka": "peato-mauvaka", "T. Attissogbe": "theo-attissogbe", "M. Guillard": "mickael-guillard",
    "A. Jelonch": "anthony-jelonch", "O. Jegou": "oscar-jegou", "P. Barassi": "pierre-louis-barassi", "U. Atonio": "uini-atonio", "C. Baille": "cyril-baille",
    "B. Serin": "baptiste-serin", "N. Le Garrec": "nolann-le-garrec", "N. le Garrec": "nolann-le-garrec", "M. Lucu": "maxime-lucu",
    "F. Brau-Boirie": "florian-brau-boirie", "F. Brau Boirie": "florian-brau-boirie", "G. Villiere": "gabin-villiere", "C. Woki": "cameron-woki",
    "P. Bourgarit": "pierre-bourgarit", "D. Aldegheri": "dino-aldegheri", "P. Boudehent": "pierre-boudehent", "G. Drean": "gael-drean",
    "A. Roumat": "alexandre-roumat", "J. Gros": "jean-baptiste-gros", "L. Nouchi": "lenni-nouchi", "L. Barre": "leo-barre",
    "R. Taofifenua": "romain-taofifenua", "H. Auradou": "hugo-auradou", "J. Brennan": "joshua-brennan", "D. Cretin": "dylan-cretin", "R. Neti": "rodrigue-neti",
    "K. Gourgues": "kalvin-gourgues", "G. Arfeuil": "gregoire-arfeuil", "T. Matiu": "temo-matiu", "G. Colombe": "georges-henri-colombe", "H. Reus": "hugo-reus",
    "T. Staniforth": "thomas-staniforth", "D. Bamba": "demba-bamba", "T. Laclayat": "thomas-laclayat", "D. Priso": "dany-priso",
    "M. Gazzotti": "marko-gazzotti", "T. Millet": "theo-millet", "R. Montagne": "regis-montagne", "M. Domon": "marius-domon", "R. Wardi": "reda-wardi",
    "U. Seunes": "ugo-seunes", "M. Lamothe": "maxime-lamothe", "J. Gibson-Park": "jamison-gibson-park", "J. Conan": "jack-conan",
    "J. van der Flier": "josh-van-der-flier", "C. Doris": "caelan-doris", "T. Beirne": "tadgh-beirne", "S. McCloskey": "stuart-mccloskey",
    "D. Sheehan": "dan-sheehan", "P. O'Mahony": "peter-omahony", "R. Henshaw": "robbie-henshaw", "B. Aki": "bundee-aki", "H. Keenan": "hugo-keenan",
    "J. Lowe": "james-lowe", "G. Ringrose": "garry-ringrose", "J. Ryan": "james-ryan", "R. Baloucoune": "robert-baloucoune", "J. Osborne": "jamie-osborne",
    "S. Prendergast": "sam-prendergast", "C. Prendergast": "cian-prendergast", "J. Stockdale": "jacob-stockdale", "T. Furlong": "tadhg-furlong",
    "A. Porter": "andrew-porter", "R. Kelleher": "ronan-kelleher", "J. Crowley": "jack-crowley", "J. McCarthy": "joe-mccarthy", "C. Murray": "conor-murray",
    "I. Henderson": "iain-henderson", "C. Healy": "cian-healy", "N. Timoney": "nick-timoney", "R. Baird": "ryan-baird", "C. Frawley": "ciaran-frawley",
    "M. Hansen": "mackenzie-hansen", "G. Coombes": "gavin-coombes", "F. Bealham": "finlay-bealham", "J. Boyle": "jack-boyle", "G. McCarthy": "gus-mccarthy",
    "C. Casey": "craig-casey", "J. Loughman": "jeremy-loughman", "T. O'Toole": "tom-otoole", "T. Clarkson": "tom-clarkson", "H. Byrne": "harry-byrne",
    "C. Blade": "caolin-blade", "T. Farrell": "thomas-farrell", "T. Stewart": "tom-stewart", "N. Doak": "nathan-doak", "J. Postlethwaite": "jude-postlethwaite",
    "E. Edogbo": "edwin-edogbo", "C. Izuchukwu": "cormac-izuchukwu", "M. Milne": "michael-milne", "T. O'Brien": "tom-obrien", "J. O'Brien": "jimmy-obrien",
    "R. Herring": "rory-herring", "M. Gallagher": "matt-gallagher", "B. Ward": "bryn-ward", "G. Ford": "george-ford", "B. Earl": "ben-earl",
    "O. Lawrence": "ollie-lawrence", "T. Freeman": "tommy-freeman", "O. Chessum": "oliver-chessum", "H. Pollock": "henry-pollock", "M. Smith": "marcus-smith",
    "M. Itoje": "maro-itoje", "T. Curry": "tom-curry", "B. Curry": "ben-curry", "E. Genge": "ellis-genge", "J. George": "jamie-george",
    "A. Mitchell": "alex-mitchell", "F. Steward": "freddie-steward", "H. Arundell": "henry-arundell", "A. Coles": "alex-coles",
    "L. Cowan-Dickie": "luke-cowan-dickie", "J. van Poortvliet": "jack-van-poortvliet", "S. Underhill": "sam-underhill", "E. Daly": "elliot-daly",
    "H. Slade": "henry-slade", "W. Stuart": "will-stuart", "C. Cunningham-South": "chandler-cunningham-south", "T. Willis": "ted-willis",
    "F. Dingwall": "fraser-dingwall", "J. Heyes": "joe-heyes", "T. Roebuck": "tommy-roebuck", "G. Martin": "george-martin",
    "O. Sleightholme": "ollie-sleightholme", "F. Smith": "finlay-smith", "G. Furbank": "george-furbank", "H. Randall": "harry-randall",
    "G. Pepper": "guy-pepper", "O. Beard": "oscar-beard", "C. Murley": "cadan-murley", "B. Spencer": "ben-spencer", "R. Quirke": "rafi-quirke",
    "M. Ojomoh": "manny-ojomoh", "E. Iyogun": "emmanuel-iyogun", "G. Kloska": "george-kloska", "T. Davison": "trevor-davison",
    "S. Atkinson": "sebastien-atkinson", "J. Kenningham": "jack-kenningham", "G. Fisilau": "greg-fisilau", "B. Rodd": "bevan-rodd", "T. Dan": "theo-dan",
    "F. Baxter": "fin-baxter", "W. Rowlands": "will-rowlands", "T. Hill": "tom-hill", "B. Muncaster": "ben-muncaster", "D. Edwards": "daniel-edwards",
    "A. Clark": "arthur-clark", "J. Roberts": "joe-roberts", "F. Russell": "finn-russell", "R. Darge": "rory-darge", "K. Steyn": "kyle-steyn",
    "J. Dempsey": "jack-dempsey", "H. Jones": "huw-jones", "B. White": "ben-white", "B. Kinghorn": "blair-kinghorn", "M. Fagerson": "matt-fagerson",
    "Z. Fagerson": "zander-fagerson", "D. van der Merwe": "duhan-van-der-merwe", "D. Graham": "darcy-graham", "J. Gray": "jonny-gray",
    "E. Ashman": "ewan-ashman", "P. Schoeman": "pierre-schoeman", "D. Cherry": "david-cherry", "G. Gilchrist": "grant-gilchrist", "G. Horne": "george-horne",
    "J. Ritchie": "jamie-ritchie", "S. McDowall": "stafford-mcdowall", "S. Tuipulotu": "sione-tuipulotu", "J. Bayliss": "josh-bayliss",
    "R. Sutherland": "rory-sutherland", "T. Jordan": "tom-jordan", "J. Dobie": "jamie-dobie", "K. Rowe": "kyle-rowe", "M. Bradbury": "magnus-bradbury",
    "A. Hastings": "adam-hastings", "F. Burke": "fergus-burke", "S. Skinner": "sam-skinner", "J. Bhatti": "jamie-bhattie", "W. Hurd": "will-hurd",
    "N. McBeth": "nathan-mcbeth", "G. Brown": "gregor-brown", "E. Millar-Mills": "elliot-millar-mills", "M. Williamson": "max-williamson",
    "S. Cummings": "scott-cummings", "D. Rae": "darcy-rae", "A. Craig": "alex-craig", "F. Douglas": "freddie-douglas", "G. Warr": "gus-warr",
    "G. Hiddleston": "gregor-hiddleston", "O. Smith": "ollie-smith", "R. Hutchinson": "rory-hutchinson", "G. Turner": "george-turner",
    "M. Walker": "murphy-walker", "S. Stephen": "seb-stephen", "J. Mann": "jack-mann", "P. Harrison": "patrick-harrison", "A. Masibaka": "alex-masibaka",
    "M. Sykes": "marshall-sykes", "E. Mee": "ellis-mee", "J. Evans": "jarrod-evans", "T. Faletau": "taulupe-faletau", "B. Murray": "blair-murray-1",
    "J. Morgan": "jac-morgan", "G. Anscombe": "gareth-anscombe", "B. Thomas": "ben-thomas", "T. Reffell": "tommy-reffell", "T. Rogers": "tom-rogers",
    "W. John": "will-john", "G. Thomas": "gareth-thomas", "N. Tompkins": "nick-tompkins", "K. Assiratti": "kieron-assiratti", "E. Dee": "elliott-dee",
    "R. Williams": "rhodri-williams", "C. Tshiunza": "christ-tshiunza", "E. Bevan": "ellis-bevan", "R. Carre": "rhys-carre", "A. Mann": "alex-mann",
    "S. Costelow": "sam-costelow", "O. Cracknell": "olly-cracknell", "A. Beard": "adam-beard", "M. Grady": "mason-grady", "J. Hawkins": "joe-hawkins",
    "H. Deaves": "harri-deaves", "B. Carter": "ben-carter", "R. Elias": "ryan-elias", "T. Francis": "tomas-francis", "O. Watkin": "owen-watkin",
    "K. Hardy": "kieran-hardy", "R. Morgan-Williams": "reuben-morgan-williams", "L. Hennessey": "louie-hennessey", "G. Hamer-Webb": "gabriel-hamer-webb",
    "L. Belcher": "liam-belcher", "R. Woodman": "ryan-woodman", "S. Wainwright": "sam-wainwright", "T. Plumtree": "taine-plumtree", "H. Thomas": "harry-thomas",
    "F. Thomas": "freddie-thomas", "J. Macleod": "josh-macleod", "E. Lloyd": "evan-lloyd", "J. Hathaway": "josh-hathaway", "A. Griffin": "archie-griffin",
    "D. Lamb": "dino-lamb", "M. Llewellyn": "max-llewellyn", "C. Nash": "calvin-nash", "E. Johnson": "ewan-johnson", "M. Currie": "matt-currie",
    "L. Cannone": "lorenzo-cannone", "M. Zuliani": "manuelzuliani", "T. Menoncello": "tommaso-menoncello", "A. Capuozzo": "ange-capuozzo",
    "M. Lamaro": "michele-lamaro", "P. Garbisi": "paolo-garbisi", "A. Garbisi": "alessandro-garbisi", "N. Cannone": "niccolo-cannone",
    "M. Ioane": "monty-ioane", "G. Nicotera": "giacomo-nicotera", "D. Fischetti": "danilo-fischetti", "S. Ferrari": "simone-ferrari",
    "T. Allan": "tommaso-allan", "F. Ruzza": "federico-ruzza", "R. Vintcent": "ross-vintcent", "M. Page-Relo": "martin-page-relo",
    "S. Varney": "stephen-varney", "J. Brex": "juan-ignacio-brex", "N. Brex": "juan-ignacio-brex", "P. Odogwu": "paolo-odogwu", "L. Lynagh": "louis-lynagh",
    "L. Pani": "lorenzo-pani", "G. Lucchesi": "gianmarcolucchesi", "A. Fusco": "alessandro-fusco", "G. Zilocchi": "giosue-zilocchi",
    "M. Spagnolo": "mirco-spagnolo", "M. Riccioni": "marco-riccioni", "R. Favretto": "riccardo-favretto", "G. Bertaccini": "giulio-bertaccini",
    "D. Mazza": "damiano-mazza", "S. Locatelli": "samuele-locatelli", "A. Izekor": "alessandro-izekor", "G. da Re": "giacomo-da-re",
    "P. Dimcheff": "pablo-dimcheff", "T. di Bartolomeo": "tommaso-di-bartolomeo", "M. Hasa": "muhamed-hasa", "L. Marin": "leonardo-marin",
    "A. Zambonin": "andrea-zambonin", "S. Negri": "sebastian-negri-da-ollegio", "M. Zanon": "marco-zanon", "S. Gesi": "simone-gesi",
    "J. Trulla": "jacopo-trulla", "D. Odiase": "david-odiase",
}

SIX_NATIONS_KEYWORDS = {"six nations", "six-nations", "6 nations"}


def extract_bio(soup):
    bio = {}
    for detail in div.find_all("div", class_="detail") if (div := soup.find("div", class_="player-details")) else []:
        label, img, value_div = detail.find("h3"), detail.find("img"), detail.find("div")
        if label and (img or value_div):
            bio[label.get_text(strip=True)] = img.get("alt", "") if img else value_div.get_text(strip=True)
    return bio


def _app_json(soup, app_id: str) -> list:
    div = soup.find("div", id=app_id)
    return (json.loads(div.get_text()) or []) if div else []


def extract_comp_stats(soup):
    rows = []
    for block in _app_json(soup, "app-comp-stats"):
        comp = block.get("competition", {})
        s = comp.get("stats", {}).get("stats", {})
        rows.append(
            {
                "competition": comp.get("name"),
                "season": comp.get("season", {}).get("label"),
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
                **{
                    k: s.get(k, 0)
                    for k in ("dominant_tackles", "turnovers_won", "turnovers_conceded", "kicks_from_hand", "penalties_conceded", "yellow_cards", "red_cards")
                },
            }
        )
    return rows


def extract_match_log(soup):
    return [
        {
            "competition": block.get("title"),
            "match": g.get("title"),
            "date": g.get("date"),
            "opposition": g.get("opposition", {}).get("name"),
            "win": g.get("win"),
            "draw": g.get("draw"),
            "mins": g.get("stats", {}).get("mins"),
            **{k: g.get("stats", {}).get(k, 0) for k in ("tries", "yellow_cards", "red_cards", "conversions")},
        }
        for block in _app_json(soup, "app-competitions")
        for g in block.get("games", [])
    ]


def is_six_nations(comp_name: str | None) -> bool:
    return bool(comp_name) and any(kw in comp_name.lower() for kw in SIX_NATIONS_KEYWORDS)
