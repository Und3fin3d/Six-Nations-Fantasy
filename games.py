import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import ElementClickInterceptedException, TimeoutException, NoSuchElementException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
import traceback
import time

LOGIN_URL = "https://account.sixnationsrugby.com/en/sign-in"
STATS_URL = "https://fantasy.sixnationsrugby.com/m6n/#/game/stats"
MATCH_URL = "https://fantasy.sixnationsrugby.com/m6n/#/game/play/me"

def scrape_rugby_stats():
    chrome_options = Options()
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        driver.get(LOGIN_URL)
        print("Please log in manually within 2 minutes...")
        wait = WebDriverWait(driver, 120)
        wait.until(lambda d: "fantasy.sixnationsrugby.com" in d.current_url)
        driver.get(STATS_URL)
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".fs-table-responsive table")))
        
        all_players_data = []
        page_num = 1
        has_more_pages = True

        while has_more_pages:
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, "html.parser")

            tooltips = {}
            for div in soup.select('.cdk-describedby-message-container div[role="tooltip"]'):
                tid = div.get("id")
                if tid:
                    tooltips[tid] = div.text

            headers = ["Player", "Nation", "Position"]
            table = soup.select_one(".fs-table-responsive table")
            if not table:
                break

            for th in table.select("thead th"):
                if th.get("aria-describedby"):
                    header_text = th.select_one(".header-text span")
                    if header_text:
                        headers.append(header_text.text)

            players_data = []
            for row in table.select("tbody tr"):
                player = {}
                name_cell = row.select_one(".nom-joueur")
                if name_cell:
                    player_link = name_cell.select_one("a")
                    if player_link:
                        player["Player"] = player_link.text.strip()
                    player_img = name_cell.select_one("img")
                    if player_img and player_img.get("aria-describedby"):
                        tooltip_id = player_img.get("aria-describedby")
                        player["Nation"] = tooltips.get(tooltip_id, "")
                    position_class = None
                    for cls in row.select_one("td").get("class", []):
                        if cls.startswith("position"):
                            position_class = cls
                            break
                    if position_class:
                        pos_num = position_class.replace("position", "")
                        positions = {
                            "1": "Prop", "2": "Hooker", "3": "Prop", "4": "Lock",
                            "5": "Lock", "6": "Back Row", "7": "Back Row", "8": "Number 8",
                            "9": "Scrum-half", "10": "Fly-half", "11": "Wing", "12": "Centre",
                            "13": "Centre", "14": "Wing", "15": "Full-back"
                        }
                        player["Position"] = positions.get(pos_num, f"Position {pos_num}")
                cells = row.select("td")
                for i, cell in enumerate(cells[1:], 3):
                    if i < len(headers):
                        value = cell.select_one("span")
                        if value:
                            text_value = value.text.strip()
                            try:
                                if "." in text_value:
                                    player[headers[i]] = float(text_value)
                                elif text_value:
                                    player[headers[i]] = int(text_value)
                                else:
                                    player[headers[i]] = 0
                            except ValueError:
                                player[headers[i]] = text_value
                players_data.append(player)

            all_players_data.extend(players_data)

            try:
                try:
                    next_button = driver.find_element(By.CSS_SELECTOR, "button[aria-label='Next page']")
                except NoSuchElementException:
                    try:
                        next_button = driver.find_element(By.CSS_SELECTOR, "button.mat-mdc-paginator-navigation-next")
                    except NoSuchElementException:
                        next_button = driver.find_element(By.XPATH, "//button[contains(@class, 'paginator') and contains(@class, 'next')]")
                is_disabled = next_button.get_attribute("disabled") == "true" or "mat-disabled" in next_button.get_attribute("class")
                if is_disabled:
                    has_more_pages = False
                else:
                    try:
                        next_button.click()
                    except (ElementClickInterceptedException, StaleElementReferenceException):
                        driver.execute_script("arguments[0].click();", next_button)
                    try:
                        WebDriverWait(driver, 10).until(EC.staleness_of(table))
                    except:
                        pass
                    page_num += 1
                    time.sleep(1)
            except NoSuchElementException:
                has_more_pages = False
            except Exception:
                has_more_pages = False

        df = pd.DataFrame(all_players_data)
        df.to_csv("games.csv", index=False)
        print("Scraping complete.")
    except Exception:
        traceback.print_exc()
    finally:
        input("Press Enter to exit...")
        driver.quit()

def scrape_match_data():
    chrome_options = Options()
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        driver.get(LOGIN_URL)
        print("Please log in manually within 2 minutes...")
        wait = WebDriverWait(driver, 120)
        wait.until(lambda d: "fantasy.sixnationsrugby.com" in d.current_url)
        driver.get(MATCH_URL)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "calendrier-matchs-2024-6n")))
        time.sleep(2)  # extra wait for dynamic content

        all_match_data = []
        # Find the container that holds all the matches
        match_container = driver.find_element(By.CSS_SELECTOR, "calendrier-matchs-2024-6n")
        # Locate each match element; adjust the selector if needed based on actual DOM structure
        matches = match_container.find_elements(By.CSS_SELECTOR, ".ctr-match.fs-box")
        print(f"Found {len(matches)} matches.")

        for match in matches:
            # Click the score element to reveal extra match details
            try:
                score_elem = match.find_element(By.CSS_SELECTOR, ".ctr-match-score")
                score_elem.click()
                # Wait briefly for additional details to load (adjust if there's an animation)
                time.sleep(1)
            except Exception as e:
                print("Error clicking match score:", e)
            
            # Get match date by navigating to the parent container if available.
            try:
                # The match date might be outside the match box, so we look upward for the closest date container.
                date_elem = match.find_element(By.XPATH, "ancestor::div[contains(@class, 'ctr-calendrier-match-2024-6N')]//div[contains(@class, 'ctr-match-date')]//div[contains(@class, 'match-date')]")
                match_date = date_elem.text.strip()
            except Exception:
                match_date = ""

            # Extract team names and scores from the match table
            try:
                table = match.find_element(By.TAG_NAME, "table")
                cells = table.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 3:
                    home_team_elem = cells[0].find_element(By.CSS_SELECTOR, ".club")
                    home_team = home_team_elem.text.strip()
                    away_team_elem = cells[2].find_element(By.CSS_SELECTOR, ".club")
                    away_team = away_team_elem.text.strip()
                    # In the center cell, find the score details.
                    score_detail = cells[1].find_element(By.CSS_SELECTOR, ".match-score-detail")
                    scores = score_detail.find_elements(By.TAG_NAME, "span")
                    if len(scores) >= 3:
                        home_score = scores[0].text.strip()
                        away_score = scores[2].text.strip()
                    else:
                        home_score, away_score = "", ""
                else:
                    home_team, away_team, home_score, away_score = "", "", "", ""
            except Exception:
                home_team, away_team, home_score, away_score = "", "", "", ""

            match_data = {
                "Date": match_date,
                "HomeTeam": home_team,
                "HomeScore": home_score,
                "AwayTeam": away_team,
                "AwayScore": away_score
            }
            all_match_data.append(match_data)

        df = pd.DataFrame(all_match_data)
        df.to_csv("matches.csv", index=False)
        print("Match data scraping complete.")
    except Exception:
        traceback.print_exc()
    finally:
        input("Press Enter to exit...")
        driver.quit()

if __name__ == "__main__":
    # Uncomment the function you want to run:
    # scrape_rugby_stats()
    scrape_match_data()