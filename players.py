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
import threading
import sys

LOGIN_URL = "https://account.sixnationsrugby.com/en/sign-in"
STATS_URL = "https://fantasy.sixnationsrugby.com/m6n/#/game/stats"

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

        # Start a background input watcher so the user can type 'q' + Enter
        # at any time to request the scraper stop early.
        stop_event = threading.Event()

        def input_watcher(ev):
            while not ev.is_set():
                try:
                    line = input()
                except EOFError:
                    break
                if line.strip().lower() == 'q':
                    ev.set()
                    print("Stop requested by user ('q').")
                    break

        watcher = threading.Thread(target=input_watcher, args=(stop_event,), daemon=True)
        watcher.start()
        print("Type 'q' then Enter at any time to stop scraping.")

        wait = WebDriverWait(driver, 120)
        wait.until(lambda d: "fantasy.sixnationsrugby.com" in d.current_url)
        driver.get(STATS_URL)
        time.sleep(3)
        wait = WebDriverWait(driver, 30)

        TABLE_SELECTORS = [
            ".fs-table-responsive table",
            "table.mat-mdc-table",
            "table",
        ]

        table_selector = None
        for selector in TABLE_SELECTORS:
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                table_selector = selector
                print(f"Table found with selector: {selector}")
                break
            except TimeoutException:
                print(f"Selector not found: {selector}, trying next...")

        if table_selector is None:
            print(f"Current URL: {driver.current_url}")
            print(f"Page title: {driver.title}")
            print("Page source snippet:")
            print(driver.page_source[:2000])
            raise TimeoutException("Could not find a stats table with any known selector.")
        
        all_players_data = []
        page_num = 1
        has_more_pages = True

        while has_more_pages:
            if stop_event.is_set():
                print("Stop requested — exiting scrape loop.")
                break
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, "html.parser")

            tooltips = {}
            for div in soup.select('.cdk-describedby-message-container div[role="tooltip"]'):
                tid = div.get("id")
                if tid:
                    tooltips[tid] = div.text

            headers = ["Player", "Nation"]
            table = soup.select_one(table_selector)
            if not table:
                print(f"Table not found in page source with selector: {table_selector}")
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
                cells = row.select("td")
                for i, cell in enumerate(cells[1:], 2):
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
        df.to_csv("rugby_stats.csv", index=False)
        print("Scraping complete.")
    except Exception:
        traceback.print_exc()
    finally:
        input("Press Enter to exit...")
        driver.quit()

if __name__ == "__main__":
    scrape_rugby_stats()