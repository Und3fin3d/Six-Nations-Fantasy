import argparse
import datetime
import hashlib
import json
import os
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


ROOT = 'https://fantasy.nationschampionshiprugby.com'


def run(output):
    os.makedirs(output, exist_ok=False)
    records, pages, links = [], [], set()
    response = requests.get(ROOT, timeout=45)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    assets = [urljoin(ROOT, node['src']) for node in soup.select('script[src]')]
    records.append(dict(url=ROOT, status=response.status_code, bytes=len(response.content),
                        sha256=hashlib.sha256(response.content).hexdigest(), script_assets=assets))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(locale='en-GB')
        page = context.new_page()
        for route in ('/en/', '/en/rules/', '/en/how-to-play/', '/en/faqs/'):
            url = urljoin(ROOT, route)
            try:
                result = page.goto(url, wait_until='networkidle', timeout=90000)
                text = page.locator('body').inner_text(timeout=20000)
                hrefs = page.locator('a[href]').evaluate_all('(nodes) => nodes.map(node => ({text: node.innerText, href: node.href}))')
                for item in hrefs:
                    if re.search(r'rule|scor|how.to.play|faq|term', item['text'] + ' ' + item['href'], re.I):
                        links.add(item['href'])
                pages.append(dict(requested_url=url, final_url=page.url, status=result.status if result else None,
                                  fetched_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  title=page.title(), text=text, links=hrefs,
                                  text_sha256=hashlib.sha256(text.encode()).hexdigest()))
            except Exception as error:
                pages.append(dict(requested_url=url, error=f'{type(error).__name__}: {error}'))
        for url in sorted(links):
            parsed = urlparse(url)
            if parsed.hostname != urlparse(ROOT).hostname or any(item.get('final_url') == url for item in pages):
                continue
            try:
                result = page.goto(url, wait_until='networkidle', timeout=90000)
                text = page.locator('body').inner_text(timeout=20000)
                pages.append(dict(requested_url=url, final_url=page.url, status=result.status if result else None,
                                  fetched_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  title=page.title(), text=text,
                                  text_sha256=hashlib.sha256(text.encode()).hexdigest()))
            except Exception as error:
                pages.append(dict(requested_url=url, error=f'{type(error).__name__}: {error}'))
        context.close()
        browser.close()
    with open(f'{output}/public_pages.json', 'w') as handle:
        json.dump(pages, handle, indent=2)
    with open(f'{output}/retrieval.json', 'w') as handle:
        json.dump(dict(public_requests=records, discovered_rules_links=sorted(links),
                       rugby_api_calls=0, credentials_used=False,
                       interpretation='Current official public pages, not proof of unchanged historical rules. No scoring or model alteration is made by this audit.'), handle, indent=2)
    for index, item in enumerate(pages):
        with open(f'{output}/page_{index:02d}.txt', 'w') as handle:
            handle.write(item.get('final_url', item['requested_url']) + '\n\n' + item.get('text', item.get('error', '')))
        print(item.get('final_url', item['requested_url']), len(item.get('text', '')), item.get('error', ''), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
