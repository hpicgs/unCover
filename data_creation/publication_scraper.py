import time
import requests
from bs4 import BeautifulSoup
from misc.mock_database import Database
from misc.nlp_helpers import preprocess_article
from data_creation.page_processor import PageProcessor


def get_listed_articles(url, source):
    try:
        page = requests.get(url).text
    except Exception as e:
        print(f"Error fetching page: {e}")
        return None
    soup = BeautifulSoup(page, features='html.parser')
    article_links = soup.find_all('a')
    if source == 'theguardian':
        return set([article_link.get('href') for article_link in article_links if
                    article_link.get('data-link-name') == 'article'])
    elif source == 'ntv':
        return set([article_link.get('href') for article_link in article_links if
                    'article' in article_link.get('href') and 'mediathek' not in article_link.get('href')])


def find_authors_articles(n: int, source: str, author: str) -> list[any]:
    if source == 'theguardian':
        pagenum = 1
        result = set()
        while len(result) < n:
            print(f"Fetching guardian page {pagenum}...")
            time.sleep(0.2)
            urls = get_listed_articles(f"https://theguardian.com/profile/{author}?page={pagenum}", source)
            if urls and not any(link in result for link in urls):
                result.update(urls)
            else:
                break
            pagenum += 1
        return list(result)
    elif source == 'ntv':
        return list(get_listed_articles(f"https://www.n-tv.de/autoren/{author}.html", source))
    return list()


def scrape_publication(database: Database, publication:str, author: str, narticles: int=10, max: int=0) -> None:
    if "theguardian.com" in publication:
        article_urls = find_authors_articles(narticles, "theguardian", author)
    elif "n-tv.de" in publication:
        article_urls = find_authors_articles(narticles, "ntv", author)
    else:
        print("The mentioned publication is currently not supported, please use [theguardian.com|n-tv.de]")
        exit(1)
    if max > 0:
        article_urls = article_urls[:max]
    for article_url in article_urls:
        time.sleep(0.2)
        print(article_url)
        page = requests.get(article_url).text
        processor = PageProcessor(page)
        processed_page = preprocess_article(processor.get_fulltext(separator="\n"))
        author = processor.get_author().replace(' ', '')
        database.insert_article(processed_page, article_url, author)
