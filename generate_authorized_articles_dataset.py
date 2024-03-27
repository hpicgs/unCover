import re
import requests
import argparse
import time
from bs4 import BeautifulSoup
from scraper.article_scraper import GoogleScraper
from scraper.page_processor import PageProcessor
from misc.mock_database import DatabaseAuthorship, GermanDatabase


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


def find_authors_articles(n, source, author):
    if source == "theguardian":
        pagenum = 1
        result = []
        while len(result) < n:
            print(f"Fetching guardian page {pagenum}...")
            time.sleep(0.2)
            urls = get_listed_articles(f"https://theguardian.com/profile/{author}?page={pagenum}", source)
            if urls and not any(link in result for link in urls):
                result += urls
            else:
                break
            pagenum += 1
        return result
    elif source == "ntv":
        return get_listed_articles(f"https://www.n-tv.de/autoren/{author}.html", source)
    return []

def preprocess_article(doc):
    paragraph = re.sub("[ \t\r\f]+", ' ', doc)  # \s without \n
    return paragraph


def generate_author_dataset(site, author, narticles=10):
    # writes the narticles most recent articles into the mock database
    article_urls = []
    database = None
    # only works for theguardian.com and n-tv.de
    if "theguardian.com" in site:
        database = DatabaseAuthorship
        article_urls = find_authors_articles(narticles, "theguardian", author)
    elif "n-tv.de" in site:
        database = GermanDatabase
        article_urls = find_authors_articles(narticles, "ntv", author)
    if narticles != 0:
        article_urls = article_urls[:narticles]
    for article_url in article_urls:
        time.sleep(0.2)
        print(article_url)
        page = requests.get(article_url).text
        processor = PageProcessor(page)
        processed_page = preprocess_article(processor.get_fulltext(separator="\n"))
        author = processor.get_author().replace(" ", "")
        database.insert_article(processed_page, article_url, author)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store_true', required=False,
                        help="scrape articles in dataset creation mode, without google news")
    parser.add_argument('--publication', action='store', type=str, required=False,
                        help="url to the news publication used in dataset mode")
    parser.add_argument('--author', action='store', type=str, required=False,
                        help="url to the chosen author's page on the publication")
    parser.add_argument('--narticles', action='store', type=int, default=10, required=False,
                        help="maximum number of articles to scrape")
    parser.add_argument('--query', action='store', type=str, required=False,
                        help="scrape articles in query mode, using the parameter from this argument")

    args = parser.parse_args()
    if args.dataset:
        if (not args.publication) or (not args.author):
            parser.error("--dataset requires that both --author and --publication are set")
        generate_author_dataset(args.publication, args.author, args.narticles)
    else:
        if not args.query:
            parser.error("please provide a --query to search news.google.com")
        scraper = GoogleScraper(verbose=True)
        urls = scraper.find_news_urls_for_query(args.query, args.narticles)
        for url in urls:
            page = requests.get(url).text
            processor = PageProcessor(page)
            processed_page = preprocess_article(processor.get_fulltext(separator="\n"))
            author = processor.get_author()
            DatabaseAuthorship.insert_article(processed_page, url, author)
