from scraper.article_scraper import GoogleScraper
from scraper.page_processor import PageProcessor
from database.mock_database import DatabaseAuthorship
from definitions import QUERIES
from dotenv import load_dotenv
import re, requests



def preprocess_article(doc):
    paragraph = re.sub("\s+", " ", doc)
    if any(c in paragraph.lower() for c in ["'", '"', "/", "\n","_","|", "\\", "@", "copyright"]) or len(paragraph) < 10:
        return None
    return paragraph


if __name__ == '__main__':
    scraper = GoogleScraper(verbose=True)
    for query in QUERIES:
        urls = scraper.find_news_urls_for_query(query, 5)
        for url in urls:
            print(url)
            page = requests.get(url).text
            processor = PageProcessor(page)
            processed_page = preprocess_article(processor.get_fulltext())
            author = processor.get_author()
            DatabaseAuthorship.insert_article(processed_page, url, author)

