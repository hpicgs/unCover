from scraper.article_scraper import GoogleScraper
from scraper.page_processor import PageProcessor
from database.mock_database import DatabaseAuthorship
from definitions import QUERIES
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re, requests, argparse, time



def preprocess_article(doc):
    paragraph = re.sub("\s+", " ", doc)
    if any(c in paragraph.lower() for c in ["'", '"', "/", "\n","_","|", "\\", "@", "copyright"]) or len(paragraph) < 10:
        return None
    return paragraph


def generate_author_dataset(site, author, narticles=10):
    #writes the narticles most recent articles into the mock database
    article_urls = []
    if site == "https://www.theguardian.com":
        def get_listed_articles(url):
            try:
                page = requests.get(url).text
            except:
                return None
            soup = BeautifulSoup(page, features="html.parser")
            article_links = soup.find_all("a")
            return [article_link.get("href") for article_link in article_links if article_link.get("data-link-name") == "article"][::2]
        pagenum = 1
        while len(article_urls) < narticles:
            time.sleep(0.2)
            url = author + "?page=" + str(pagenum)
            urls = get_listed_articles(url)
            if urls and not any(link in article_urls for link in urls):
                article_urls += urls
            else:
                break
            pagenum += 1
    for article_url in article_urls[:narticles]:
        time.sleep(0.2)
        print(article_url)
        page = requests.get(article_url).text
        processor = PageProcessor(page)
        processed_page = preprocess_article(processor.get_fulltext())
        author = processor.get_author()
        DatabaseAuthorship.insert_article(processed_page, article_url, author)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--author", action="store", type=str)
    parser.add_argument("--publication", action="store", type=str)
    parser.add_argument("--narticles", action="store", type=int)

    args = parser.parse_args()
    if args.dataset:
        generate_author_dataset(args.publication, args.author, args.narticles)
    else:
        scraper = GoogleScraper(verbose=True)
        for query in QUERIES:
            urls = scraper.find_news_urls_for_query(query, 5)
            for url in urls:
                page = requests.get(url).text
                processor = PageProcessor(page)
                processed_page = preprocess_article(processor.get_fulltext())
                author = processor.get_author()
                DatabaseAuthorship.insert_article(processed_page, url, author)

