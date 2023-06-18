from scraper.article_scraper import GoogleScraper
from scraper.page_processor import PageProcessor
from database.mock_database import DatabaseAuthorship
from bs4 import BeautifulSoup
import re, requests, argparse, time



def preprocess_article(doc):
    paragraph = re.sub("\s+", " ", doc)
    return paragraph


def generate_author_dataset(site, author, narticles=10):
    #writes the narticles most recent articles into the mock database
    article_urls = []
    if "theguardian.com" in site:
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
            url = f"{site}/profile/{author}?page={pagenum}"
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
        processed_page = preprocess_article(processor.get_fulltext(separator="\n"))
        print(len(processed_page.split("\n")))
        author = processor.get_author()
        DatabaseAuthorship.insert_article(processed_page, article_url, author)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="store_true", required=False, help="scrape articles in dataset creation mode, without google news")
    parser.add_argument("--publication", action="store", type=str, required=False, help="url to the news publication used in dataset mode")
    parser.add_argument("--author", action="store", type=str, required=False, help="url to the chosen author's page on the publication")
    parser.add_argument("--narticles", action="store", type=int, default=10, required=False, help="maximum number of articles to scrape")
    parser.add_argument("--query", action="store", type=str, required=False, help="scrape articles in query mode, using the parameter from this argument")

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

