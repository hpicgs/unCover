from scraper.article_scraper import GoogleScraper
from scraper.page_processor import PageProcessor
from generator.gpt3_generator import generate_gpt3_news_from_original
from generator.grover_generation import generate_grover_news_from_original
from database.mock_database import DatabaseGenArticles
import re, requests, argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--narticles", action="store", type=int, default=1, required=False, help="maximum number of articles to scrape")
    parser.add_argument("--queries", action="store", type=str, required=False, help="scrape articles for a given query, insert multiple values comma separated")
    parser.add_argument("--gpt3", action="store_true", required=False, help="use gpt3 for text generation")
    parser.add_argument("--grover", action="store_true", required=False, help="use gpt3 for text generation")

    args = parser.parse_args()
    if not args.queries:
            parser.error("please provide at least one query")
    for query in args.queries.split(","):
        scraper = GoogleScraper(verbose=True)
        urls = scraper.find_news_urls_for_query(query, args.narticles)
        for url in urls:
            page = requests.get(url).text
            processor = PageProcessor(page)
            processed_page = re.sub("\s+", " ", processor.get_fulltext())
            print(processed_page)
            if args.gpt3:
                DatabaseGenArticles.insert_article(generate_gpt3_news_from_original(processed_page), url, "gpt3")
            if args.grover:
                DatabaseGenArticles.insert_article(generate_grover_news_from_original(processed_page, "base"), url, "grover")
