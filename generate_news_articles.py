import json

from scraper.article_scraper import GoogleScraper
from scraper.page_processor import PageProcessor
from generator.gpt3_generator import generate_gpt3_news_from_original
from grover.sample.contextual_generate import generate_grover_news_from_original
from database.mock_database import DatabaseGenArticles
from definitions import MODELS_DIR
import re, requests, argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--narticles", action="store", type=int, default=3, required=False, help="maximum number of articles to scrape per query, that will then be used by each method to generate")
    parser.add_argument("--queries", action="store", type=str, required=False, help="scrape articles for a given query, insert multiple values comma separated")
    parser.add_argument("--gpt3", action="store_true", required=False, help="use gpt3 for text generation")
    parser.add_argument("--grover", action="store_true", required=False, help="use grover for text generation")

    args = parser.parse_args()
    if not args.queries:
            parser.error("please provide at least one query")
    if not args.grover and not args.gpt3:
        parser.error("please provide at least one method for generating news")
    for query in args.queries.split(","):
        print("Round starting for " + query)
        count = 1
        scraper = GoogleScraper(verbose=True)
        urls = scraper.find_news_urls_for_query(query, args.narticles)
        print(urls)
        for url in urls:
            print("Current URL Nr: " + str(count) + " " + url)
            count += 1
            page = requests.get(url).text
            processor = PageProcessor(page)
            print("fetched page")
            processed_page = re.sub("\s+", " ", processor.get_fulltext())
            title = processor.get_title()
            print("start processing")
            if args.gpt3:
                tmp = generate_gpt3_news_from_original(processed_page)
                if tmp is not None:
                    DatabaseGenArticles.insert_article(tmp, url, "gpt3")

            if args.grover:
                grover_input = json.dumps({"url": url, "url_used": url, "title": title, "text": processed_page,
                                    "summary": "", "authors": [], "publish_date": "04-19-2023", "domain": "www.com",
                                    "warc_date":"20190424064330", "status": "success", "split": "gen", "inst_index": 0})
                DatabaseGenArticles.insert_article(
                    generate_grover_news_from_original(grover_input, "base", MODELS_DIR), url, "grover")
