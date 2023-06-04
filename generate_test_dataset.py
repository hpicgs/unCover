from database.mock_database import TestDatabase
import argparse, requests, re, json
from scraper.page_processor import PageProcessor
from scraper.article_scraper import GoogleScraper
from generator.gpt_generator import generate_gpt3_news_from, generate_gpt2_news_from
from grover.sample.contextual_generate import generate_grover_news_from_original
from definitions import MODELS_DIR

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--narticles", action="store", type=int, default=3, required=False,
                        help="maximum number of articles to scrape per query, that will then be used by each method to generate")
    parser.add_argument("--max_amount", action="store", type=int, default=100, required=False,
                        help="maximum number of texts to store into database per method")
    parser.add_argument("--queries", action="store", type=str, required=False,
                        help="scrape articles for a given query, insert multiple values comma separated")
    parser.add_argument("--methods", action="store", type=str, required=False, default="",
                        help="what data should be part of the test data, available are human, gpt2, gpt3, grover")

    args = parser.parse_args()
    methods = args.methods.split(",")
    if not args.queries:
        parser.error("please provide at least one query")
    if len(methods) == 0:
        parser.error("please provide at least one method")
    elif not ("human" in methods or "gpt3" in methods or "gpt2" in methods or "grover" in methods):
        parser.error("the provided methods are not valid, please separate them by ','")
    count = 1
    for query in args.queries.split(","):
        print("Round starting for " + query)
        scraper = GoogleScraper(verbose=True)
        try:
            urls = scraper.find_news_urls_for_query(query, args.narticles)
        except:
            try:
                urls = scraper.find_news_urls_for_query(query, args.narticles)
            except:
                print("error fetching urls")
                continue
        print(urls)
        for url in urls:
            print(f"Current URL Nr: {count} {url}")
            try:
                page = requests.get(url, timeout=60).text
                processor = PageProcessor(page)
                print("fetched page")
            except:
                print("error fetching page")
                continue
            processed_page = re.sub("\s+", " ", processor.get_fulltext())
            title = processor.get_title()
            print("start processing")
            gpt3, gpt2, grover, human = None, None, None, None
            if "gpt3" in methods:
                # we will use gpt3 for generating news from titles and whole articles for the same amount
                if count % 2 == 0:
                    gpt3 = generate_gpt3_news_from(title)
                else:
                    gpt3 = generate_gpt3_news_from(processed_page)
                if gpt3 is None:
                    print("article at this url too long for gpt3; -> skipping for consistency")
                    continue
            if "gpt2" in methods:
                try:
                    gpt2 = generate_gpt2_news_from(title)
                except:
                    print("article produces error for gpt2; -> skipping for consistency")
                    continue
                if len(gpt2) < 200:  # make sure gpt2 generated well enough
                    print("article by gpt2 is too short; -> skipping for consistency")
                    continue
            if "grover" in methods:
                try:
                    grover_input = json.dumps({"url": url, "url_used": url, "title": title, "text": processed_page,
                                               "summary": "", "authors": [], "publish_date": "04-19-2023",
                                               "domain": "www.com", "warc_date": "20190424064330", "status": "success",
                                               "split": "gen","inst_index": 0})
                    grover = generate_grover_news_from_original(grover_input, "base", MODELS_DIR)
                except:
                    print("article produces error for grover; -> skipping for consistency")
                    continue
            if "human" in methods:
                human = processor.get_fulltext(separator="\n")
                if len(human) < 600:  # this is to filter out error messages and other scraping mistakes
                    print("original article is too short; -> skipping for consistency")
                    continue
                TestDatabase.insert_article(human , url, "human")

            if gpt3:
                TestDatabase.insert_article(gpt3, url, "gpt3")
            if gpt2:
                TestDatabase.insert_article(gpt2, url, "gpt2")
            if grover:
                TestDatabase.insert_article(grover, url, "grover")

            if count == args.max_amount:
                break
            count += 1
        if count == args.max_amount:
            print("reached maximum amount of articles, ended on query: " + query)
            break
    print("finished at count: " + str(count))
