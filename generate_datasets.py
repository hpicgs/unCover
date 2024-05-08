import argparse
import requests
from typing import Optional
from definitions import DATABASE_AUTHORS_PATH, DATABASE_GEN_PATH, DATABASE_TEST_PATH, DATABASE_GERMAN_PATH, \
    DATABASE_GERMAN_TEST_PATH
from misc.logger import printProgressBar
from misc.mock_database import Database
from misc.nlp_helpers import preprocess_article
from data_creation.article_scraper import GoogleScraper
from data_creation.page_processor import PageProcessor
from data_creation.publication_scraper import scrape_publication
from data_creation.news_generation import phrase_generation, query_generation, single_query_generation


def scrape_google_news(query: str, n: int, database: Optional[Database] = None) -> Optional[list[tuple]]:
    visited = set()
    scraper = GoogleScraper(verbose=True)
    try:
        urls = scraper.find_news_urls_for_query(query, n, args.german)
    except:
        try:
            urls = scraper.find_news_urls_for_query(query, n, args.german)
        except Exception as e:
            print(f"error fetching urls: {e}")
            return []
    results = []
    for url in urls:
        if url in visited:
            continue
        visited.add(url)
        try:
            page = requests.get(url).text
            processor = PageProcessor(page)
        except Exception as e:
            print(f"error fetching page: {e}")
            continue
        processed_page = preprocess_article(processor.get_fulltext(separator="\n"))
        author = processor.get_author()
        if database:
            database.insert_article(processed_page, author, url)
        else:
            results.append((processed_page, url))
    if not database:
        return results


def process_human_authors(args: argparse.Namespace) -> None:
    database = Database(DATABASE_GERMAN_PATH if args.german else DATABASE_AUTHORS_PATH)
    if args.dataset:
        if (not args.publication) or (not args.author):
            print("--dataset requires that both --author and --publication are set")
            exit(1)
        scrape_publication(database, args.publication, args.author, args.narticles, args.max_amount)
        exit(0)
    for query in args.queries.split(','):
        scrape_google_news(query, args.narticles, database)


def generate_ai_news(args: argparse.Namespace) -> None:
    database = Database(DATABASE_GERMAN_PATH if args.german else DATABASE_GEN_PATH)
    if args.phrases:
        if args.german:
            print("please use the german generation only with query mode")
            exit(1)
        if not args.gpt3 and not args.gpt2:
            print("please provide at least one valid method for generating news with phrases [gpt3, gpt2]")
            exit(1)
        phrase_generation(database, args)
    else:
        articles = []
        for query in args.queries.split(','):
            articles.extend(scrape_google_news(query, args.narticles))
        query_generation(database, articles, args)


def generate_test_dataset(args: argparse.Namespace) -> None:
    database = Database(DATABASE_GERMAN_TEST_PATH if args.german else DATABASE_TEST_PATH)
    i = 0
    for query in args.queries.split(','):
        printProgressBar(i, args.max_amount, fill='█')
        for article in scrape_google_news(query, args.narticles, database):
            skip = False
            if i == args.max_amount:
                return
            insertions = []
            for model in args.models.split(','):
                tmp = single_query_generation(article[0], model, article[1])
                if tmp is None:
                    skip = True
                    break
                insertions.append((tmp, model))
            if skip:
                continue
            database.insert_article(article[0], article[1], 'human' if args.human != 'verified' else 'human-verified')
            for insertion in insertions:
                database.insert_article(insertion[0], article[1], insertion[1])
            i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general arguments
    parser.add_argument('--dataset', action='store', required=True, choices=['human', 'ai', 'test'],
                        help="type of dataset that should be created")
    parser.add_argument('--narticles', action='store', type=int, default=1, required=False,
                        help="maximum number of articles to scrape per query/ phrase")
    parser.add_argument('--max-amount', action='store', type=int, required=False,
                        help="maximum number of texts to process")
    parser.add_argument('--german', action='store_true', required=False,
                        help="use german dataset instead of english")

    # scraping arguments
    parser.add_argument('--publication', action='store', type=str, required=False,
                        help="url to the news publication used for scraping")
    parser.add_argument('--author', action='store', type=str, required=False,
                        help="url name of the chosen author's page on the publication")
    parser.add_argument('--queries', action='store', type=str, required=False,
                        help="queries for google news separated by comma when not using publication")

    # generation arguments
    parser.add_argument('--phrases', action='store', type=str, required=False,
                        help="phrases for generation separated by comma, can not be used with queries")

    # human label for test dataset
    parser.add_argument('--human', action='store', required=False, choices=['random', 'verified'],
                        default='random', help="defines the human label of the currently scraped data")

    # available models
    parser.add_argument('--models', action='store', type=str, required=False,
                        help="models uses for generation separated by comma [gpt2, gpt3, gpt4, gemini, grover-[base, large, mega]")
    args = parser.parse_args()
    # ensure only one of queries, phrases or publication is set
    if not sum([1 for arg in [args.queries, args.phrases, args.publication] if arg]) == 1:
        parser.error("please provide either at least one query, at least one phrase or a publication")

    if args.mode == 'human':
        process_human_authors(args)
        exit(0)
    if args.mode == 'ai':
        generate_ai_news(args)
    elif args.mode == 'test':
        generate_test_dataset(args)
    else:
        print("Invalid mode")
        parser.print_help()
        exit(1)
