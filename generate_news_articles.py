import json
import os
import sys
import re
import requests
import argparse
from data_creation.article_scraper import GoogleScraper
from data_creation.page_processor import PageProcessor
from data_creation.gpt_generator import generate_gpt4_news_from, generate_gpt3_news_from, generate_gpt2_news_from
from data_creation.gemini_generator import generate_gemini_news_from
from misc.mock_database import DatabaseGenArticles, GermanDatabase
from definitions import ROOT_DIR, MODELS_DIR

sys.path.append(os.path.join(ROOT_DIR, 'data_creation', 'grover'))
from data_creation.grover.sample.contextual_generate import generate_grover_news_from_original


def query_generation(queries, args):
    visited = set()
    database = DatabaseGenArticles
    if args.german:
        database = GermanDatabase
    for query in queries:
        print(f"Round starting for '{query}'...")
        count = 1
        scraper = GoogleScraper(verbose=True)
        try:
            urls = scraper.find_news_urls_for_query(query, args.narticles, args.german)
        except:
            try:
                urls = scraper.find_news_urls_for_query(query, args.narticles, args.german)
            except:
                print("error fetching urls")
                continue
        print(urls)
        for url in urls:
            if url in visited:
                continue
            visited.add(url)
            print(f"Current URL Nr: {count} {url}")
            count += 1
            try:
                page = requests.get(url).text
            except Exception as e:
                print(f"error fetching page: {e}")
                continue
            processor = PageProcessor(page)
            print("fetched page")
            processed_page = re.sub(r"\s+", ' ', processor.get_fulltext())
            if len(processed_page) < 600:  # this is to filter out error messages and other scraping mistakes
                print("original article is too short; -> skipping for consistency")
                continue
            title = processor.get_title()
            print("start processing")
            if args.gpt2:
                database.insert_article(generate_gpt2_news_from(title), url, 'gpt2')
            if args.gpt3:
                tmp = generate_gpt3_news_from(processed_page)
                if tmp is not None:
                    database.insert_article(tmp, url, 'gpt3')
            if args.gpt4:
                tmp = generate_gpt4_news_from(processed_page, args.german)
                if tmp is not None:
                    database.insert_article(tmp, url, 'gpt4')
            if args.gemini:
                tmp = generate_gemini_news_from(processed_page, args.german)
                if tmp is not None:
                    database.insert_article(tmp, url, 'gemini')
            if args.grover:
                grover_input = json.dumps({'url': url, 'url_used': url, 'title': title, 'text': processed_page,
                                           'summary': '', 'authors': [], 'publish_date': '04-19-2023',
                                           'domain': 'www.com',
                                           'warc_date': '20190424064330', 'status': 'success', 'split': 'gen',
                                           'inst_index': 0})
                database.insert_article(
                    generate_grover_news_from_original(grover_input, args.grover, MODELS_DIR), url, 'grover')


def phrase_generation(phrases, args):
    for phrase in phrases:
        print(f"Working on: '{phrase}'...")
        if args.gpt2:
            DatabaseGenArticles.insert_article(generate_gpt2_news_from(phrase), phrase, 'gpt2-phrase')
        if args.gpt3:
            tmp = generate_gpt3_news_from(phrase)
            if tmp is not None:
                DatabaseGenArticles.insert_article(tmp, phrase, 'gpt3-phrase')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--narticles', action='store', type=int, default=5, required=False,
                        help="maximum number of articles to scrape per query, that will then be used by each method to generate")
    parser.add_argument('--queries', action='store', type=str, required=False,
                        help="scrape articles for a given query, insert multiple values comma separated")
    parser.add_argument('--phrases', action='store', type=str, required=False,
                        help="generate an articles by comma separated phrases")
    parser.add_argument('--gpt4', action='store_true', required=False,
                        help="use gpt4 for text generation, only implemented for whole articles")
    parser.add_argument('--gpt3', action='store_true', required=False,
                        help="use gpt3 for text generation")
    parser.add_argument('--gpt2', action='store_true', required=False,
                        help="use gpt2 for text generation, only uses title or phrase not whole articles")
    parser.add_argument('--gemini', action='store_true', required=False,
                        help="use gemini for text generation, only implemented for whole articles")
    parser.add_argument('--grover', action='store', type=str, required=False,
                        help="use grover for text generation and mention model size, does not work with phrases")
    parser.add_argument('--german', action='store_true', required=False,
                        help="use german dataset instead of english")

    args = parser.parse_args()
    if args.queries and args.phrases:
        parser.error("please provide either queries or phrases")
    elif not args.queries and not args.phrases:
        parser.error("please provide at least one query or phrase")
    elif not args.queries and args.phrases:
        if args.german:
            parser.error("please use the german generation only with query mode")
        if not args.gpt3 and not args.gpt2:
            parser.error("please provide at least one valid method for generating news with phrases")
        phrase_generation(args.phrases.split(','), args)
    elif args.queries and not args.phrases:
        if all([args.grover, args.gemini, args.gpt4, args.gpt3, args.gpt2]):
            parser.error("please provide at least one method valid for generating news with queries")
        query_generation(args.queries.split(','), args)
