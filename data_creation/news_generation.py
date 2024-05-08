import argparse
import os
import re
import sys
import json
from typing import Optional
import openai
from enum import Enum
import google.generativeai as genai
from misc.mock_database import Database
from transformers import pipeline, set_seed
from misc.logger import printProgressBar
from definitions import ROOT_DIR, MODELS_DIR, GPT_KEY, OPENAI_ORGA, GOOGLE_API_KEY

sys.path.append(os.path.join(ROOT_DIR, 'data_creation', 'grover'))
from data_creation.grover.sample.contextual_generate import generate_grover_news_from_original


def generate_gemini_news_from(doc, german=False) -> Optional[str]:
    print("Starting Gemini Request")
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    if not german:
        doc += "\n\nWrite a long news article about this topic."
    else:
        doc += "\n\nSchreibe einen langen Nachrichten Artikel über das Thema."
    response = model.generate_content(doc)
    try:
        return response.text
    except ValueError as e:
        print(f"Error while generating gemini article as none was generated: {e}")
        print(response.prompt_feedback)
        return None


def generate_gpt3_news_from(doc, german=False, size=1000) -> Optional[str]:
    print("Starting GPT3 Request")
    client = openai.OpenAI(
        api_key=GPT_KEY,
        organization=OPENAI_ORGA
    )
    if not german:
        doc += "\n\nWrite a long news article about this topic."
    else:
        doc += "\n\nSchreibe einen langen Nachrichten Artikel über das Thema."
    response = None
    while response is None:
        try:
            response = client.completions.create(model='gpt-3.5-turbo-instruct', prompt=doc, max_tokens=size,
                                                 temperature=0.4)
        except openai.APIConnectionError or openai.Timeout:
            continue
        except openai.BadRequestError:
            break

    print("GPT3 finished")
    return response.choices[0].text if response else None


def generate_gpt4_news_from(doc, german=False, size=1000) -> Optional[str]:
    print("Starting GPT 4 Request")
    client = openai.OpenAI(
        api_key=GPT_KEY,
        organization=OPENAI_ORGA
    )
    if not german:
        doc += "\n\nWrite a long news article about this topic."
    else:
        doc += "\n\nSchreibe einen langen Nachrichten Artikel über das Thema."
    response = None
    while response is None:
        try:
            response = client.chat.completions.create(model='gpt-4', messages=[{'role': 'user', 'content': doc}],
                                                      max_tokens=size, temperature=0.4)
        except openai.APIConnectionError or openai.Timeout:
            continue
        except openai.RateLimitError:
            break

    print("GPT4 finished")
    return response.choices[0].message.content if response else None


def generate_gpt2_news_from(doc, size=1000) -> Optional[str]:
    print("Starting GPT2 Generation")
    doc_prompt = "I would write a news article about the topic like this:"
    generator = pipeline('text-generation', model='gpt2-large')
    set_seed(42)
    doc += f". {doc_prompt}"
    result = generator(doc, max_length=size, num_return_sequences=1)[0]['generated_text']
    print("GPT2 Finished")
    return (result.rsplit('.', 1)[0] + '.').split(doc_prompt, 1)[1]


def single_query_generation(article: str, m: str, url: str) -> Optional[str]:
    if 'gpt2' in m:
        return generate_gpt2_news_from(article[:25])
    if 'gpt3' in m:
        return generate_gpt3_news_from(article)
    if 'gpt4' in m:
        return generate_gpt4_news_from(article)
    if 'gemini' in m:
        return generate_gemini_news_from(article)
    if 'grover' in m:
        grover_input = json.dumps({'url': url, 'url_used': url, 'title': article[:25],
                                   'text': article, 'summary': '', 'authors': [],
                                   'publish_date': '04-19-2023', 'domain': 'www.com', 'warc_date': '20190424064330',
                                   'status': 'success', 'split': 'gen', 'inst_index': 0})
        return generate_grover_news_from_original(grover_input, m.replace('grover-', '', 1), MODELS_DIR)


def query_generation(database: Database, articles: list[Optional[tuple]], args: argparse.Namespace) -> None:
    for i, article in enumerate(articles):
        printProgressBar(i, len(articles), fill='█')
        print(url := article[1])
        processed_page = re.sub(r"\s+", ' ', article[0])
        if len(processed_page) < 600:  # this is to filter out error messages and other scraping mistakes
            print("\noriginal article is too short; -> skipping for consistency")
            continue

        for model in args.models.split(','):
            tmp = single_query_generation(processed_page, model, url)
            if tmp is not None:
                database.insert_article(tmp, url, model)


def phrase_generation(database: Database, args: argparse.Namespace) -> None:
    for phrase in args.phrases.split(','):
        print(f"Working on: '{phrase}'...")
        if args.gpt2:
            database.insert_article(generate_gpt2_news_from(phrase), phrase, 'gpt2-phrase')
        if args.gpt3:
            tmp = generate_gpt3_news_from(phrase)
            if tmp is not None:
                database.insert_article(tmp, phrase, 'gpt3-phrase')
