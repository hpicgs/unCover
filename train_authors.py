import os
import pickle
import argparse
import json
import pandas as pd
from nltk.parse.corenlp import CoreNLPDependencyParser

from definitions import STYLOMETRY_DIR
from stylometry.char_trigrams import char_trigrams
from stylometry.semantic_trigrams import sem_trigrams
from stylometry.classifier import trigram_distribution, logistic_regression, used_authors, random_forest
from misc.mock_database import DatabaseAuthorship, DatabaseGenArticles, GermanDatabase
from misc.logger import printProgressBar


def normalize(type, f):
    maxi, mini = [], []
    for i in range(len(f[0])):
        ma = max([f[j][i] for j in range(len(f))]) + 0.01
        mi = min([f[j][i] for j in range(len(f))]) - 0.01
        for j in range(len(f)):
            f[j][i] = (f[j][i] - mi) / (ma - mi)
        maxi.append(ma)
        mini.append(mi)
    with open(os.path.join(STYLOMETRY_DIR, f"{type}_normalization.json"), 'w') as fp:
        json.dump([mini, maxi], fp)
    return f


def get_training_samples(type, min_articles, database):
    result = []
    authors = database.get_authors()
    print(f"{type} authors:")
    print(authors)
    trainable = []
    for author in authors:
        full_article_list = [(article['text'], author) for article in
                             database.get_articles_by_author(author)]
        result += full_article_list[:len(full_article_list)]
        if len(full_article_list) >= min_articles:
            print(f"chose author: {author}")
            trainable.append(author)
    print(f"trainable {type} authors:")
    print(trainable)
    return result


def fit_model(name, samples, labels, n, model_type='logistic'):
    print(f"Training {name}...")
    with open(os.path.join(STYLOMETRY_DIR, f"{name}{n}.pickle"), 'wb') as f:
        if model_type == 'logistic':
            model = logistic_regression(samples, labels)
        else:
            model = random_forest(samples, labels)
        pickle.dump(model, f)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nfeatures', action='store', required=False, type=int, default=100,
                        help="number of char trigram & semantic trigram features used in the distribution")
    parser.add_argument('--minarticles', action='store', required=False, type=int, default=50,
                        help="minimum number of articles required for training on a specific author/model")
    parser.add_argument('--german', action='store_true', required=False,
                        help="use the german test database instead of the english one")
    args = parser.parse_args()
    n_features = args.nfeatures

    os.makedirs(STYLOMETRY_DIR, exist_ok=True)
    if args.german:
        training_data = get_training_samples('German', args.minarticles, GermanDatabase)
    else:
        training_data = (get_training_samples('Human', args.minarticles, DatabaseAuthorship))
        training_data.extend(get_training_samples('Machine', args.minarticles, DatabaseGenArticles))
    print(f"number of training articles:{len(training_data)}")

    char_grams = [char_trigrams(article_tuple[0]) for article_tuple in training_data]
    if args.german:
        parser = CoreNLPDependencyParser(url="http://localhost:9001")
        sem_grams = [sem_trigrams(article_tuple[0], parser, 'german') for article_tuple in training_data]
        file_appendix = '_german'
    else:
        parser = CoreNLPDependencyParser(url="http://localhost:9000")
        sem_grams = [sem_trigrams(article_tuple[0], parser) for article_tuple in training_data]
        file_appendix = ''
    character_distribution = trigram_distribution(char_grams, n_features)
    semantic_distribution = trigram_distribution(sem_grams, n_features)
    character_distribution.to_csv(os.path.join(STYLOMETRY_DIR, f"char_distribution{n_features}{file_appendix}.csv"))
    semantic_distribution.to_csv(os.path.join(STYLOMETRY_DIR, f"sem_distribution{n_features}{file_appendix}.csv"))
    print("Distributions Done\n\n")

    char, sem = [], []
    for i, author in enumerate(used_authors.keys()):
        if args.german and author in ['gtp2', 'gtp3', 'gpt3-phrase', 'grover']:
            continue  # not supported in german
        printProgressBar(i, len(used_authors.keys()) - 1)
        truth_table = [1 if author == article_tuple[1] else 0 for article_tuple in training_data]
        char.append(
            fit_model(f"{author.replace('/', '_')}_char{file_appendix}", character_distribution.values, truth_table,
                      n_features))
        sem.append(
            fit_model(f"{author.replace('/', '_')}_sem{file_appendix}", semantic_distribution.values, truth_table,
                      n_features))
    truth_table = [1 if used_authors[article_tuple[1]] == 'ai' else 0 for article_tuple in
                   training_data]
    char_results, sem_results = [], []
    for iterrow in character_distribution.iterrows():
        char_results.append([c.predict_proba(iterrow[1].values.reshape(1, -1))[0][1] for c in char])
    for iterrow in semantic_distribution.iterrows():
        sem_results.append([s.predict_proba(iterrow[1].values.reshape(1, -1))[0][1] for s in sem])

    char_results = normalize(f"char{file_appendix}", char_results)
    sem_results = normalize(f"sem{file_appendix}", sem_results)
    model_results = []
    for i in range(len(training_data)):
        model_results.append((char_results[i], sem_results[i]))

    features = pd.DataFrame([char for char, sem in model_results])
    fit_model(f"char_final{file_appendix}", features.values, truth_table, n_features, 'forest')
    features = pd.DataFrame([sem for char, sem in model_results])
    fit_model(f"sem_final{file_appendix}", features.values, truth_table, n_features, 'forest')
    features = pd.DataFrame([char + sem for char, sem in model_results])
    fit_model(f"style_final{file_appendix}", features.values, truth_table, n_features, 'forest')
    print("TRAINING DONE!")
