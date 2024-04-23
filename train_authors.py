import os
import pickle
import argparse
import json
import pandas as pd
from nltk.parse.corenlp import CoreNLPDependencyParser
from definitions import STYLOMETRY_DIR
from stylometry.char_trigrams import char_trigrams
from stylometry.syntactic_trigrams import syn_trigrams
from stylometry.word_trigrams import word_trigrams
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
                        help="number of char trigram & syntactic trigram features used in the distribution")
    parser.add_argument('--minarticles', action='store', required=False, type=int, default=50,
                        help="minimum number of articles required for training on a specific author/model")
    parser.add_argument('--german', action='store_true', required=False,
                        help="use the german test database instead of the english one")
    parser.add_argument('--stored', action='store', required=False,
                        help="use previously created trigram distributions")
    args = parser.parse_args()
    n_features = args.nfeatures

    os.makedirs(STYLOMETRY_DIR, exist_ok=True)
    if args.german:
        training_data = get_training_samples('German', args.minarticles, GermanDatabase)
    else:
        training_data = (get_training_samples('Human', args.minarticles, DatabaseAuthorship))
        training_data.extend(get_training_samples('Machine', args.minarticles, DatabaseGenArticles))
    print(f"number of training articles:{len(training_data)}")

    if args.stored:
        if args.german:
            file_appendix = '_german'
        else:
            file_appendix = ''
        character_distribution = pd.read_csv(
            os.path.join(STYLOMETRY_DIR, f"char_distribution{n_features}{file_appendix}.csv"),
            index_col=0)
        syntactic_distribution = pd.read_csv(
            os.path.join(STYLOMETRY_DIR, f"syn_distribution{n_features}{file_appendix}.csv"),
            index_col=0)
        word_distribution = pd.read_csv(
            os.path.join(STYLOMETRY_DIR, f"word_distribution{n_features}{file_appendix}.csv"),
            index_col=0)
    else:
        char_grams = []
        for i, article_tuple in enumerate(training_data):
            printProgressBar(i, len(training_data) - 1, fill='█')
            char_grams.append(char_trigrams(article_tuple[0]))
        syn_grams = []
        if args.german:
            parser = CoreNLPDependencyParser(url="http://localhost:9001")
            for i, article_tuple in enumerate(training_data):
                printProgressBar(i, len(training_data) - 1, fill='█')
                syn_grams.append(syn_trigrams(article_tuple[0], parser, 'german'))
            file_appendix = '_german'
        else:
            parser = CoreNLPDependencyParser(url="http://localhost:9000")
            for i, article_tuple in enumerate(training_data):
                printProgressBar(i, len(training_data) - 1, fill='█')
                syn_grams.append(syn_trigrams(article_tuple[0], parser))
            file_appendix = ''
        word_grams = []
        for i, article_tuple in enumerate(training_data):
            printProgressBar(i, len(training_data) - 1, fill='█')
            word_grams.append(word_trigrams(article_tuple[0], args.german))
        character_distribution = trigram_distribution(char_grams, n_features)
        syntactic_distribution = trigram_distribution(syn_grams, n_features)
        word_distribution = trigram_distribution(word_grams, n_features)
        character_distribution.to_csv(os.path.join(STYLOMETRY_DIR, f"char_distribution{n_features}{file_appendix}.csv"))
        syntactic_distribution.to_csv(os.path.join(STYLOMETRY_DIR, f"syn_distribution{n_features}{file_appendix}.csv"))
        word_distribution.to_csv(os.path.join(STYLOMETRY_DIR, f"word_distribution{n_features}{file_appendix}.csv"))
    print("Distributions Done\n\n")

    char, syn, word = [], [], []
    for i, author in enumerate(used_authors.keys()):
        if args.german and author in ['gpt2', 'gpt3', 'gpt3-phrase', 'grover', 'human6', 'human7']:
            continue  # not supported in german
        printProgressBar(i, len(used_authors.keys()) - 1)
        truth_table = [1 if author == article_tuple[1] else 0 for article_tuple in training_data]
        char.append(
            fit_model(f"{author.replace('/', '_')}_char{file_appendix}", character_distribution.values, truth_table,
                      n_features))
        syn.append(
            fit_model(f"{author.replace('/', '_')}_syn{file_appendix}", syntactic_distribution.values, truth_table,
                      n_features))
        word.append(
            fit_model(f"{author.replace('/', '_')}_word{file_appendix}", word_distribution.values, truth_table,
                      n_features))
    truth_table = [1 if used_authors[article_tuple[1]] == 'ai' else 0 for article_tuple in
                   training_data]
    char_results, syn_results, word_results = [], [], []
    for iterrow in character_distribution.iterrows():
        char_results.append([c.predict_proba(iterrow[1].values.reshape(1, -1))[0][1] for c in char])
    for iterrow in syntactic_distribution.iterrows():
        syn_results.append([s.predict_proba(iterrow[1].values.reshape(1, -1))[0][1] for s in syn])
    for iterrow in word_distribution.iterrows():
        word_results.append([w.predict_proba(iterrow[1].values.reshape(1, -1))[0][1] for w in word])

    char_results = normalize(f"char{file_appendix}", char_results)
    syn_results = normalize(f"syn{file_appendix}", syn_results)
    word_results = normalize(f"word{file_appendix}", word_results)
    model_results = []
    for i in range(len(training_data)):
        model_results.append((char_results[i], syn_results[i], word_results[i]))

    features = pd.DataFrame([char for char, _, _ in model_results])
    fit_model(f"char_final{file_appendix}", features.values, truth_table, n_features, 'forest')
    features = pd.DataFrame([syn for _, syn, _ in model_results])
    fit_model(f"syn_final{file_appendix}", features.values, truth_table, n_features, 'forest')
    features = pd.DataFrame([word for _, _, word in model_results])
    fit_model(f"word_final{file_appendix}", features.values, truth_table, n_features, 'forest')
    features = pd.DataFrame([char + syn + word for char, syn, word in model_results])
    fit_model(f"style_final{file_appendix}", features.values, truth_table, n_features, 'forest')
    print("TRAINING DONE!")
