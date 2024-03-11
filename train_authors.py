import pandas as pd
from nltk.parse.corenlp import CoreNLPDependencyParser

from definitions import STYLOMETRY_DIR, DATABASE_AUTHORS_PATH, DATABASE_GEN_PATH
from stylometry.char_trigrams import char_trigrams
from stylometry.semantic_trigrams import sem_trigrams
from stylometry.logistic_regression import trigram_distribution, logistic_regression, used_authors
from database.mock_database import DatabaseAuthorship, DatabaseGenArticles
import os, sys, pickle, argparse, json


def normalize(type, f):
    maxi, mini = [], []
    for i in range(len(f[0])):
        ma = max([f[j][i] for j in range(len(f))]) + 0.01
        mi = min([f[j][i] for j in range(len(f))]) - 0.01
        for j in range(len(f)):
            f[j][i] = (f[j][i] - mi) / (ma - mi)
        maxi.append(ma)
        mini.append(mi)
    with open(os.path.join(STYLOMETRY_DIR, f"{type}_normalization.json"), "w") as fp:
        json.dump([mini, maxi], fp)
    return f


def get_training_samples(type, min_articles, training_data, database):
    authors = database.get_authors()
    print(f"{type} authors:")
    print(authors)
    trainable = []
    for author in authors:
        full_article_list = [(article["text"], author) for article in
                             database.get_articles_by_author(author)]
        training_data += full_article_list[:int(len(full_article_list) * 0.8)]
        if len(full_article_list) >= min_articles:
            print(f"chose author: {author}")
            trainable.append(author)
    print(f"trainable {type} authors:")
    print(trainable)
    return training_data


def fit_model(name, features, truth_table, n):
    print(f"Training {name}...")
    with open(os.path.join(STYLOMETRY_DIR, f"{name}{n}.pickle"), "wb") as f:
        model = logistic_regression(features, truth_table)
        pickle.dump(model, f)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nfeatures", action="store", required=False, type=int, default=100,
                        help="number of char trigram & semantic trigram features used in the distribution")
    parser.add_argument("--minarticles", action="store", required=False, type=int, default=50,
                        help="minimum number of articles required for training on a specific author/model")
    args = parser.parse_args()
    n_features = args.nfeatures

    if not os.path.isfile(DATABASE_AUTHORS_PATH):
        print("Error: no database for human authors was provided")
        sys.exit(1)
    if not os.path.isfile(DATABASE_GEN_PATH):
        print("Error: no database for machine authors was provided")
        sys.exit(1)

    os.makedirs(STYLOMETRY_DIR, exist_ok=True)
    training_data = []
    training_data = get_training_samples("Human", args.minarticles, training_data, DatabaseAuthorship)
    training_data = get_training_samples("Machine", args.minarticles, training_data, DatabaseGenArticles)
    print("number of training articles:", len(training_data))

    parser = CoreNLPDependencyParser(url="http://localhost:9000")
    char_grams = [char_trigrams(article_tuple[0]) for article_tuple in training_data]
    sem_grams = [sem_trigrams(article_tuple[0], parser) for article_tuple in training_data]
    character_distribution = trigram_distribution(char_grams, n_features)
    semantic_distribution = trigram_distribution(sem_grams, n_features)
    character_distribution.to_csv(os.path.join(STYLOMETRY_DIR, f"char_distribution{n_features}.csv"))
    semantic_distribution.to_csv(os.path.join(STYLOMETRY_DIR, f"sem_distribution{n_features}.csv"))
    print("Distributions Done\n\n")

    char, sem = [], []
    for author in used_authors.keys():
        truth_table = [1 if author == article_tuple[1] else 0 for article_tuple in training_data]
        char.append(fit_model(f"{author.replace('/', '_')}_char", character_distribution.values, truth_table, n_features))
        sem.append(fit_model(f"{author.replace('/', '_')}_sem", semantic_distribution.values, truth_table, n_features))
    truth_table = [1 if article_tuple[1] != '' and used_authors[article_tuple[1]] == "ai" else 0 for article_tuple in
                   training_data]
    char_results, sem_results = [], []
    for iterrow in character_distribution.iterrows():
        char_results.append([c.predict_proba(iterrow[1].values.reshape(1, -1))[0][1] for c in char])
    for iterrow in semantic_distribution.iterrows():
        sem_results.append([s.predict_proba(iterrow[1].values.reshape(1, -1))[0][1] for s in sem])

    char_results = normalize("char", char_results)
    sem_results = normalize("sem", sem_results)
    model_results = []
    for i in range(len(training_data)):
        model_results.append((char_results[i], sem_results[i]))

    features = pd.DataFrame([char for char, sem in model_results])
    fit_model("char_final", features, truth_table, n_features)
    features = pd.DataFrame([sem for char, sem in model_results])
    fit_model("sem_final", features, truth_table, n_features)
    features = pd.DataFrame([char + sem for char, sem in model_results])
    fit_model("style_final", features, truth_table, n_features)
    print("TRAINING DONE!")
