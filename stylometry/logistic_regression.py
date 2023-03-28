import os.path
import pickle

import numpy as np
import pandas as pd
from definitions import STYLOMETRY_DIR
from stylometry.char_trigrams import char_trigrams
from stylometry.semantic_trigrams import sem_trigrams
from sklearn.linear_model import LogisticRegression
from nltk.parse.corenlp import CoreNLPDependencyParser

authors = ["gpt3",
           "grover",
           "https:__www.theguardian.com_profile_hannah-ellis-petersen",
           "https:__www.theguardian.com_profile_leyland-cecco",
           "https:__www.theguardian.com_profile_martin-chulov"
           ]


def _most_common_trigrams(trigram_lists: list[dict], max_features: int):
    features = set((feature for trigrams in trigram_lists for feature in trigrams.keys()))
    feature_occurances = {
        feature: sum((0 if feature not in trigrams else trigrams[feature] for trigrams in trigram_lists))
        for feature in features}

    items = list(feature_occurances.items())
    items.sort(key=lambda e: e[1], reverse=True)
    return [key for key, _ in items[:max_features]]


def trigram_distribution(trigram_lists: list[dict], max_features: int = 10):
    features = _most_common_trigrams(trigram_lists, max_features)

    return fixed_trigram_distribution(trigram_lists, features)


def fixed_trigram_distribution(trigram_lists, features):
    values = list()
    for trigrams in trigram_lists:
        count = sum(c for c in trigrams.values())
        values.append(np.transpose(np.array([
            0 if feature not in trigrams else trigrams[feature] / count
            for feature in features])))

    return pd.DataFrame(values, columns=features)


def logistic_regression(trigram_dataframe: pd.DataFrame, truth_labels: list()):
    if len(trigram_dataframe) <= 1:
        return
    regression = LogisticRegression(random_state=42)
    return regression.fit(trigram_dataframe, truth_labels)


def predict_author(text: str, n_features: int = 100):
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    char_features = list(pd.read_csv(os.path.join(STYLOMETRY_DIR, "char_distribution" + str(n_features) + ".csv")).columns)[1:]
    sem_features = [eval(feature) for feature in list(pd.read_csv(os.path.join(STYLOMETRY_DIR, "sem_distribution" + str(n_features) + ".csv")).columns)[1:]]
    char_grams = char_trigrams(text)
    sem_grams = sem_trigrams(text, parser)
    char_distribution = fixed_trigram_distribution([char_grams], char_features)
    sem_distribution = fixed_trigram_distribution([sem_grams], sem_features)
    char_confidence = {}
    sem_confidence = {}
    for author in authors:
        with open(os.path.join(STYLOMETRY_DIR, author + "_char" + str(n_features) + ".pickle"), "rb") as fp:
            char_confidence[author] = pickle.load(fp).predict_proba(char_distribution)
        with open(os.path.join(STYLOMETRY_DIR, author + "_sem" + str(n_features) + ".pickle"), "rb") as fp:
            sem_confidence[author] = pickle.load(fp).predict_proba(sem_distribution)
    machine = any(sem_confidence[author][0][1] > 0.251 for author in authors[:2])
    human = any(sem_confidence[author][0][1] > 0.13 for author in authors[2:])
    if machine == human:
        return 0
    elif machine:
        return 1
    elif human:
        return -1
