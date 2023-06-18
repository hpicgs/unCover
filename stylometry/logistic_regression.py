import os.path
import pickle

import numpy as np
import pandas as pd
from definitions import STYLOMETRY_DIR, CHAR_MACHINE_CONFIDENCE, CHAR_HUMAN_CONFIDENCE, SEM_MACHINE_CONFIDENCE, SEM_HUMAN_CONFIDENCE
from stylometry.char_trigrams import char_trigrams
from stylometry.semantic_trigrams import sem_trigrams
from sklearn.linear_model import LogisticRegression
from nltk.parse.corenlp import CoreNLPDependencyParser

used_authors = {
    "gpt2":"ai",
    "gpt3":"ai",
    "gpt3-phrase":"ai",
    "grover":"ai",
    "https:__www.theguardian.com_profile_hannah-ellis-petersen":"human",
    "https:__www.theguardian.com_profile_leyland-cecco":"human",
    "https:__www.theguardian.com_profile_martin-chulov":"human",
    "https:__www.theguardian.com_profile_julianborger":"human",
    "https:__www.theguardian.com_profile_helen-sullivan":"human"
}


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
    regression = LogisticRegression(solver='liblinear', max_iter=100, random_state=42)
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
    authors = used_authors.keys()
    for author in authors:
        with open(os.path.join(STYLOMETRY_DIR, author + "_char" + str(n_features) + ".pickle"), "rb") as fp:
            char_confidence[author] = pickle.load(fp).predict_proba(char_distribution)
        with open(os.path.join(STYLOMETRY_DIR, author + "_sem" + str(n_features) + ".pickle"), "rb") as fp:
            sem_confidence[author] = pickle.load(fp).predict_proba(sem_distribution)

    machine_char = any(char_confidence[author][0][1] > CHAR_MACHINE_CONFIDENCE for author in authors if used_authors[author] == "ai")
    human_char = any(char_confidence[author][0][1] > CHAR_HUMAN_CONFIDENCE for author in authors if used_authors[author] == "human")
    machine_sem = any(sem_confidence[author][0][1] > SEM_MACHINE_CONFIDENCE for author in authors if used_authors[author] == "ai")
    human_sem = any(sem_confidence[author][0][1] > SEM_HUMAN_CONFIDENCE for author in authors if used_authors[author] == "human")

    char = 0
    if machine_char == human_char:
        char = 0
    elif machine_char:
        char = 1
    elif human_char:
        char = -1
    sem = 0
    if machine_sem == human_sem:
        sem = 0
    elif machine_sem:
        sem = 1
    elif human_sem:
        sem = -1
    return [char, sem]
