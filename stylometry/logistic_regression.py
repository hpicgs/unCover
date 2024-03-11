import os.path, json
import pickle

import numpy as np
import pandas as pd
from definitions import STYLOMETRY_DIR, CHAR_MACHINE_CONFIDENCE, CHAR_HUMAN_CONFIDENCE, SEM_MACHINE_CONFIDENCE, \
    SEM_HUMAN_CONFIDENCE, STYLE_MACHINE_CONFIDENCE, STYLE_HUMAN_CONFIDENCE
from stylometry.char_trigrams import char_trigrams
from stylometry.semantic_trigrams import sem_trigrams
from sklearn.linear_model import LogisticRegression
from nltk.parse.corenlp import CoreNLPDependencyParser

used_authors = {
    "gpt2": "ai",
    "gpt3": "ai",
    "gpt4": "ai",
    "gpt3-phrase": "ai",
    "grover": "ai",
    "gemini": "ai",
    "human1": "human",
    "human2": "human",
    "human3": "human",
    "human4": "human",
    "human5": "human"
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


def logistic_regression(dataframe: pd.DataFrame, truth_labels: list()):
    if len(dataframe) <= 1:
        return
    regression = LogisticRegression(solver='liblinear', max_iter=100, random_state=42)
    reg = regression.fit(dataframe, truth_labels)
    print("regression score: " + str(reg.score(dataframe, truth_labels)))
    return reg


def predict_author(text: str, n_features: int = 100):
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    char_features = list(
        pd.read_csv(os.path.join(STYLOMETRY_DIR, "char_distribution" + str(n_features) + ".csv")).columns)[1:]
    sem_features = [eval(feature) for feature in list(
        pd.read_csv(os.path.join(STYLOMETRY_DIR, "sem_distribution" + str(n_features) + ".csv")).columns)[1:]]
    char_grams = char_trigrams(text)
    sem_grams = sem_trigrams(text, parser)
    char_distribution = fixed_trigram_distribution([char_grams], char_features)
    sem_distribution = fixed_trigram_distribution([sem_grams], sem_features)
    char_confidence = []
    sem_confidence = []
    authors = used_authors.keys()
    with open(os.path.join(STYLOMETRY_DIR, "char_normalization.json"), "rb") as fp:
        char_min, char_max = json.loads(fp.read())
    with open(os.path.join(STYLOMETRY_DIR, "sem_normalization.json"), "rb") as fp:
        sem_min, sem_max = json.loads(fp.read())
    for i, author in enumerate(authors):
        with open(os.path.join(STYLOMETRY_DIR, author + "_char" + str(n_features) + ".pickle"), "rb") as fp:
            char_confidence.append(
                (pickle.load(fp).predict_proba(char_distribution)[0][1] - char_min[i] / (char_max[i] - char_min[i])))
        with open(os.path.join(STYLOMETRY_DIR, author + "_sem" + str(n_features) + ".pickle"), "rb") as fp:
            sem_confidence.append(
                (pickle.load(fp).predict_proba(sem_distribution)[0][1] - sem_min[i] / (sem_max[i] - sem_min[i])))

    with open(os.path.join(STYLOMETRY_DIR, "char_final" + str(n_features) + ".pickle"), "rb") as fp:
        char = pickle.load(fp).predict_proba(np.array(char_confidence).reshape(1, -1))[0]
    with open(os.path.join(STYLOMETRY_DIR, "sem_final" + str(n_features) + ".pickle"), "rb") as fp:
        sem = pickle.load(fp).predict_proba(np.array(sem_confidence).reshape(1, -1))[0]
    with open(os.path.join(STYLOMETRY_DIR, "style_final" + str(n_features) + ".pickle"), "rb") as fp:
        style = pickle.load(fp).predict_proba(np.array((char_confidence + sem_confidence)).reshape(1, -1))[0]

    char = 1 if char[1] > CHAR_MACHINE_CONFIDENCE else -1 if char[0] > CHAR_HUMAN_CONFIDENCE else 0
    sem = 1 if sem[1] > SEM_MACHINE_CONFIDENCE else -1 if sem[0] > SEM_HUMAN_CONFIDENCE else 0
    style = 1 if style[1] > STYLE_MACHINE_CONFIDENCE else -1 if style[0] > STYLE_HUMAN_CONFIDENCE else 0

    return [char, sem, style]
