import os.path
import json
import pickle
import numpy as np
import pandas as pd
from misc.definitions import STYLOMETRY_DIR, CHAR_MACHINE_CONFIDENCE, CHAR_HUMAN_CONFIDENCE, SEM_MACHINE_CONFIDENCE, \
    SEM_HUMAN_CONFIDENCE, STYLE_MACHINE_CONFIDENCE, STYLE_HUMAN_CONFIDENCE
from stylometry.char_trigrams import char_trigrams
from stylometry.semantic_trigrams import sem_trigrams
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from nltk.parse.corenlp import CoreNLPDependencyParser

used_authors = {
    'gpt2': 'ai',
    'gpt3': 'ai',
    'gpt4': 'ai',
    'gpt3-phrase': 'ai',
    'grover': 'ai',
    'gemini': 'ai',
    'human1': 'human',
    'human2': 'human',
    'human3': 'human',
    'human4': 'human',
    'human5': 'human'
}


def _most_common_trigrams(trigram_lists: list[dict], max_features: int):
    features = set((feature for trigrams in trigram_lists for feature in trigrams.keys()))
    feature_occurances = {
        feature: sum((0 if feature not in trigrams else trigrams[feature] for trigrams in trigram_lists))
        for feature in features}

    items = list(feature_occurances.items())
    items.sort(key=lambda e: e[1], reverse=True)
    return [key for key, _ in items[:max_features]]


def trigram_distribution(trigram_lists: list[dict], max_features: int = 100):
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


def logistic_regression(dataframe: pd.DataFrame, truth_labels: list[int]):
    if len(dataframe) <= 1:
        return
    regression = LogisticRegression(solver='liblinear', max_iter=100, random_state=42, C=0.9)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=9732)
    n_scores = cross_val_score(regression, dataframe, truth_labels, scoring='accuracy', cv=cv, n_jobs=-1)
    print("Mean Accuracy: %.3f (%.3f)" % (np.mean(n_scores), np.std(n_scores)))
    reg = regression.fit(dataframe, truth_labels)
    print(f"final regression score: {reg.score(dataframe, truth_labels)}")
    return reg


def predict_author(text: str, n_features: int = 100, file_appendix: str = '') -> list[int]:
    if file_appendix == '':
        parser = CoreNLPDependencyParser(url="http://localhost:9000")
        sem_grams = sem_trigrams(text, parser)
    elif file_appendix == '_german':
        parser = CoreNLPDependencyParser(url="http://localhost:9001")
        sem_grams = sem_trigrams(text, parser, 'german')
    else:
        raise ValueError("file_appendix must be either '' or '_german'")
    char_features = list(
        pd.read_csv(os.path.join(STYLOMETRY_DIR, f"char_distribution{n_features}{file_appendix}.csv")).columns)[1:]
    sem_features = [eval(feature) for feature in list(
        pd.read_csv(os.path.join(STYLOMETRY_DIR, f"sem_distribution{n_features}{file_appendix}.csv")).columns)[1:]]
    char_grams = char_trigrams(text)
    char_distribution = fixed_trigram_distribution([char_grams], char_features)
    sem_distribution = fixed_trigram_distribution([sem_grams], sem_features)
    char_confidence = []
    sem_confidence = []
    authors = used_authors.keys()
    with open(os.path.join(STYLOMETRY_DIR, f"char{file_appendix}_normalization.json"), 'rb') as fp:
        char_min, char_max = json.loads(fp.read())
    with open(os.path.join(STYLOMETRY_DIR, f"sem{file_appendix}_normalization.json"), 'rb') as fp:
        sem_min, sem_max = json.loads(fp.read())
    for i, author in enumerate(authors):
        with open(os.path.join(STYLOMETRY_DIR, author + f"_char{n_features}{file_appendix}.pickle"), 'rb') as fp:
            char_confidence.append(
                (pickle.load(fp).predict_proba(char_distribution.values)[0][1]-char_min[i])/(char_max[i]-char_min[i]))
        with open(os.path.join(STYLOMETRY_DIR, author + f"_sem{n_features}{file_appendix}.pickle"), 'rb') as fp:
            sem_confidence.append(
                (pickle.load(fp).predict_proba(sem_distribution.values)[0][1]-sem_min[i])/(sem_max[i]-sem_min[i]))

    with open(os.path.join(STYLOMETRY_DIR, f"char_final{n_features}{file_appendix}.pickle"), 'rb') as fp:
        char = pickle.load(fp).predict_proba(np.array(char_confidence).reshape(1, -1))[0]
    with open(os.path.join(STYLOMETRY_DIR, f"sem_final{n_features}{file_appendix}.pickle"), 'rb') as fp:
        sem = pickle.load(fp).predict_proba(np.array(sem_confidence).reshape(1, -1))[0]
    with open(os.path.join(STYLOMETRY_DIR, f"style_final{n_features}{file_appendix}.pickle"), 'rb') as fp:
        style = pickle.load(fp).predict_proba(np.array((char_confidence + sem_confidence)).reshape(1, -1))[0]

    char = 1 if char[1] > CHAR_MACHINE_CONFIDENCE else -1 if char[0] > CHAR_HUMAN_CONFIDENCE else 0
    sem = 1 if sem[1] > SEM_MACHINE_CONFIDENCE else -1 if sem[0] > SEM_HUMAN_CONFIDENCE else 0
    style = 1 if style[1] > STYLE_MACHINE_CONFIDENCE else -1 if style[0] > STYLE_HUMAN_CONFIDENCE else 0

    return [char, sem, style]
