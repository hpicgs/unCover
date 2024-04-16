import os.path
import json
import pickle
import numpy as np
import pandas as pd
from definitions import STYLOMETRY_DIR, STYLE_MACHINE_CONFIDENCE, STYLE_HUMAN_CONFIDENCE
from stylometry.char_trigrams import char_trigrams
from stylometry.syntactic_trigrams import syn_trigrams
from stylometry.word_trigrams import word_trigrams
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
    'human5': 'human',
    'human6': 'human',
    'human7': 'human'
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
            0 if feature not in trigrams else 10 * trigrams[feature] / count
            for feature in features])))

    return pd.DataFrame(values, columns=features)


def fit_model(model, df: pd.DataFrame, truth_labels: list[int]):
    if len(df) <= 1:
        return
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=9732)
    n_scores = cross_val_score(model, df, truth_labels, scoring='accuracy', cv=cv, n_jobs=-1)
    print("Mean Accuracy: %.3f (%.3f)" % (np.mean(n_scores), np.std(n_scores)))
    m = model.fit(df, truth_labels)
    print(f"final regression score: {m.score(df, truth_labels)}")
    return m


def logistic_regression(df: pd.DataFrame, truth_labels: list[int]):
    return fit_model(LogisticRegression(solver='liblinear', max_iter=100, random_state=42), df, truth_labels)


def random_forest(df: pd.DataFrame, truth_labels: list[int]):
    return fit_model(RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42), df, truth_labels)


def predict_author(text: str, n_features: int = 100, file_appendix: str = '') -> list[int]:
    if file_appendix == '':
        parser = CoreNLPDependencyParser(url="http://localhost:9000")
        syn_grams = syn_trigrams(text, parser)
    elif file_appendix == '_german':
        parser = CoreNLPDependencyParser(url="http://localhost:9001")
        syn_grams = syn_trigrams(text, parser, 'german')
    else:
        raise ValueError("file_appendix must be either '' or '_german'")
    char_features = list(
        pd.read_csv(os.path.join(STYLOMETRY_DIR, f"char_distribution{n_features}{file_appendix}.csv")).columns)[1:]
    syn_features = [eval(feature) for feature in list(
        pd.read_csv(os.path.join(STYLOMETRY_DIR, f"syn_distribution{n_features}{file_appendix}.csv")).columns)[1:]]
    word_features = list(
        pd.read_csv(os.path.join(STYLOMETRY_DIR, f"word_distribution{n_features}{file_appendix}.csv")).columns)[1:]
    char_grams = char_trigrams(text)
    char_distribution = fixed_trigram_distribution([char_grams], char_features)
    syn_distribution = fixed_trigram_distribution([syn_grams], syn_features)
    word_grams = word_trigrams(text, True if (file_appendix == '_german') else False)
    word_distribution = fixed_trigram_distribution([word_grams], word_features)
    char_confidence, syn_confidence, word_confidence = [], [], []
    authors = used_authors.keys()
    char_min, char_max = json.loads(
        open(os.path.join(STYLOMETRY_DIR, f"char{file_appendix}_normalization.json"), 'rb').read())
    syn_min, syn_max = json.loads(
        open(os.path.join(STYLOMETRY_DIR, f"syn{file_appendix}_normalization.json"), 'rb').read())
    word_min, word_max = json.loads(
        open(os.path.join(STYLOMETRY_DIR, f"word{file_appendix}_normalization.json"), 'rb').read())
    for i, author in enumerate(authors):
        with open(os.path.join(STYLOMETRY_DIR, author + f"_char{n_features}{file_appendix}.pickle"), 'rb') as fp:
            char_confidence.append(
                (pickle.load(fp).predict_proba(char_distribution.values)[0][1] - char_min[i]) / (
                        char_max[i] - char_min[i]))
        with open(os.path.join(STYLOMETRY_DIR, author + f"_syn{n_features}{file_appendix}.pickle"), 'rb') as fp:
            syn_confidence.append(
                (pickle.load(fp).predict_proba(syn_distribution.values)[0][1] - syn_min[i]) / (syn_max[i] - syn_min[i]))
        with open(os.path.join(STYLOMETRY_DIR, author + f"_word{n_features}{file_appendix}.pickle"), 'rb') as fp:
            word_confidence.append(
                (pickle.load(fp).predict_proba(word_distribution.values)[0][1] - word_min[i]) / (
                        word_max[i] - word_min[i]))

    with open(os.path.join(STYLOMETRY_DIR, f"char_final{n_features}{file_appendix}.pickle"), 'rb') as fp:
        char = pickle.load(fp).predict_proba(np.array(char_confidence).reshape(1, -1))[0]
    with open(os.path.join(STYLOMETRY_DIR, f"syn_final{n_features}{file_appendix}.pickle"), 'rb') as fp:
        syn = pickle.load(fp).predict_proba(np.array(syn_confidence).reshape(1, -1))[0]
    with open(os.path.join(STYLOMETRY_DIR, f"word_final{n_features}{file_appendix}.pickle"), 'rb') as fp:
        word = pickle.load(fp).predict_proba(np.array(word_confidence).reshape(1, -1))[0]
    with open(os.path.join(STYLOMETRY_DIR, f"style_final{n_features}{file_appendix}.pickle"), 'rb') as fp:
        style = pickle.load(fp).predict_proba(
            np.array((char_confidence + syn_confidence + word_confidence)).reshape(1, -1))[0]

    char = 1 if char[1] > STYLE_MACHINE_CONFIDENCE else -1 if char[0] > STYLE_HUMAN_CONFIDENCE else 0
    syn = 1 if syn[1] > STYLE_MACHINE_CONFIDENCE else -1 if syn[0] > STYLE_HUMAN_CONFIDENCE else 0
    word = 1 if word[1] > STYLE_MACHINE_CONFIDENCE else -1 if word[0] > STYLE_HUMAN_CONFIDENCE else 0
    style = 1 if style[1] > STYLE_MACHINE_CONFIDENCE else -1 if style[0] > STYLE_HUMAN_CONFIDENCE else 0

    return [char, syn, word, style]
