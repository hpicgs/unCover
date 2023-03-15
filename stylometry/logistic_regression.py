import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _most_common_trigrams(trigram_lists: list[dict], max_features: int):
    features = set((feature for trigrams in trigram_lists for feature in trigrams.keys()))
    feature_occurances = {
        feature: sum((0 if feature not in trigrams else trigrams[feature] for trigrams in trigram_lists))
    for feature in features }

    items = list(feature_occurances.items())
    items.sort(key=lambda e: e[1], reverse=True)
    return [key for key, _ in items[:max_features]]

def trigram_distribution(trigram_lists: list[dict], max_features: int = 10):
    features = _most_common_trigrams(trigram_lists, max_features)

    values = list()
    for trigrams in trigram_lists:
        count = sum(c for c in trigrams.values())
        values.append(np.transpose(np.array([
            0 if feature not in trigrams else trigrams[feature] / count
        for feature in features])))

    return pd.DataFrame(values, columns=features)

def fixed_trigram_distribution(trigram_lists: list[dict], features):
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
    regression = LogisticRegression(solver='liblinear', max_iter=50, random_state=42)
    return regression.fit(trigram_dataframe, truth_labels)
