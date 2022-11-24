import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def __most_common_trigrams(trigram_lists: list[dict], max_features: int):
    features = set((feature for trigrams in trigram_lists for feature in trigrams.keys()))
    feature_occurances = {
        feature: sum((0 if feature not in trigrams else trigrams[feature] for trigrams in trigram_lists))
    for feature in features }

    items = list(feature_occurances.items())
    items.sort(key=lambda e: e[1], reverse=True)
    return [key for key, _ in items[:max_features]]

def trigram_distribution(trigram_lists: list[dict], max_features: int = 10):
    features = __most_common_trigrams(trigram_lists, max_features)

    values = list()
    for trigrams in trigram_lists:
        count = sum(c for c in trigrams.values())
        values.append(np.transpose(np.array([
            0 if feature not in trigrams else trigrams[feature] / count
        for feature in features])))

    return pd.DataFrame(values, columns=features)

def trigram_distribution_from_existing(trigrams, distribution: pd.DataFrame):
    count = sum(c for _, c in trigrams.values())
    
    return pd.DataFrame([
        trigrams[col] / count for col in distribution.columns
    ], columns=distribution.columns)

def logit_predictor(distribution):
    model = LogisticRegression(random_state=42)
    return model.fit(distribution, [1])
