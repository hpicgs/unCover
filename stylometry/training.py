import json
import os
import argparse
import pickle

import numpy as np
import pandas as pd
from nltk.parse.corenlp import CoreNLPDependencyParser
from definitions import STYLOMETRY_DIR
from stylometry.char_trigrams import char_trigrams
from stylometry.syntactic_trigrams import syn_trigrams
from stylometry.word_trigrams import word_trigrams
from stylometry.classifier import trigram_distribution, used_authors, logistic_regression, random_forest
from misc.logger import printProgressBar


def load_distributions(appendix: str) -> tuple:
    file_path = os.path.join(STYLOMETRY_DIR, f"{appendix}.csv")
    character_distribution = pd.read_csv(file_path.format('char'), index_col=0)
    syntactic_distribution = pd.read_csv(file_path.format('syn'), index_col=0)
    word_distribution = pd.read_csv(file_path.format('word'), index_col=0)
    return character_distribution, syntactic_distribution, word_distribution


def save_distributions(appendix: str, char_grams: list, syn_grams: list, word_grams: list) -> tuple:
    distributions = [trigram_distribution(grams) for grams in [char_grams, syn_grams, word_grams]]
    for dist, name in zip(distributions, ['char', 'syn', 'word']):
        dist.to_csv(os.path.join(STYLOMETRY_DIR, f"{name}_distribution{appendix}.csv"))
    return tuple(distributions)


def preprocess_stylometry(data: dict, appendix: str, args: argparse.Namespace) -> dict:
    if args.use_stored:
        character_distribution, syntactic_distribution, word_distribution = load_distributions(appendix)
    else:
        corenlp = CoreNLPDependencyParser(url="http://localhost:9001" if args.german else "http://localhost:9000")
        char_grams, syn_grams, word_grams = [], [], []
        count, processed = sum(len(a) for a in data.values()), 0
        for label, articles in data.items():
            for article in articles:
                printProgressBar(processed, count - 1, fill='█')
                char_grams.append(char_trigrams(article))
                syn_grams.append(syn_trigrams(article, corenlp))
                word_grams.append(word_trigrams(article, args.german))
                processed += 1

        character_distribution, syntactic_distribution, word_distribution = save_distributions(appendix, char_grams,
                                                                                               syn_grams, word_grams)

    print("Distributions Done...")
    return {'char': character_distribution, 'syn': syntactic_distribution, 'word': word_distribution}


def fit_normalize_interm_results(name: str, f: list[list[float]]) -> list[list[float]]:
    max_values, min_values = [], []
    for i in range(len(f[0])):
        column_values = [f[j][i] for j in range(len(f))]
        max_value = max(column_values) + 0.01
        min_value = min(column_values) - 0.01
        max_values.append(max_value)
        min_values.append(min_value)
        for j in range(len(f)):
            f[j][i] = (f[j][i] - min_value) / (max_value - min_value)

    with open(os.path.join(STYLOMETRY_DIR, f"{name}_normalization.json"), 'w') as fp:
        json.dump([min_values, max_values], fp)
    return f


def fit_model(name: str, samples: np.ndarray, labels: list, model_type: str = 'logistic') -> object:
    print(f"Training {name}...")
    model_functions = {'logistic': logistic_regression, 'forest': random_forest}
    model = model_functions[model_type](samples, labels)

    with open(os.path.join(STYLOMETRY_DIR, f"{name}.pickle"), 'wb') as f:
        pickle.dump(model, f)
    return model


def labels_to_truth_table(labels: list[str], true_class: str, ai_v_human: bool = False) -> list[int]:
    if ai_v_human:
        return [1 if used_authors[label] == true_class else 0 for label in labels]
    return [1 if label == true_class else 0 for label in labels]


def train_individual_model(author: str, feature: str, appendix: str, samples: dict, labels: list[str]) -> object:
    truth_table = labels_to_truth_table(labels, author)
    model_name = f"{author}_{feature}{appendix}"
    return fit_model(model_name, samples[feature], truth_table)


def train_intermediate_models(samples: dict, labels: list[str], appendix: str) -> dict:
    interm_results = {'char': [], 'syn': [], 'word': []}
    for feature in ['char', 'syn', 'word']:
        models = [train_individual_model(author, feature, appendix, samples, labels) for author in set(labels)]
        for iterrow in samples[feature].iterrows():
            interm_results[feature].append(
                [model.predict_proba(iterrow[1].values.reshape(1, -1))[0][1] for model in models])
        interm_results[feature] = fit_normalize_interm_results(f"{feature}{appendix}", interm_results[feature])
    return interm_results


def train_final_model(feature: str, appendix: str, features: list[list[float]], labels: list[str]) -> None:
    truth_table = labels_to_truth_table(labels, 'ai', ai_v_human=True)
    features = pd.DataFrame(features)
    model_name = f"{feature}_final{appendix}"
    fit_model(model_name, features.values, truth_table, 'forest')


def train_stylometry(samples: dict, labels: list[str], appendix: str) -> None:
    interm_results = train_intermediate_models(samples, labels, appendix)
    for feature, results in interm_results.items():
        train_final_model(feature, appendix, results, labels)
    combined_features = [char + syn + word for char, syn, word in interm_results.values()]
    train_final_model("style", appendix, combined_features, labels)
    print("TRAINING DONE!")
