import os
import argparse
from typing import Tuple

import numpy as np
from pandas._typing import npt

from definitions import DATABASE_AUTHORS_PATH, DATABASE_GEN_PATH, DATABASE_GERMAN_PATH, STYLOMETRY_DIR, TEGMETRICS_DIR
from misc.mock_database import Database
from misc.tem_helpers import preprocess_database
from misc.tegm_training import process_tegm, train_tegm, save_tegm
from stylometry.training import process_stylometry, train_stylometry


def optimize_tegm(data: dict, labels: list, appendix: str, args: argparse.Namespace) -> None:
    best_params = None
    best_score, adj_score = 0.0, 0.0
    param_combinations = (
        np.ndarray([c, alpha, beta, gamma, delta, theta, merge, evolv])
        for c in np.arange(0.1, 1.0, 0.2)
        for alpha in [0.0, 1.1, 0.25]
        for beta in np.arange(0.25, 1.1, 0.25)
        for gamma in np.arange(0.25, 1.1, 0.25)
        for delta in [0.9, 1.25, 1.5, 1.75]
        for theta in np.arange(0.3, 1.0, 0.1625)
        for merge in np.arange(0.3, 1.0, 0.1625)
        for evolv in np.arange(0.3, 1.0, 0.1625)
    )
    with open(os.path.join(TEGMETRICS_DIR, 'hyperparameters.log'), 'a') as f:
        param_len = len(all_params := list(param_combinations))
        for i, params in enumerate(all_params):
            print(f"Training with parameter combination {i}/{param_len}: {params}")
            samples = process_tegm(data, appendix, args, 0.5, params)
            _ ,s, d = train_tegm(samples, labels, appendix, params)
            f.write(f"{params}/{s}/{d}\n")
            if s - d > adj_score:
                print(f"Better performance with score: {s}({d})")
                adj_score = s - d
                best_params = params
            else:
                print("Worse performance, skipping...")

    print(f"Best parameters: {best_params}")
    samples = process_tegm(data, appendix, args, params=best_params)
    m, s, d = train_tegm(samples, labels, appendix, best_params)
    print(f"Best score: {s}({d})")
    save_tegm(m, appendix)


def handle_invalids(s: list[npt.NDArray[np.float64]], l: list[str]) -> Tuple[list[npt.NDArray[np.float64]], list[str]]:
    invalid_indices = [i for i, sample in enumerate(s) if np.all(np.isnan(s))]
    for i in reversed(invalid_indices):
        del s[i]
        del l[i]
    return s, l


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action='store', required=False, choices=['stylometry', 'tegm', 'both'],
                        default='both', help="choose the mode of training")
    parser.add_argument('--german', action='store_true', required=False,
                        help="use the german test database instead of the english one")
    parser.add_argument('--use-stored', action='store', required=False,
                        help="use previously created train data instead of generating again")
    parser.add_argument('--n-trigram-features', action='store', required=False, type=int, default=100,
                        help="number of trigram features used in the distribution for stylometry")
    parser.add_argument('--optimize-tegm', action='store_true', required=False,
                        help="optimize the parameters of the TEGM classification through Grid Search")
    parser.add_argument('--tegm-preprocessed', action='store_true', required=False,
                        help='specifies that the database holds already preprocessed articles')
    parser.add_argument('--tegm-preprocessing', action='store_true', required=False,
                        help='only run the TEGM preprocessing and save database again')
    args = parser.parse_args()

    databases = []
    if args.german:
        print("Using German database...")
        file_appendix = '_german'
        databases.append(Database(DATABASE_GERMAN_PATH))
    else:
        print("Using English database...")
        file_appendix = ''
        databases.append(Database(DATABASE_AUTHORS_PATH))
        databases.append(Database(DATABASE_GEN_PATH))

    if args.tegm_preprocessing:
        print("Preprocessing database for TEGM...")
        for database in databases:
            preprocess_database(database)

    training_data = {}
    for database in databases:
        training_data.update(database.get_all_articles_sorted_by_labels())
    print(f"Number of training articles: {sum(len(value) for value in training_data.values())}")

    labels = []
    for label, articles in training_data.items():
        labels.extend([label] * len(articles))

    if args.mode == 'tegm' or args.mode == 'both':
        os.makedirs(TEGMETRICS_DIR, exist_ok=True)
        if args.optimize_tegm:
            print("Optimizing TEGM parameters...")
            optimize_tegm(training_data, labels, file_appendix, args)
        else:
            print("Training TEGM...")
            samples = process_tegm(training_data, file_appendix, args)
            samples, tegm_labels = handle_invalids(samples, labels)
            m, _, _ = train_tegm(samples, tegm_labels, file_appendix)
            save_tegm(m, file_appendix)
    if args.mode == 'stylometry' or args.mode == 'both':
        print("Training Stylometry...")
        os.makedirs(STYLOMETRY_DIR, exist_ok=True)
        file_appendix = f"{file_appendix}_{args.n_trigram_features}"
        samples = process_stylometry(training_data, file_appendix, args)
        train_stylometry(samples, labels, file_appendix)
