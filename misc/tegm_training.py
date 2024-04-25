import os
import pickle
import argparse
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from definitions import TEGMETRICS_DIR, TEM_PARAMS
from misc.tem_helpers import get_tecm
from misc.nlp_helpers import handle_nltk_download

class_mapping = {'human': 0, 'ai': 1}


def log_dir(p: npt.NDArray, appendix: str) -> str:
    return os.path.join(TEGMETRICS_DIR, f"logs{appendix}",
                        f"tem_metrics_{p[0]}_{p[1]}_{p[2]}_{p[3]}_{p[4]}_{p[5]}_{p[6]}_{p[7]}")


def save_tegm(model: LogisticRegression | RandomForestClassifier, appendix: str) -> None:
    with open(os.path.join(TEGMETRICS_DIR, f"model{appendix}.pickle"), 'wb') as f:
        pickle.dump(model, f)


def run_tem(articles: List[str], tem_params: npt.NDArray, preprocessed: bool) -> npt.NDArray[np.float64]:
    while True:
        try:  # handle potentially missing nltk downloads
            return get_tecm(articles, tem_params, preprocess=not preprocessed)
        except LookupError as e:
            handle_nltk_download(e)


def process_tegm(training_data: dict, appendix: str, args: argparse.Namespace, dataset_fraction: float = 1.0,
                 params: Optional[npt.NDArray[np.float64]] = TEM_PARAMS) -> list[npt.NDArray[np.float64]]:
    features_file_path = os.path.join(log_dir(params, appendix), "features.pickle")
    if args.use_stored:
        with open(features_file_path, 'rb') as f:
            features = pickle.load(f)
        if features is not None:
            return features

    os.makedirs(log_dir(params, appendix), exist_ok=True)
    features = []
    for label, articles in training_data.items():
        print(f"working on label: {label}...")
        articles = articles[:int(len(articles) * dataset_fraction)]
        features.extend(run_tem(articles, params, args.tegm_preprocessed))

    with open(features_file_path, 'wb') as f:
        pickle.dump(features, f)
    return features


def train_tegm(features: List[float], labels: List[str], appendix: str,
               params: Optional[npt.NDArray[np.float64]] = TEM_PARAMS) -> Tuple[Any, float, float]:
    scaler = StandardScaler()
    scaler.fit(features)
    with open(os.path.join(log_dir(params, appendix), f"scalar{appendix}.pickle"), 'wb') as f:
        pickle.dump(scaler, f)
    features = scaler.transform(features)

    with open(os.path.join(log_dir(params, appendix), 'results.log'), 'w') as log:
        log.write("Fitting TEGM Classifier...\n")

        # logistic regression
        df = pd.DataFrame(features)
        logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=9732)
        logreg_n_scores = cross_val_score(logreg, df, labels, scoring='accuracy', cv=cv, n_jobs=-1)
        log.write(
            'Logistic Regression Mean Accuracy: %.3f (%.3f)' % (np.mean(logreg_n_scores), np.std(logreg_n_scores)))
        with open(os.path.join(log_dir(params, appendix), 'logreg.pickle'), 'wb') as m:
            pickle.dump(logreg.fit(df, labels), m)

        # random forest
        forest = RandomForestClassifier(random_state=42)
        forest_n_scores = cross_val_score(forest, df, labels, scoring='accuracy', cv=cv, n_jobs=-1)
        log.write('Random Forest Mean Accuracy: %.3f (%.3f)' % (np.mean(forest_n_scores), np.std(forest_n_scores)))
        with open(os.path.join(log_dir(params, appendix), 'forest.pickle'), 'wb') as m:
            pickle.dump(forest.fit(df, labels), m)

        # pick better model
        return (logreg.fit(df, labels), np.mean(logreg_n_scores), np.std(logreg_n_scores)) if np.mean(
            logreg_n_scores) > np.mean(forest_n_scores) else (
            forest.fit(df, labels), np.mean(forest_n_scores), np.std(forest_n_scores))


def predict_from_tegm(metrics: npt.NDArray[np.float64], appendix: str = '') -> Tuple[int, float]:
    df = pd.DataFrame(metrics.reshape(1, -1))
    with open(os.path.join(TEGMETRICS_DIR, f"{appendix}scalar.pickle"), 'rb') as f:
        scalar = pickle.load(f)
    with open(os.path.join(TEGMETRICS_DIR, f"model{appendix}.pickle"), 'rb') as f:
        model = pickle.load(f)

    if scalar is None or model is None:
        return 0, 0

    prediction = model.predict_proba(scalar.transform(df))
    conf = prediction[0].max()
    if conf > 0.6:
        argmax = prediction[0].argmax()
        if argmax == 0:
            return -1, conf
        elif conf > 0.65:
            return 1, conf
    return 0, conf
