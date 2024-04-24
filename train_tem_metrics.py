import os
import pickle
import argparse
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from definitions import TEMMETRICS_DIR, TEM_PARAMS
from misc.mock_database import DatabaseAuthorship, DatabaseGenArticles, GermanDatabase
from misc.tem_helpers import get_tecm, preprocess
from misc.nlp_helpers import handle_nltk_download
from misc.logger import printProgressBar

author_mapping = {
    'gpt3': 1,
    'gpt2': 2,
    'gpt4': 3,
    'gpt3-phrase': 1,
    'grover': 4,
    'gemini': 5,
    'human1': 0,
    'human2': 0,
    'human3': 0,
    'human4': 0,
    'human5': 0,
    'human6': 0,
    'human7': 0
}

data_save = {
    'initialized': False
}


def log_dir(p: npt.NDArray, pre: str) -> str:
    return os.path.join(TEMMETRICS_DIR, f"{pre}logs",
                        f"tem_metrics_{p[0]}_{p[1]}_{p[2]}_{p[3]}_{p[4]}_{p[5]}_{p[6]}_{p[7]}")


def model_pickle(pre: str) -> str:
    return os.path.join(TEMMETRICS_DIR, f"{pre}model.pickle")


def run_tem(articles: List[str], tem_params: npt.NDArray, german: bool, preprocessed: bool) \
        -> npt.NDArray[np.float64]:
    try:  # check if nltk is installed and download if it is not
        return get_tecm(articles, tem_params, preprocess=not preprocessed)
    except LookupError as e:
        handle_nltk_download(e)
        return run_tem(articles, tem_params, german, preprocessed)  # recursive call to deal with multiple downloads


def prepare_train_data(database, training_data: List[float], label: List[int], portion: float,
                       tem_params: npt.NDArray, german: bool, preprocessed: bool) -> None:
    if data_save['initialized']:
        for author in data_save[database]:
            print(f"working on author: {author}...")
            articles = run_tem(data_save[author], tem_params, german, preprocessed)
            training_data.extend(articles)
            label += [author_mapping[author]] * len(articles)
        return

    data_save['portion'] = portion
    authors = database.get_authors()
    data_save[database] = authors

    for author in authors:
        print(f"working on author: {author}...")
        articles = [article['text'] for article in database.get_articles_by_author(author)]
        articles = articles[:int(len(articles) * portion)]
        data_save[author] = articles
        articles = run_tem(articles, tem_params, german, preprocessed)
        training_data.extend(articles)
        label += [author_mapping[author]] * len(articles)


def fit_model(params: npt.NDArray, features: List[float], truth_table: List[int], prefix: str) -> Tuple:
    scaler = StandardScaler()
    scaler.fit(features)
    with open(os.path.join(log_dir(params, prefix), f"{prefix}scalar.pickle"), 'wb') as f:
        pickle.dump(scaler, f)
    features = scaler.transform(features)

    with open(os.path.join(log_dir(params, prefix), 'results.log'), 'w') as log:
        log.write("Fitting TEGM Classifier...\n")

        # logistic regression
        df = pd.DataFrame(features)
        logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=9732)
        logreg_n_scores = cross_val_score(logreg, df, truth_table, scoring='accuracy', cv=cv, n_jobs=-1)
        log.write(
            'Logistic Regression Mean Accuracy: %.3f (%.3f)' % (np.mean(logreg_n_scores), np.std(logreg_n_scores)))
        with open(os.path.join(log_dir(params, prefix), 'logreg.pickle'), 'wb') as m:
            pickle.dump(logreg.fit(df, truth_table), m)

        # random forest
        forest = RandomForestClassifier(random_state=42)
        forest_n_scores = cross_val_score(forest, df, truth_table, scoring='accuracy', cv=cv, n_jobs=-1)
        log.write('Random Forest Mean Accuracy: %.3f (%.3f)' % (np.mean(forest_n_scores), np.std(forest_n_scores)))
        with open(os.path.join(log_dir(params, prefix), 'forest.pickle'), 'wb') as m:
            pickle.dump(forest.fit(df, truth_table), m)

        # pick better model
        return (logreg.fit(df, truth_table), np.mean(logreg_n_scores), np.std(logreg_n_scores)) if np.mean(
            logreg_n_scores) > np.mean(forest_n_scores) else (
            forest.fit(df, truth_table), np.mean(forest_n_scores), np.std(forest_n_scores))


def tem_metric_training(portion: float = 1.0, params: npt.NDArray = TEM_PARAMS, german: bool = False,
                        preprocessed: bool = False) -> Tuple:
    sample, truth = [], []
    prefix = 'ger_' if german else ''
    os.makedirs(log_dir(params, prefix), exist_ok=True)

    if german:
        prepare_train_data(GermanDatabase, sample, truth, portion, params, german, preprocessed)
    else:
        prepare_train_data(DatabaseAuthorship, sample, truth, portion, params, german, preprocessed)
        prepare_train_data(DatabaseGenArticles, sample, truth, portion, params, german, preprocessed)
    if not data_save['initialized']:
        data_save['initialized'] = True

    pickle.dump(sample, open(os.path.join(log_dir(params, prefix), 'features.pickle'), 'wb'))
    pickle.dump(truth, open(os.path.join(log_dir(params, prefix), 'labels.pickle'), 'wb'))

    return fit_model(params, sample, truth, prefix)


def optimize_tem(preprocessed: bool) -> Tuple:
    best_params = TEM_PARAMS
    _, mean_score, d = tem_metric_training(1.0, best_params, preprocessed=preprocessed)
    mean_score, d = 0.0, 0.0
    with open(os.path.join(TEMMETRICS_DIR, 'hyperparameters.txt'), 'w') as f:
        f.write(f"{best_params}/{mean_score}/{d}\n")
        param_combinations = (
            [c, alpha, beta, gamma, delta, theta, merge, evolv]
            for c in np.arange(0.1, 1.0, 0.2)
            for alpha in [0.0, 1.1, 0.25]
            for beta in np.arange(0.25, 1.1, 0.25)
            for gamma in np.arange(0.25, 1.1, 0.25)
            for delta in [0.9, 1.25, 1.5, 1.75]
            for theta in np.arange(0.3, 1.0, 0.1625)
            for merge in np.arange(0.3, 1.0, 0.1625)
            for evolv in np.arange(0.3, 1.0, 0.1625)
        )
        for params in param_combinations:
            m, s, d = tem_metric_training(0.5, params, preprocessed=preprocessed)
            f.write(f"{params}/{s}/{d}\n")
            if s > mean_score:
                print(f"Better performance on: {params}, mean score: {s}({d})")
                mean_score = s
                best_params = params
            else:
                print("Worse performance, skipping...")
    print(f"Best parameters: {best_params}, mean score: {mean_score}")
    return tem_metric_training(1.0, best_params, preprocessed=preprocessed)


def preprocess_data(database) -> None:
    authors = database.get_authors()
    articles = []
    for author in authors:
        print(f"fetching author: {author}...")
        articles.extend(database.get_articles_by_author(author))

    for i, article in enumerate(articles):
        printProgressBar(i, len(articles) - 1, fill='█')
        article['text'] = preprocess(article['text'])
        try:
            article['author'] = article['author'][0]
        except KeyError:
            article['source'] = article['source'][0]

    database.replace_data(articles)


def predict_from_tecm(metrics: npt.NDArray[np.float64], model_prefix: str = '') -> Tuple[int, float]:
    df = pd.DataFrame(metrics.reshape(1, -1))
    with open(os.path.join(TEMMETRICS_DIR, f"{model_prefix}scalar.pickle"), 'rb') as f:
        scalar = pickle.load(f)
    with open(model_pickle(model_prefix), 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict_proba(scalar.transform(df))
    conf = prediction[0].max()
    if conf > 0.6:
        argmax = prediction[0].argmax()
        if argmax == 0:
            return -1, conf
        elif conf > 0.65:
            return 1, conf
    return 0, conf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_stored', action='store_true', required=False,
                        help="rerun the TEM model to generate train data, and ignore existing data")
    parser.add_argument('--optimize_tem', action='store_true', required=False,
                        help="optimize the parameters of the TEM model through Grid Search")
    parser.add_argument('--german', action='store_true', required=False,
                        help="use the german test database instead of the english one")
    parser.add_argument('--preprocessed', action='store_true', required=False,
                        help='specifies that the db holds already preprocessed data')
    parser.add_argument('--preprocess', action='store_true', required=False,
                        help='only run the preprocessing and save database again')
    args = parser.parse_args()

    os.makedirs(TEMMETRICS_DIR, exist_ok=True)
    if args.preprocess:
        print("Preprocessing the database...")
        preprocess_data(DatabaseAuthorship)
        preprocess_data(DatabaseGenArticles)
        print("Preprocessing done!")
        exit(0)

    prefix = 'ger_' if args.german else ''
    with open(model_pickle(prefix), 'wb') as f:
        feature_file = os.path.join(TEMMETRICS_DIR, f"{prefix}features.pickle")
        label_file = os.path.join(TEMMETRICS_DIR, f"{prefix}labels.pickle")
        if args.optimize_tem:
            if args.german:
                # print error and exit
                parser.error("Optimization is not supported for the german database")
            print("Optimizing TEM model...")
            model, score, deviation = optimize_tem(args.preprocessed)
        elif args.use_stored and os.path.exists(feature_file) and os.path.exists(label_file):
            print("Loading existing training data...")
            features = pickle.load(open(feature_file, 'rb'))
            labels = pickle.load(open(label_file, 'rb'))
            model, score, deviation = fit_model(TEM_PARAMS, features, labels, prefix)
        else:
            model, score, deviation = tem_metric_training(german=args.german)
        print(f"Saving model with mean score: {score}")
        pickle.dump(model, f)

    print("TRAINING DONE!")
