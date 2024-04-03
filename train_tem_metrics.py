import os
import pickle
import argparse
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from definitions import TEMMETRICS_DIR, TEM_PARAMS
from misc.mock_database import DatabaseAuthorship, DatabaseGenArticles, GermanDatabase
from misc.tem_helpers import get_tecm
from misc.nlp_helpers import handle_nltk_download

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
    'human5': 0
}

data_save = {
    'initialized': False
}


def model_pickle(pre):
    return os.path.join(TEMMETRICS_DIR, f"{pre}model.pickle")


def run_tem(articles, tem_params, german):
    try:  # check if nltk is installed and download if it is not
        return get_tecm(articles, tem_params, german=german)
    except LookupError as e:
        handle_nltk_download(e)
        return run_tem(articles, tem_params, german)  # recursive call to deal with multiple downloads


def prepare_train_data(database, training_data, label, portion, tem_params, german=False):
    if data_save['initialized']:
        for author in data_save[database]:
            print(f"working on author: {author}...")
            tmp = run_tem(data_save[author], tem_params, german)
            training_data.extend(tmp)
            label += [author_mapping[author]] * len(tmp)
        return
    data_save['portion'] = portion
    authors = database.get_authors()
    data_save[database] = authors
    for author in authors:
        print(f"working on author: {author}...")
        tmp = [article['text'] for article in database.get_articles_by_author(author)]
        tmp = tmp[:int(len(tmp) * portion)]
        data_save[author] = tmp
        tmp = run_tem(tmp, tem_params, german)
        training_data.extend(tmp)
        label += [author_mapping[author]] * len(tmp)


def fit_model(features, truth_table):
    df = pd.DataFrame(features)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=9732)
    n_scores = cross_val_score(model, df, truth_table, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    return model.fit(df, truth_table), np.mean(n_scores)


def tem_metric_training(portion=1.0, params=None, german=False):
    sample, truth = [], []
    if german:
        prepare_train_data(GermanDatabase, sample, truth, portion, params, True)
    else:
        prepare_train_data(DatabaseAuthorship, sample, truth, portion, params)
        prepare_train_data(DatabaseGenArticles, sample, truth, portion, params)
    if not data_save['initialized']:
        data_save['initialized'] = True
    pickle.dump(sample, open(feature_file, 'wb'))
    pickle.dump(truth, open(label_file, 'wb'))
    return fit_model(sample, truth)


def optimize_tem():
    best_params = TEM_PARAMS
    _, mean_score = tem_metric_training(0.33, best_params)
    with open(os.path.join(TEMMETRICS_DIR, 'hyperparameters.txt'), 'w') as f:
        f.write(f"Parameters: {best_params} mean score: {mean_score}\n")
        for c in np.arange(0.0, 1.0, 0.05):
            for alpha in np.arange(0.0, 1.0, 0.2):
                for beta in np.arange(0.0, 1.0, 0.2):
                    for gamma in np.arange(0.0, 1.0, 0.2):
                        for delta in np.arange(0.0, 1.0, 0.2):
                            for theta in np.arange(0.0, 1.0, 0.05):
                                for merge in np.arange(0.0, 1.0, 0.05):
                                    for evolv in np.arange(0.0, 1.0, 0.05):
                                        params = np.array([c, alpha, beta, gamma, delta, theta, merge, evolv])
                                        _, s = tem_metric_training(0.33, params)
                                        if s > mean_score:
                                            f.write(f"Parameters: {params} mean score: {s}\n")
                                            mean_score = s
                                            best_params = params
                                        else:
                                            print("Worse performance, skipping...")
    print(f"Best parameters: {best_params}")
    data_save['initialized'] = False
    return tem_metric_training(1.0, best_params)


def predict_from_tecm(metrics: npt.NDArray[np.float64], model_prefix=''):
    df = pd.DataFrame(metrics.reshape(1, -1))
    with open(model_pickle(model_prefix), 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict_proba(df)
    argmax = prediction[0].argmax()
    if argmax > 1:  # if class is one of AI classes, we want to only output the AI class 1
        argmax = 1
    return argmax, prediction[0].max()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_store', action='store_true', required=False,
                        help="rerun the TEM model to generate train data, and ignore existing data")
    parser.add_argument('--optimize_tem', action='store_true', required=False,
                        help="optimize the parameters of the TEM model through Grid Search")
    parser.add_argument('--german' , action='store_true', required=False,
                        help="use the german test database instead of the english one")
    args = parser.parse_args()
    os.makedirs(TEMMETRICS_DIR, exist_ok=True)
    prefix = 'german' if args.german else ''
    with open(model_pickle(prefix), 'wb') as f:
        feature_file = os.path.join(TEMMETRICS_DIR, f"{prefix}features.pickle")
        label_file = os.path.join(TEMMETRICS_DIR, f"{prefix}labels.pickle")
        if args.optimize_tem:
            if args.german:
                # print error and exit
                parser.error("Optimization is not supported for the german database")
            print("Optimizing TEM model...")
            model, score = optimize_tem()
        elif args.use_stored and os.path.exists(feature_file) and os.path.exists(label_file):
            print("Loading existing training data...")
            features = pickle.load(open(feature_file, 'rb'))
            labels = pickle.load(open(label_file, 'rb'))
            model, score = fit_model(features, labels)
        else:
            model, score = tem_metric_training()
        print(f"Saving model with mean score: {score}")
        pickle.dump(model, f)
    print("TRAINING DONE!")
