import os
import pickle
import argparse
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from definitions import TEMMETRICS_DIR, TEM_PARAMS
from misc.logger import printProgressBar
from misc.mock_database import DatabaseAuthorship, DatabaseGenArticles
from misc.tem_helpers import get_default_tecm
from misc.nlp_helpers import handle_nltk_download

author_mapping = {
    "gpt3": 1,
    "gpt2": 2,
    "gpt4": 3,
    "gpt3-phrase": 1,
    "grover": 4,
    "gemini": 5,
    "human1": 0,
    "human2": 0,
    "human3": 0,
    "human4": 0,
    "human5": 0,
    "": 0
}

model_pickle = os.path.join(TEMMETRICS_DIR, 'model.pickle')


def run_tem(data, tem_params):
    try:  # check if nltk is installed and download if it is not
        result = []
        for i, article in enumerate(data):
            printProgressBar(i, len(data) - 1)
            try:
                result.append(get_default_tecm(article, tem_params))
            except ValueError:
                continue
            except Exception as e:
                print("Error while processing article: ", e)
        return result
    except LookupError as e:
        handle_nltk_download(e)
        return run_tem(data, tem_params)  # recursive call to deal with multiple downloads


def prepare_train_data(database, training_data, label, portion, tem_params):
    for author in database.get_authors():
        print("working on author: " + author + "...")
        tmp = [article["text"] for article in database.get_articles_by_author(author)]
        tmp = run_tem(tmp[:int(len(tmp) * portion)], tem_params)
        training_data += tmp
        label += [author_mapping[author]] * len(tmp)


def fit_model(features, truth_table):
    df = pd.DataFrame(features)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=9732)
    n_scores = cross_val_score(model, df, truth_table, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    return model.fit(df, truth_table), np.mean(n_scores)


def tem_metric_training(portion=1.0, params=None):
    sample, truth = [], []
    prepare_train_data(DatabaseAuthorship, sample, truth, portion, params)
    prepare_train_data(DatabaseGenArticles, sample, truth, portion, params)
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
    return tem_metric_training(1.0, best_params)


def predict_from_tecm(metrics: npt.NDArray[np.float64]):
    df = pd.DataFrame(metrics.reshape(1, -1))
    with open(model_pickle, "rb") as f:
        model = pickle.load(f)
    prediction = model.predict_proba(df)
    argmax = prediction[0].argmax()
    if argmax > 1:  # if class is one of AI classes we want to only output the AI class 1
        argmax = 1
    return argmax, prediction[0].max()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_stored", action="store_true", required=False,
                        help="rerun the TEM model to generate train data, and ignore existing data")
    parser.add_argument("--optimize_tem", action="store_true", required=False,
                        help="optimize the parameters of the TEM model through Grid Search")
    args = parser.parse_args()
    os.makedirs(TEMMETRICS_DIR, exist_ok=True)
    with open(model_pickle, 'wb') as f:
        feature_file = os.path.join(TEMMETRICS_DIR, 'features.pickle')
        label_file = os.path.join(TEMMETRICS_DIR, 'labels.pickle')
        if args.optimize_tem:
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
