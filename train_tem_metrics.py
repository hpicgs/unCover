import os
import pickle
import argparse
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from definitions import TEMMETRICS_DIR, TEM_PARAMS, ROOT_DIR
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

# data_save = {
#     'initialized': False
# }


def tmp_log_dir(params):
    return os.path.join(ROOT_DIR, "logs", f"tem_metrics_{params[0]}_{params[1]}_{params[2]}_{params[3]}_{params[4]}_{params[5]}_{params[6]}_{params[7]}")


def model_pickle(pre):
    return os.path.join(TEMMETRICS_DIR, f"{pre}model.pickle")


def run_tem(articles, tem_params, german, preprocessed):
    try:  # check if nltk is installed and download if it is not
        return get_tecm(articles, tem_params, preprocess=not preprocessed)
    except LookupError as e:
        handle_nltk_download(e)
        return run_tem(articles, tem_params, german, preprocessed)  # recursive call to deal with multiple downloads


def prepare_train_data(database, training_data, label, portion, tem_params, german, preprocessed):
    # if data_save['initialized']:
    #     for author in data_save[database]:
    #         print(f"working on author: {author}...")
    #         tmp = run_tem(data_save[author], tem_params, german, preprocessed)
    #         training_data.extend(tmp)
    #         label += [author_mapping[author]] * len(tmp)
    #     return
    # data_save['portion'] = portion
    authors = database.get_authors()
    # data_save[database] = authors

    for author in authors:
        print(f"working on author: {author}...")
        tmp = [article['text'] for article in database.get_articles_by_author(author)]
        tmp = tmp[:int(len(tmp) * portion)]
        # data_save[author] = tmp
        tmp = run_tem(tmp, tem_params, german, preprocessed)
        training_data.extend(tmp)
        label += [author_mapping[author]] * len(tmp)


def fit_model(params, features, truth_table):
    os.makedirs(tmp_log_dir(params), exist_ok=True)
    with open(os.path.join(tmp_log_dir(params), 'results.log'), 'w') as log:
        log.write("Fitting TEGM Classifier...\n")
        # logistic regression
        df = pd.DataFrame(features)
        logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=9732)
        logreg_n_scores = cross_val_score(logreg, df, truth_table, scoring='accuracy', cv=cv, n_jobs=-1)
        log.write('Logistic Regression Mean Accuracy: %.3f (%.3f)' % (np.mean(logreg_n_scores), np.std(logreg_n_scores)))
        with open(os.path.join(tmp_log_dir(params), 'logreg.pickle'), 'wb') as m:
            pickle.dump(logreg.fit(df, truth_table), m)
        # random forest
        forest = RandomForestClassifier(random_state=42)
        forest_n_scores = cross_val_score(forest, df, truth_table, scoring='accuracy', cv=cv, n_jobs=-1)
        log.write('Random Forest Mean Accuracy: %.3f (%.3f)' % (np.mean(forest_n_scores), np.std(forest_n_scores)))
        with open(os.path.join(tmp_log_dir(params), 'forest.pickle'), 'wb') as m:
            pickle.dump(forest.fit(df, truth_table), m)
        # SVM
        svm = SVC(kernel='poly', degree=8, random_state=42)
        svm_n_scores = cross_val_score(svm, df, truth_table, scoring='accuracy', cv=cv, n_jobs=-1)
        log.write('SVM Mean Accuracy: %.3f (%.3f)' % (np.mean(svm_n_scores), np.std(svm_n_scores)))
        with open(os.path.join(tmp_log_dir(params), 'svm.pickle'), 'wb') as m:
            pickle.dump(svm.fit(df, truth_table), m)
        # pick best model
        if np.mean(logreg_n_scores) > np.mean(forest_n_scores):
            if np.mean(logreg_n_scores) > np.mean(svm_n_scores):
                return logreg.fit(df, truth_table), np.mean(logreg_n_scores), np.std(logreg_n_scores)
            else:
                return svm.fit(df, truth_table), np.mean(svm_n_scores), np.std(svm_n_scores)
        elif np.mean(forest_n_scores) > np.mean(svm_n_scores):
            return forest.fit(df, truth_table), np.mean(forest_n_scores), np.std(forest_n_scores)
        else:
            return svm.fit(df, truth_table), np.mean(svm_n_scores), np.std(svm_n_scores)


def tem_metric_training(portion=1.0, params=None, german=False, preprocessed=False):
    sample, truth = [], []
    if german:
        prepare_train_data(GermanDatabase, sample, truth, portion, params, german, preprocessed)
    else:
        prepare_train_data(DatabaseAuthorship, sample, truth, portion, params, german, preprocessed)
        prepare_train_data(DatabaseGenArticles, sample, truth, portion, params, german, preprocessed)
    # if not data_save['initialized']:
    #     data_save['initialized'] = True
    pickle.dump(sample, open(os.path.join(tmp_log_dir(params), 'features.pickle'), 'wb'))
    pickle.dump(truth, open(os.path.join(tmp_log_dir(params), 'labels.pickle'), 'wb'))
    return fit_model(params, sample, truth)


def optimize_tem(c, theta, p):
    best_params = TEM_PARAMS
    best_model = None
    # _, mean_score, d = tem_metric_training(1.0, best_params, preprocessed=p)
    mean_score, d = 0.0, 0.0
    with open(os.path.join(TEMMETRICS_DIR, 'hyperparameters.txt'), 'w') as f:
        # f.write(f"{best_params}/{mean_score}/{d}\n")
        for merge in np.arange(0.3, 1.0, 0.1625):
            for evolv in np.arange(0.3, 1.0, 0.1625):
                params = np.array([c, 0, 0, 0, 1, theta, merge, evolv])
                m, s, d = tem_metric_training(1.0, params, preprocessed=p)
                f.write(f"{params}/{s}/{d}\n")
                if s > mean_score:
                    print(f"Better performance on: {params}, mean score: {s}({d})")
                    mean_score = s
                    best_params = params
                    best_model = m
                else:
                    print("Worse performance, skipping...")
    print(f"Best parameters: {best_params}, mean score: {mean_score}")
    return best_model


def preprocess_data(database):
    authors = database.get_authors()
    articles = []
    for author in authors:
        print(f"fetching author: {author}...")
        articles.extend(database.get_articles_by_author(author))
    for i, article in enumerate(articles):
        printProgressBar(i, len(articles)-1, fill='█')
        article['text'] = preprocess(article['text'])
        try:
            article['author'] = article['author'][0]
        except KeyError:
            article['source'] = article['source'][0]

    database.replace_data(articles)


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
    parser.add_argument('--use_stored', action='store_true', required=False,
                        help="rerun the TEM model to generate train data, and ignore existing data")
    parser.add_argument('--optimize_tem', action='store_true', required=False,
                        help="optimize the parameters of the TEM model through Grid Search")
    parser.add_argument('--german' , action='store_true', required=False,
                        help="use the german test database instead of the english one")
    parser.add_argument('--tem_params', type=str, required=False,
                        help='specify the parameters for the TEM model as a string of 2 floats separated by commas')
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
    prefix = 'german' if args.german else ''
    with open(model_pickle(prefix), 'wb') as f:
        feature_file = os.path.join(TEMMETRICS_DIR, f"{prefix}features.pickle")
        label_file = os.path.join(TEMMETRICS_DIR, f"{prefix}labels.pickle")
        if args.optimize_tem:
            if args.german:
                # print error and exit
                parser.error("Optimization is not supported for the german database")
            print("Optimizing TEM model...")
            params = np.array([float(x) for x in args.tem_params.split(',')])
            model, score, deviation = optimize_tem(params[0], params[1], args.preprocessed)
        elif args.use_stored and os.path.exists(feature_file) and os.path.exists(label_file):
            print("Loading existing training data...")
            features = pickle.load(open(feature_file, 'rb'))
            labels = pickle.load(open(label_file, 'rb'))
            model, score, deviation = fit_model(TEM_PARAMS, features, labels)
        else:
            model, score, deviation = tem_metric_training()
        print(f"Saving model with mean score: {score}")
        pickle.dump(model, f)
    print("TRAINING DONE!")
