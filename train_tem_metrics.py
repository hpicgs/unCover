import os
import pickle
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from definitions import MODELS_DIR
from misc.logger import printProgressBar
from misc.mock_database import DatabaseAuthorship, DatabaseGenArticles
from misc.tem_helpers import get_default_tecm
from misc.nlp_helpers import handle_nltk_download

author_mapping = {
    "gpt3": 1,
    "gpt2": 2,
    "gpt4": 3,
    "gpt3-phrase": 4,
    "grover": 5,
    "gemini": 6,
    "human1": 0,
    "human2": 0,
    "human3": 0,
    "human4": 0,
    "human5": 0,
    "": 0
}

model_pickle = os.path.join(MODELS_DIR, "tem_metrics", 'metrics_model.pickle')


def run_tem(data):
    try:  # check if nltk is installed and download if it is not
        result = []
        for i, article in enumerate(data):
            printProgressBar(i, len(data) - 1)
            try:
                result.append(get_default_tecm(article))
            except ValueError:
                continue
        return result
    except LookupError as e:
        handle_nltk_download(e)
        return run_tem(data)  # recursive call to deal with multiple downloads


def prepare_train_data(database, training_data, label):
    for author in database.get_authors():
        print("working on author: " + author + "...")
        tmp = run_tem([article["text"] for article in database.get_articles_by_author(author)])
        training_data += tmp
        label += [author_mapping[author]] * len(tmp)


def tem_metric_training():
    features = []
    labels = []
    prepare_train_data(DatabaseAuthorship, features, labels)
    prepare_train_data(DatabaseGenArticles, features, labels)
    X = pd.DataFrame(features)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=9732)
    n_scores = cross_val_score(model, X, labels, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    return model.fit(X, labels)


def predict_from_tecm(metrics: npt.NDArray[np.float64]):
    df = pd.DataFrame(metrics, index=[0])
    with open(model_pickle, "rb") as f:
        model = pickle.load(f)
    prediction = model.predict_proba(df)
    argmax = prediction[0].argmax()
    if argmax > 1:  # if class is one of AI classes we want to only output the AI class 1
        argmax = 1
    return argmax, prediction[0].max()


if __name__ == '__main__':
    os.makedirs(os.path.join(MODELS_DIR, "tem_metrics"), exist_ok=True)
    with open(model_pickle, 'wb') as f:
        model = tem_metric_training()
        pickle.dump(model, f)
    print("TRAINING DONE!")
