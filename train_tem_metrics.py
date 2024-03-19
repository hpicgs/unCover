import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from definitions import MODELS_DIR
from misc.mock_database import DatabaseAuthorship, DatabaseGenArticles
from misc.tem_helpers import get_default_tecm
from misc.nlp_helpers import handle_nltk_download

author_mapping = {
    "gpt3":1,
    "gpt2":2,
    "gpt4":3,
    "gpt3-phrase":4,
    "grover":5,
    "gemini":6,
    "human":0
}

def prepare_train_data(database, training_data, label):
    for author in database.get_authors():
        articles = [article["text"] for article in database.get_articles_by_author(author)]
        try: # check if nltk is installed and download if it is not
            training_data += [get_default_tecm(article) for article in articles]
        except LookupError as e:
            handle_nltk_download(e)
            training_data += [get_default_tecm(article) for article in articles]
        label += [author_mapping[author]] * len(articles)

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
    with open(os.path.join(MODELS_DIR, "tem_metrics", 'metrics_model.pickle'), "rb") as f:
        model = pickle.load(f)
    prediction = model.predict_proba(df)
    argmax = prediction[0].argmax()
    if argmax > 1:  # if class is one of AI classes we want to only output the AI class 1
        argmax = 1
    return argmax, prediction[0].max()


if __name__ == '__main__':
    os.create_dir(MODELS_DIR, "tem_metrics", exist_ok=True)
    with open(os.path.join(MODELS_DIR, "tem_metrics", 'metrics_model.pickle'), 'wb') as f:
        pickle.dump(tem_metric_training(), f)
