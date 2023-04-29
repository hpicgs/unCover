import csv

from analyze_tem import te_analysis_data
import argparse
import os
import pickle
from definitions import DATABASE_FILES_PATH, MODELS_DIR
from numpy import mean
from numpy import std
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

author_mapping = {
    "gpt3":0,
    "gpt2":0,
    "gpt3-phrase":0,
    "grover":0,
    "human":1
}


def tem_metric_training(path):
    classes = [f.path for f in os.scandir(path) if f.is_dir()]
    feature_names = []
    features = []
    labels = []
    for c in classes:
        with open(os.path.join(c, "_stats.csv"), 'r' ) as file:
            reader = csv.reader(file)
            for i,line in enumerate(reader):
                if i == 0:
                    feature_names = line[1:-1]
                    continue
                features.append(line[1:-1])
                labels.append(author_mapping[os.path.basename(c)])
    X = pd.DataFrame(features,columns=feature_names)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=9732)
    n_scores = cross_val_score(model, X, labels, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    return model.fit(X, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', action="store", required=False,
                        default=os.path.join(DATABASE_FILES_PATH, "tem_stats"),
                        help='directory where the training data is located')
    args = parser.parse_args()
    with open(os.path.join(os.path.join(MODELS_DIR, "tem_metrics", 'metrics_model.pickle')), 'wb') as f:
        pickle.dump(tem_metric_training(args.path), f)
