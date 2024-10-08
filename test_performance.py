import argparse
import json
import csv
import os.path

import requests
import numpy as np
from misc.mock_database import TestDatabase, GermanTestDatabase
from misc.tem_helpers import get_tegm
from misc.logger import printProgressBar
from stylometry.classifier import predict_author, used_authors
from main import get_prediction
from train_tem_metrics import predict_from_tecm
from definitions import WINSTON_API_KEY, DATABASE_FILES_PATH

source_mapping = {
    'human-verified': -1,
    'human': -1,
    'gpt2': 1,
    'gpt3': 1,
    'gpt4': 1,
    'gemini': 1,
    'grover': 1
}


def eval_argugpt():
    with open(os.path.join(DATABASE_FILES_PATH, 'argugpt-test.csv'), 'r') as file:
        #structure: id,prompt_id,prompt,text,model,temperature,exam_type,score,score_level
        reader = csv.reader(file)
        data = list(reader)
    total = 0
    correct = 0
    for row in data:
        if row[0] == 'id':
            continue
        total += 1
        style = predict_author(row[3])
        tecm = get_tegm([row[3]], drop_invalids=False)[0]
        if np.all(np.isnan(tecm)):
            print("\nTEM processing error, skipping...")
            total -= 1
            continue
        te = predict_from_tecm(tecm)
        prediction = get_prediction(style, te)
        if prediction == int(row[4]):
            correct += 1
    print(f"Accuracy: {round(correct / total * 100, 2)}%")


def predict_sota(text, german):
    payload = {
        'language': 'de' if german else 'en',
        'sentences': False,
        'text': text,
        'version': 'latest'
    }
    headers = {
        'Authorization': f"Bearer {WINSTON_API_KEY}",
        'Content-Type': 'application/json'
    }

    response = requests.request('POST', "https://api.gowinston.ai/functions/v1/predict", json=payload, headers=headers)
    try:
        case = json.loads(response.text)['score']
    except Exception as e:
        print("\nError while fetching sota: ", e)
        return 0
    if case <= 40:
        return 1
    elif case >= 60:
        return -1
    else:
        return 0


def _initial_metrics(keys=None):
    if keys is None:
        keys = [-1, 0, 1]
    predictions = ['char_style', 'syn_style', 'word_style', 'total_style', 'te', 'total', 'sota']
    return {key: {pred: 0 for pred in predictions} for key in keys}


def calculate_metrics(predictions, src, total):
    true = predictions[source_mapping[src]]
    unsure = predictions[0]
    false = predictions[-source_mapping[src]]
    print(f"{prediction} true: {round(true / total * 100, 2)}%")
    print(f"{prediction} unsure: {round(unsure / total * 100, 2)}%")
    print(f"{prediction} false: {round(false / total * 100, 2)}%\n")
    return true, unsure, false


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compareSOTA', action='store_true', required=False,
                        help="If true, generate the performance on winston.ai as SOTA")
    parser.add_argument('--german', action='store_true', required=False,
                        help="use the german test database instead of the english one")
    parser.add_argument('--evalArguGPT' , action='store_true', required=False,
                        help="Íf true, evaluate the performance on ArguGPT data")
    args = parser.parse_args()
    if args.german:
        data = GermanTestDatabase.get_all_articles_sorted_by_methods()
    else:
        data = TestDatabase.get_all_articles_sorted_by_methods()
    predictions_per_author = {}
    total_count = 0
    num_articles = len(data) * 200
    used_authors = {**used_authors, **{'human-verified': 'human', 'human': 'human'}}
    for source in data:
        source_count = 0
        articles = data[source]
        char_style_predictions = {-1: 0, 0: 0, 1: 0}
        syn_style_predictions = {-1: 0, 0: 0, 1: 0}
        word_style_predictions = {-1: 0, 0: 0, 1: 0}
        total_style_predictions = {-1: 0, 0: 0, 1: 0}
        te_predictions = {-1: 0, 0: 0, 1: 0}
        total_predictions = {-1: 0, 0: 0, 1: 0}
        sota_predictions = {-1: 0, 0: 0, 1: 0}
        for article in articles:
            total_count += 1
            printProgressBar(total_count, num_articles)
            if args.compareSOTA:
                if len(article) > 100000:
                    article = article[:100000]
                sota = predict_sota(article, args.german)
                sota_predictions.update({sota: sota_predictions.get(sota) + 1})
            if len(article) > 120000:
                article = article[:120000]
            style_prediction = predict_author(article, file_appendix='_german' if args.german else '')
            tecm = get_tegm([article], drop_invalids=False)[0]
            if np.all(np.isnan(tecm)):
                print("\nTEM processing error, skipping...")
                total_count -= 1
                num_articles -= 1
                continue
            te_prediction = predict_from_tecm(tecm, model_prefix='german' if args.german else '')
            source_count += 1
            author = get_prediction(style_prediction, te_prediction)
            char_style_predictions.update({style_prediction[0]: char_style_predictions.get(style_prediction[0]) + 1})
            syn_style_predictions.update({style_prediction[1]: syn_style_predictions.get(style_prediction[1]) + 1})
            word_style_predictions.update({style_prediction[2]: word_style_predictions.get(style_prediction[2]) + 1})
            total_style_predictions.update({style_prediction[3]: char_style_predictions.get(style_prediction[3]) + 1})
            te_predictions.update({te_prediction[0]: te_predictions.get(te_prediction[0]) + 1})
            total_predictions.update({author: total_predictions.get(author) + 1})

        predictions_per_author.update({source: {'count': source_count,
                                                'char_style': char_style_predictions,
                                                'syn_style': syn_style_predictions,
                                                'word_style': word_style_predictions,
                                                'total_style': total_style_predictions,
                                                'te': te_predictions,
                                                'total': total_predictions}})
        if args.compareSOTA:
            predictions_per_author[source].update({'sota': sota_predictions})
    print(f"total Number of articles analyzed: {total_count}")
    # print(predictions_per_author)
    metrics = {
        'total': _initial_metrics(keys=[1]),
        'human': _initial_metrics(),
        'ai': _initial_metrics()
    }
    for source in predictions_per_author:
        print(f"\nStarting for Source: {source}...")
        count = predictions_per_author[source]['count']
        for prediction in predictions_per_author[source]:
            if prediction == 'count':
                continue
            (metrics[used_authors[source]][source_mapping[source]][prediction],
             metrics[used_authors[source]][0][prediction],
             metrics[used_authors[source]][-source_mapping[source]][prediction]) = (
                calculate_metrics(predictions_per_author[source][prediction], source, count))
            metrics['total'][1][prediction] += metrics[used_authors[source]][source_mapping[source]][prediction]

    for p in predictions_per_author['human']:  # using human just to access prediction types
        if p == 'count':
            continue
        print(f"\n{p} accuracy: {round(metrics['total'][1][p] / total_count * 100, 2)}%")
        precision = metrics['ai'][1][p] / (metrics['ai'][1][p] + metrics['ai'][-1][p] + 0.5 * metrics['ai'][0][p])
        recall = metrics['ai'][1][p] / (
                metrics['ai'][1][p] + 1.5 * metrics['human'][1][p] + 0.5 * metrics['human'][0][p])
        f1 = 2 * precision * recall / (precision + recall)
        print(f"{p} weighted f1: {round(f1 * 100, 2)}%")
