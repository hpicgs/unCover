import argparse
import json
import requests
from misc.mock_database import TestDatabase
from misc.tem_helpers import get_default_tecm
from stylometry.logistic_regression import predict_author, used_authors
from main import get_prediction
from train_tem_metrics import predict_from_tecm
from definitions import WINSTON_API_KEY

source_mapping = {
    "human-verified": -1,
    "human": -1,
    "gpt2": 1,
    "gpt3": 1,
    "gpt4": 1,
    "gemini": 1,
    "grover": 1
}


# Print iterations progress
def printProgressBar(iteration, total, decimals=1, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(100 * iteration // total)
    bar = fill * filledLength + '-' * (100 - filledLength)
    print(f'\r |{bar}| {percent}% ', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def predict_sota(text):
    payload = {
        "language": "en",
        "sentences": False,
        "text": text,
        "version": "latest"
    }
    headers = {
        "Authorization": f"Bearer {WINSTON_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", "https://api.gowinston.ai/functions/v1/predict", json=payload, headers=headers)
    case = json.loads(response.text)["score"]
    if case <= 40:
        return 1
    elif case >= 60:
        return -1
    else:
        return 0


def calculate_metrics(predictions, source):
    true = predictions[source_mapping[source]]
    unsure = predictions[0]
    false = predictions[-source_mapping[source]]
    print(prediction, " true: ", str(round(true / count * 100, 2)), "%")
    print(prediction, " unsure: ", str(round(unsure / count * 100, 2)), "%")
    print(prediction, " false: ", str(round(false / count * 100, 2)), "%\n")
    return true, unsure, false


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--compareSOTA", action="store_true", required=False,
                        help="If true, generate the performance on winston.ai as SOTA")
    args = parser.parse_args()
    data = TestDatabase.get_all_articles_sorted_by_methods()
    predictions_per_author = {}
    total_count = 0
    for source in data:
        source_count = 0
        articles = data[source]
        char_style_predictions = {-1: 0, 0: 0, 1: 0}
        sem_style_predictions = {-1: 0, 0: 0, 1: 0}
        total_style_predictions = {-1: 0, 0: 0, 1: 0}
        te_predictions = {-1: 0, 0: 0, 1: 0}
        total_predictions = {-1: 0, 0: 0, 1: 0}
        sota_predictions = {-1: 0, 0: 0, 1: 0}
        for article in articles:
            total_count += 1
            printProgressBar(total_count, 1373)
            if args.compareSOTA:
                if len(article) > 100000:
                    article = article[:100000]
                sota = predict_sota(article)
                sota_predictions.update({sota: sota_predictions.get(sota) + 1})
            if len(article) > 120000:
                article = article[:120000]
            try:
                style_prediction = predict_author(article)
                tecm = get_default_tecm(article)
                te_prediction = predict_from_tecm(tecm)
            except AttributeError as e:  # some texts are still not working for tem
                print("te error: ", e)
                total_count -= 1
                continue
            source_count += 1
            author = get_prediction(style_prediction, te_prediction)
            sem_style_predictions.update({style_prediction[1]: sem_style_predictions.get(style_prediction[1]) + 1})
            char_style_predictions.update({style_prediction[0]: char_style_predictions.get(style_prediction[0]) + 1})
            style_prediction = 0 if style_prediction[0] == -style_prediction[1] or style_prediction == [0, 0] else \
                1 if style_prediction[0] == 1 or style_prediction[1] == 1 else -1
            total_style_predictions.update({style_prediction: total_style_predictions.get(style_prediction) + 1})
            if te_prediction[1] < 0.6:
                te_prediction = 0
            elif te_prediction[0] == 0:
                te_prediction = -1
            else:
                te_prediction = 1
            te_predictions.update({te_prediction: te_predictions.get(te_prediction) + 1})
            total_predictions.update({author: total_predictions.get(author) + 1})

        predictions_per_author.update({source: {"count": source_count,
                                                "char_style": char_style_predictions,
                                                "sem_style": sem_style_predictions,
                                                "total_style": total_style_predictions, "te": te_predictions,
                                                "total": total_predictions}})
        if args.compareSOTA:
            predictions_per_author[source].update({"sota": sota_predictions})
    print("total Number of articles analyzed: ", str(total_count))
    # print(predictions_per_author)
    metrics = {
        "total": {1: {"char_style": 0, "sem_style": 0, "total_style": 0, "te": 0, "total": 0, "sota": 0}},
        "human": {
            -1: {"char_style": 0, "sem_style": 0, "total_style": 0, "te": 0, "total": 0, "sota": 0},
            0: {"char_style": 0, "sem_style": 0, "total_style": 0, "te": 0, "total": 0, "sota": 0},
            1: {"char_style": 0, "sem_style": 0, "total_style": 0, "te": 0, "total": 0, "sota": 0}},
        "ai": {
            1: {"char_style": 0, "sem_style": 0, "total_style": 0, "te": 0, "total": 0, "sota": 0},
            0: {"char_style": 0, "sem_style": 0, "total_style": 0, "te": 0, "total": 0, "sota": 0},
            -1: {"char_style": 0, "sem_style": 0, "total_style": 0, "te": 0, "total": 0, "sota": 0}}}
    for source in predictions_per_author:
        print("\nStarting for Source: ", source)
        count = predictions_per_author[source]["count"]
        for prediction in predictions_per_author[source]:
            if prediction == "count":
                continue
            (metrics[used_authors[source]][source_mapping[source]][prediction],
             metrics[used_authors[source]][0][prediction],
             metrics[used_authors[source]][-source_mapping[source]][prediction]) = (
                calculate_metrics(predictions_per_author[source][prediction], source))
            metrics["total"][1][prediction] += metrics[used_authors[source]][source_mapping[source]][prediction]

    for source in predictions_per_author:
        print("\nStarting for Source: ", source)
        count = predictions_per_author[source]["count"]
        for prediction in predictions_per_author[source]:
            if prediction == "count":
                continue
            print(prediction, " true: ", str(round(predictions_per_author[source][prediction][source_mapping[source]] /
                                                   count * 100, 2)), "%")
            print(prediction, " unsure: ", str(round(predictions_per_author[source][prediction][0] /
                                                     count * 100, 2)), "%")
            print(prediction, " false: ",
                  str(round(predictions_per_author[source][prediction][-source_mapping[source]] /
                            count * 100, 2)), "%\n")
    for p in predictions_per_author["human"]:  # using human just to access prediction types
        if p == "count":
            continue
        print("\n" + p + " accuracy: ", str(round(metrics["total"][1][p] / total_count * 100, 2)),
              "%")
        precision = metrics["ai"][1][p] / (metrics["ai"][1][p] + metrics["ai"][-1][p])
        recall = metrics["ai"][1][p] / (
                0.7 * metrics["ai"][1][p] + 1.5 * metrics["human"][1][p] + 0.5 * metrics["ai"][0][p])
        f1 = 2 * precision * recall / (precision + recall)
        print(p + " f1: ", str(round(f1 * 100, 2)), "%")
