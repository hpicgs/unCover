from database.mock_database import TestDatabase
from stylometry.logistic_regression import predict_author
from main import get_prediction, run_tem_on

source_mapping = {
    "human": -1,
    "gpt2": 1,
    "gpt3": 1,
    "grover": 1
}

# Print iterations progress
def printProgressBar (iteration, total, decimals = 1, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(100 * iteration // total)
    bar = fill * filledLength + '-' * (100 - filledLength)
    print(f'\r |{bar}| {percent}% ', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

if __name__ == '__main__':
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
        for article in articles:
            total_count += 1
            printProgressBar(total_count, 771)
            if(len(article) > 120000):
                article = article[:120000]
            try:
                style_prediction = predict_author(article)
                te_prediction, _ = run_tem_on(article)
            except AttributeError:  # some texts are still not working for tem
                total_count -= 1
                continue
            source_count += 1
            author = get_prediction(style_prediction, te_prediction)
            sem_style_predictions.update({style_prediction[1]: sem_style_predictions.get(style_prediction[1]) + 1})
            char_style_predictions.update({style_prediction[0]: char_style_predictions.get(style_prediction[0]) + 1})
            style_prediction = 0 if style_prediction[0] == -style_prediction[1] or style_prediction == [0, 0] else \
                1 if style_prediction[0] == 1 or style_prediction[1] == 1 else -1
            total_style_predictions.update({style_prediction: total_style_predictions.get(style_prediction) + 1})
            te_prediction = te_prediction[0]
            if te_prediction == 0:
                te_prediction = -1
            te_predictions.update({te_prediction: te_predictions.get(te_prediction) + 1})
            total_predictions.update({author: total_predictions.get(author) + 1})
        predictions_per_author.update({source: {"count": source_count,
                                                "char_style": char_style_predictions,
                                                "sem_style": sem_style_predictions,
                                                "total_style": total_style_predictions, "te": te_predictions,
                                                "total": total_predictions}})
    print("total Number of articles analyzed: ", str(total_count))
    print(predictions_per_author)
    total_char_true, total_sem_true, total_style_true, total_te_true, total_total_true = 0, 0, 0, 0, 0
    ai_total_char_true, ai_total_sem_true, ai_total_style_true, ai_total_te_true, ai_total_total_true = 0, 0, 0, 0, 0
    ai_total_char_unsure, ai_total_sem_unsure, ai_total_style_unsure, ai_total_te_unsure, ai_total_total_unsure = 0, 0, 0, 0, 0
    ai_total_char_false, ai_total_sem_false, ai_total_style_false, ai_total_te_false, ai_total_total_false = 0, 0, 0, 0, 0
    human_total_char_true, human_total_sem_true, human_total_style_true, human_total_te_true, human_total_total_true = 0, 0, 0, 0, 0
    human_total_char_unsure, human_total_sem_unsure, human_total_style_unsure, human_total_te_unsure, human_total_total_unsure = 0, 0, 0, 0, 0
    human_total_char_false, human_total_sem_false, human_total_style_false, human_total_te_false, human_total_total_false = 0, 0, 0, 0, 0
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
            print(prediction, " false: ", str(round(predictions_per_author[source][prediction][-source_mapping[source]] /
                                                   count * 100, 2)), "%\n")
        total_char_true += predictions_per_author[source]["char_style"][source_mapping[source]]
        total_sem_true += predictions_per_author[source]["sem_style"][source_mapping[source]]
        total_style_true += predictions_per_author[source]["total_style"][source_mapping[source]]
        total_te_true += predictions_per_author[source]["te"][source_mapping[source]]
        total_total_true += predictions_per_author[source]["total"][source_mapping[source]]
        if source == "human":
            continue
        ai_total_char_true += predictions_per_author[source]["char_style"][1]
        ai_total_sem_true += predictions_per_author[source]["sem_style"][1]
        ai_total_style_true += predictions_per_author[source]["total_style"][1]
        ai_total_te_true += predictions_per_author[source]["te"][1]
        ai_total_total_true += predictions_per_author[source]["total"][1]
        ai_total_char_unsure += predictions_per_author[source]["char_style"][0]
        ai_total_sem_unsure += predictions_per_author[source]["sem_style"][0]
        ai_total_style_unsure += predictions_per_author[source]["total_style"][0]
        ai_total_te_unsure += predictions_per_author[source]["te"][0]
        ai_total_total_unsure += predictions_per_author[source]["total"][0]
        ai_total_char_false += predictions_per_author[source]["char_style"][-1]
        ai_total_sem_false += predictions_per_author[source]["sem_style"][-1]
        ai_total_style_false += predictions_per_author[source]["total_style"][-1]
        ai_total_te_false += predictions_per_author[source]["te"][-1]
        ai_total_total_false += predictions_per_author[source]["total"][-1]
    human_total_char_true += predictions_per_author["human"]["char_style"][-1]
    human_total_sem_true += predictions_per_author["human"]["sem_style"][-1]
    human_total_style_true += predictions_per_author["human"]["total_style"][-1]
    human_total_te_true += predictions_per_author["human"]["te"][-1]
    human_total_total_true += predictions_per_author["human"]["total"][-1]
    human_total_char_unsure += predictions_per_author["human"]["char_style"][0]
    human_total_sem_unsure += predictions_per_author["human"]["sem_style"][0]
    human_total_style_unsure += predictions_per_author["human"]["total_style"][0]
    human_total_te_unsure += predictions_per_author["human"]["te"][0]
    human_total_total_unsure += predictions_per_author["human"]["total"][0]
    human_total_char_false += predictions_per_author["human"]["char_style"][1]
    human_total_sem_false += predictions_per_author["human"]["sem_style"][1]
    human_total_style_false += predictions_per_author["human"]["total_style"][1]
    human_total_te_false += predictions_per_author["human"]["te"][1]
    human_total_total_false += predictions_per_author["human"]["total"][1]
    print("\nchar accuracy: ", str(round(total_char_true / total_count * 100, 2)), "%")
    print("sem accuracy: ", str(round(total_sem_true / total_count * 100, 2)), "%")
    print("style accuracy: ", str(round(total_style_true / total_count * 100, 2)), "%")
    print("te accuracy: ", str(round(total_te_true / total_count * 100, 2)), "%")
    print("final accuracy: ", str(round(total_total_true / total_count * 100, 2)), "%")
    precision_char = ai_total_char_true / (ai_total_char_true + ai_total_char_false)
    precision_sem = ai_total_sem_true / (ai_total_sem_true + ai_total_sem_false)
    precision_style = ai_total_style_true / (ai_total_style_true + ai_total_style_false)
    precision_te = ai_total_te_true / (ai_total_te_true + ai_total_te_false)
    precision_total = ai_total_total_true / (ai_total_total_true + ai_total_total_false)
    recall_char = ai_total_char_true / (ai_total_char_true + ai_total_char_false + 0.5 * ai_total_char_unsure)
    recall_sem = ai_total_sem_true / (ai_total_sem_true + ai_total_sem_false + 0.5 * ai_total_sem_unsure)
    recall_style = ai_total_style_true / (ai_total_style_true + ai_total_style_false + 0.5 * ai_total_style_unsure)
    recall_te = ai_total_te_true / (ai_total_te_true + ai_total_te_false + 0.5 * ai_total_te_unsure)
    recall_total = ai_total_total_true / (ai_total_total_true + ai_total_total_false + 0.5 * ai_total_total_unsure)
    f1_char = 2 * precision_char * recall_char / (precision_char + recall_char)
    f1_sem = 2 * precision_sem * recall_sem / (precision_sem + recall_sem)
    f1_style = 2 * precision_style * recall_style / (precision_style + recall_style)
    f1_te = 2 * precision_te * recall_te / (precision_te + recall_te)
    f1_total = 2 * precision_total * recall_total / (precision_total + recall_total)
    print("\nchar f1: ", str(round(f1_char * 100, 2)), "%")
    print("sem f1: ", str(round(f1_sem * 100, 2)), "%")
    print("style f1: ", str(round(f1_style * 100, 2)), "%")
    print("te f1: ", str(round(f1_te * 100, 2)), "%")
    print("final f1: ", str(round(f1_total * 100, 2)), "%")

