from definitions import STYLOMETRY_DIR, CHAR_MACHINE_CONFIDENCE, CHAR_HUMAN_CONFIDENCE, SEM_MACHINE_CONFIDENCE, SEM_HUMAN_CONFIDENCE
from database.mock_database import DatabaseAuthorship, DatabaseGenArticles
from stylometry.char_trigrams import char_trigrams
from stylometry.semantic_trigrams import sem_trigrams
from stylometry.logistic_regression import fixed_trigram_distribution
from nltk.parse.corenlp import CoreNLPDependencyParser
import pandas as pd
import pickle, os, argparse

used_authors = {
    "gpt2":"ai",
    "gpt3":"ai",
    "gpt3-phrase":"ai",
    "grover":"ai",
    "https:__www.theguardian.com_profile_hannah-ellis-petersen":"human",
    "https:__www.theguardian.com_profile_leyland-cecco":"human",
    "https:__www.theguardian.com_profile_martin-chulov":"human",
    "https:__www.theguardian.com_profile_julianborger":"human",
    "https:__www.theguardian.com_profile_helen-sullivan":"human"
}

parser = argparse.ArgumentParser()
parser.add_argument("--nfeatures", action="store", required=False, type=int, default=100, help="number of char trigram & semantic trigram features used in the distribution")
args = parser.parse_args()    

nfeatures = str(args.nfeatures) + "_2"

def write_test_distributions():
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    char_features = list(pd.read_csv(os.path.join(STYLOMETRY_DIR, f"char_distribution{nfeatures}.csv")).columns)[1:]
    sem_features = [eval(feature) for feature in list(pd.read_csv(os.path.join(STYLOMETRY_DIR, f"sem_distribution{nfeatures}.csv")).columns)[1:]]
    author_frame = pd.DataFrame({"author":[]})
    char_frames = []
    sem_frames = []
    authors = used_authors.keys()
    for author in authors:
        print(f"working on author {author}")
        if used_authors[author] == "human":
            full_article_list = [(article["text"], author) for article in DatabaseAuthorship.get_articles_by_author(author.replace("_", "/"))]
        elif used_authors[author] == "ai":
            full_article_list = [(article["text"], author) for article in DatabaseGenArticles.get_articles_by_author(author.replace("_", "/"))]
        test_data = full_article_list[int(len(full_article_list)*0.8):]
        print("creating char trigrams")
        char_grams = [char_trigrams(article_tuple[0]) for article_tuple in test_data]
        print("creating sem trigrams")
        sem_grams = [sem_trigrams(article_tuple[0], parser) for article_tuple in test_data]
        char_distribution = fixed_trigram_distribution(char_grams, char_features)
        sem_distribution = fixed_trigram_distribution(sem_grams, sem_features)
        author_frame = pd.concat([author_frame, pd.DataFrame({"author":[author]*len(test_data)})])
        char_frames.append(char_distribution)
        sem_frames.append(sem_distribution)
    full_char_distribution = char_frames[0]
    full_sem_distribution = sem_frames[0]
    for i in range(len(char_frames)-1):
        full_char_distribution = pd.concat([full_char_distribution, char_frames[i+1]])
        full_sem_distribution = pd.concat([full_sem_distribution, sem_frames[i+1]])
    full_char_distribution.insert(0, "author", author_frame["author"].to_list())
    full_sem_distribution.insert(0, "author", author_frame["author"].to_list())
    full_char_distribution.to_csv(os.path.join(STYLOMETRY_DIR, f"test_char_distribution{nfeatures}.csv"))
    full_sem_distribution.to_csv(os.path.join(STYLOMETRY_DIR, f"test_sem_distribution{nfeatures}.csv"))

def char_model_prediction(inp):
    authors = used_authors.keys()
    models = {}
    for author in authors:
        with open(os.path.join(STYLOMETRY_DIR, f"{author}_char{nfeatures}.pickle"), "rb") as fp:
            models[author] = pickle.load(fp)
    confidence_values = {}
    for author in authors:
        confidence_values[author] = models[author].predict_proba(inp)
    print(confidence_values)
    final_predictions = []
    raw_predictions = []
    for i in range(inp.shape[0]):
        machine = any(confidence_values[author][i][1] > CHAR_MACHINE_CONFIDENCE for author in authors if used_authors[author] == "ai")
        human = any(confidence_values[author][i][1] > CHAR_HUMAN_CONFIDENCE for author in authors if used_authors[author] == "human")
        if (machine and human) or (not human and not machine):
            final_predictions.append(0)
        elif machine:
            final_predictions.append(1)
        elif human:
            final_predictions.append(-1)
        machine = max(confidence_values[author][i][1] for author in authors if used_authors[author] == "ai")
        human = max(confidence_values[author][i][1] for author in authors if used_authors[author] == "human")
        raw_predictions.append((machine, human))

    return final_predictions

def sem_model_prediction(inp):
    authors = used_authors.keys()
    models = {}
    for author in authors:
        with open(os.path.join(STYLOMETRY_DIR, f"{author}_sem{nfeatures}.pickle"), "rb") as fp:
            models[author] = pickle.load(fp)
    confidence_values = {}
    for author in authors:
        confidence_values[author] = models[author].predict_proba(inp)
    final_predictions = []
    raw_predictions = []
    for i in range(inp.shape[0]):
        machine = any(confidence_values[author][i][1] > SEM_MACHINE_CONFIDENCE for author in authors if used_authors[author] == "ai")
        human = any(confidence_values[author][i][1] > SEM_HUMAN_CONFIDENCE for author in authors if used_authors[author] == "human")
        if (machine and human) or (not human and not machine):
            final_predictions.append(0)
        elif machine:
            final_predictions.append(1)
        elif human:
            final_predictions.append(-1)
        machine = max(confidence_values[author][i][1] for author in authors if used_authors[author] == "ai")
        human = max(confidence_values[author][i][1] for author in authors if used_authors[author] == "human")
        raw_predictions.append((machine, human))

    return final_predictions

def char_performance():
    test_dataframe = pd.read_csv(os.path.join(STYLOMETRY_DIR, f"test_char_distribution{nfeatures}.csv"))
    correct_class = []
    for i in range(test_dataframe.shape[0]):
        if used_authors[test_dataframe.iloc[i]["author"]] == "ai":
            correct_class.append(1)
        elif used_authors[test_dataframe.iloc[i]["author"]] == "human":
            correct_class.append(-1)
        else:
            correct_class.append(0)
    predictions = char_model_prediction(test_dataframe.drop(["author", "Unnamed: 0"], axis=1))
    print(predictions)
    accuracy = sum([1 if prediction == correct_class[i] else 0 for i, prediction in enumerate(predictions)]) / len(correct_class)
    count_ai = max(correct_class.count(1), 1)
    count_human = max(correct_class.count(-1), 1)
    true_ai = sum([1 if prediction == correct_class[i] and prediction == 1 else 0 for i, prediction in enumerate(predictions)]) / count_ai
    false_ai = sum([1 if prediction != correct_class[i] and prediction == 1 else 0 for i, prediction in enumerate(predictions)]) / count_human
    true_human = sum([1 if prediction == correct_class[i] and prediction == -1 else 0 for i, prediction in enumerate(predictions)]) / count_human
    false_human = sum([1 if prediction != correct_class[i] and prediction == -1 else 0 for i, prediction in enumerate(predictions)]) / count_ai
    unsure_ai = sum([1 if prediction == 0 and correct_class[i] == 1 else 0 for i, prediction in enumerate(predictions)]) / count_ai
    unsure_human = sum([1 if prediction == 0 and correct_class[i] == -1 else 0 for i, prediction in enumerate(predictions)]) / count_human
    unsure_total = sum([1 if prediction == 0 else 0 for prediction in predictions]) / len(predictions)
    print([true_ai, false_ai, true_human, false_human, unsure_ai, unsure_human])
    return {"accuracy":accuracy, "ai_true_positives":true_ai, "ai_false_positives":false_ai, "unsure":unsure_total}

def sem_performance():
    test_dataframe = pd.read_csv(os.path.join(STYLOMETRY_DIR, f"test_sem_distribution{nfeatures}.csv"))
    correct_class = []
    for i in range(test_dataframe.shape[0]):
        if used_authors[test_dataframe.iloc[i]["author"]] == "ai":
            correct_class.append(1)
        elif used_authors[test_dataframe.iloc[i]["author"]] == "human":
            correct_class.append(-1)
        else:
            correct_class.append(0)
    predictions = sem_model_prediction(test_dataframe.drop(["author", "Unnamed: 0"], axis=1))
    print(predictions)
    accuracy = sum([1 if prediction == correct_class[i] else 0 for i, prediction in enumerate(predictions)]) / len(correct_class)
    count_ai = max(correct_class.count(1), 1)
    count_human = max(correct_class.count(-1), 1)
    true_ai = sum([1 if prediction == correct_class[i] and prediction == 1 else 0 for i, prediction in enumerate(predictions)]) / count_ai
    false_ai = sum([1 if prediction != correct_class[i] and prediction == 1 else 0 for i, prediction in enumerate(predictions)]) / count_human
    true_human = sum([1 if prediction == correct_class[i] and prediction == -1 else 0 for i, prediction in enumerate(predictions)]) / count_human
    false_human = sum([1 if prediction != correct_class[i] and prediction == -1 else 0 for i, prediction in enumerate(predictions)]) / count_ai
    unsure_ai = sum([1 if prediction == 0 and correct_class[i] == 1 else 0 for i, prediction in enumerate(predictions)]) / count_ai
    unsure_human = sum([1 if prediction == 0 and correct_class[i] == -1 else 0 for i, prediction in enumerate(predictions)]) / count_human
    unsure_total = sum([1 if prediction == 0 else 0 for prediction in predictions]) / len(predictions)
    print([true_ai, false_ai, true_human, false_human, unsure_ai, unsure_human])
    return {"accuracy":accuracy, "ai_true_positives":true_ai, "ai_false_positives":false_ai, "unsure":unsure_total}


write_test_distributions()
print(char_performance())
print(sem_performance())
