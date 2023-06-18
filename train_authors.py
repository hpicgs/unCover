from nltk.parse.corenlp import CoreNLPDependencyParser

from definitions import STYLOMETRY_DIR, DATABASE_AUTHORS_PATH, DATABASE_GEN_PATH
from stylometry.char_trigrams import char_trigrams
from stylometry.semantic_trigrams import sem_trigrams
from stylometry.logistic_regression import trigram_distribution, logistic_regression
from database.mock_database import DatabaseAuthorship, DatabaseGenArticles
import os, sys, pickle, argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nfeatures", action="store", required=False, type=int, default=100, help="number of char trigram & semantic trigram features used in the distribution")
    parser.add_argument("--minarticles", action="store", required=False, type=int, default=50, help="minimum number of articles required for training on a specific author/model")
    args = parser.parse_args()
    nfeatures = args.nfeatures
    minarticles = args.minarticles
    
    if not os.path.isfile(DATABASE_AUTHORS_PATH):
         print("Error: no database for human authors was provided")
         sys.exit(1)
    if not os.path.isfile(DATABASE_GEN_PATH):
        print("Error: no database for machine authors was provided")
        sys.exit(1)

    authors = DatabaseAuthorship.get_authors()
    machines = DatabaseGenArticles.get_methods()
    print("Human authors:")
    print(authors)
    print("Language models:")
    print(machines)
    training_data = []
    os.makedirs(STYLOMETRY_DIR, exist_ok=True)

    trainable_authors = []
    for author in authors:
        full_article_list = [(article["text"], author) for article in DatabaseAuthorship.get_articles_by_author(author)]
        training_data += full_article_list[:int(len(full_article_list)*0.8)]
        if len(full_article_list) >= minarticles:
            print(f"chose author: {author}")
            trainable_authors.append(author)
    trainable_machines = []
    for machine in machines:
        full_article_list = [(article["text"], machine) for article in DatabaseMachines.get_articles_by_author(machine)]
        training_data += full_article_list[:int(len(full_article_list)*0.8)]
        if len(full_article_list) >= minarticles:
            print("chose language model: {machine}")
            trainable_machines.append(machine)


    print('number of training articles:', len(training_data))
    print('trainable authors:')
    print(trainable_authors)
    print('trainable language models:')
    print(trainable_machines)

    char_grams = [char_trigrams(article_tuple[0]) for article_tuple in training_data]
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    sem_grams = [sem_trigrams(article_tuple[0], parser) for article_tuple in training_data]
    
    character_distribution = trigram_distribution(char_grams, nfeatures)
    semantic_distribution = trigram_distribution(sem_grams, nfeatures)
    character_distribution.to_csv("models/stylometry/char_distribution{nfeatures}.csv")
    semantic_distribution.to_csv("models/stylometry/sem_distribution{nfeatures}.csv")

    training_subjects = trainable_authors + trainable_machines
    for author in training_subjects:
        truth_table = [1 if author == article_tuple[1] else 0 for article_tuple in training_data]
        print(truth_table)
        with open(os.path.join(STYLOMETRY_DIR, f"{author.replace('/', '_')_char{nfeatures}.pickle"), "wb") as f:
            pickle.dump(logistic_regression(character_distribution, truth_table), f)
        with open(os.path.join(STYLOMETRY_DIR, f"author.replace('/', '_')_sem{nfeatures}.pickle"), "wb") as f:
            pickle.dump(logistic_regression(semantic_distribution, truth_table), f)
    print('TRAINING DONE!')
