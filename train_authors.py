import pickle
from nltk.parse.corenlp import CoreNLPDependencyParser

from definitions import STYLOMETRY_DIR, DATABASE_AUTHORS_PATH
from stylometry.char_trigrams import char_trigrams
from stylometry.semantic_trigrams import sem_trigrams
from stylometry.logistic_regression import trigram_distribution, logistic_regression
from database.mock_database import DatabaseAuthorship
import os, zipfile, sys

if __name__ == "__main__":
    if not os.path.isfile(DATABASE_AUTHORS_PATH):
        if os.path.isfile("mock-authors.zip"):
            with zipfile.ZipFile("mock-authors.zip", "r") as zf:
                zf.extractall(DATABASE_FILES_PATH)
        else:
            print("ERROR: no zipped database was provided")
            sys.exit(1)

    authors = DatabaseAuthorship.get_authors()
    training_data = []
    os.makedirs(STYLOMETRY_DIR, exist_ok=True)

    for author in authors:
        full_article_list = [(article["text"], author) for article in DatabaseAuthorship.get_articles_by_author(author)]
        training_data += full_article_list[:int(len(full_article_list)*0.8)]

    print('number of training articles:', len(training_data))

    char_grams = [char_trigrams(article_tuple[0]) for article_tuple in training_data]
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    sem_grams = [sem_trigrams(article_tuple[0], parser) for article_tuple in training_data]
    
    character_distribution = trigram_distribution(char_grams)
    semantic_distribution = trigram_distribution(sem_grams)
    print(character_distribution)
    print(semantic_distribution)
    character_distribution.to_csv("char_distribution.csv")
    semantic_distribution.to_csv("sem_distribution.csv")

    for author in authors:
        truth_table = [1 if article_tuple[1] == author else 0 for article_tuple in training_data]
        with open(os.path.join(STYLOMETRY_DIR, author.replace('/', '_') + '_char.pickle'), 'wb') as f:
            pickle.dump(logistic_regression(character_distribution, truth_table), f)
        with open(os.path.join(STYLOMETRY_DIR, author.replace('/', '_') + '_sem.pickle'), 'wb') as f:
            pickle.dump(logistic_regression(semantic_distribution, truth_table), f)
    print('TRAINING DONE!')
