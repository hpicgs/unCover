import pickle, time, subprocess
import nltk
from nltk.parse.corenlp import CoreNLPDependencyParser

from definitions import STYLOMETRY_DIR, DATABASE_AUTHORS_PATH
from stylometry.char_trigrams import char_trigrams
from stylometry.semantic_trigrams import sem_trigrams
from stylometry.logistic_regression import trigram_distribution, logistic_regression
from database.mock_database import DatabaseAuthorship
import os, zipfile, sys

#needed because docker cannot run two bash tasks at the same time
def preparation():
    nltk.download("punkt")
    server  = subprocess.Popen(["java", "-mx4g", "-cp", "models/stanford-corenlp-4.5.1/*", "edu.stanford.nlp.pipeline.StanfordCoreNLPServer", "-port", "9000", "-timeout", "15000"], stdout=subprocess.PIPE)
    time.sleep(40)
    return server


if __name__ == "__main__":
    #the docker container needs to download this separately
    preparation()
    if not os.path.isfile(DATABASE_AUTHORS_PATH):
        if os.path.isfile("mock-authors.zip"):
            with zipfile.ZipFile("mock-authors.zip", "r") as zf:
                zf.extractall(DATABASE_FILES_PATH)
        else:
            print("ERROR: no zipped database was provided")
            sys.exit(1)

    authors = DatabaseAuthorship.get_authors()
    print(authors)
    training_data = []
    os.makedirs(STYLOMETRY_DIR, exist_ok=True)

    trainable_authors = []
    for author in authors:
        full_article_list = [(article["text"], author) for article in DatabaseAuthorship.get_articles_by_author(author)]
        training_data += full_article_list[:int(len(full_article_list)*0.8)]
        if len(full_article_list) > 2:
            trainable_authors.append(author)


    print('number of training articles:', len(training_data))

    char_grams = [char_trigrams(article_tuple[0]) for article_tuple in training_data]
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    sem_grams = [sem_trigrams(article_tuple[0], parser) for article_tuple in training_data]
    
    character_distribution = trigram_distribution(char_grams)
    semantic_distribution = trigram_distribution(sem_grams)
    #print(character_distribution)
    #print(semantic_distribution)
    character_distribution.to_csv("models/stylometry/char_distribution.csv")
    semantic_distribution.to_csv("models/stylometry/sem_distribution.csv")

    for author in trainable_authors:
        truth_table = [1 if author in article_tuple[1] else 0 for article_tuple in training_data]
        print(truth_table)
        with open(os.path.join(STYLOMETRY_DIR, author.replace('/', '_') + '_char.pickle'), 'wb') as f:
            pickle.dump(logistic_regression(character_distribution, truth_table), f)
        with open(os.path.join(STYLOMETRY_DIR, author.replace('/', '_') + '_sem.pickle'), 'wb') as f:
            pickle.dump(logistic_regression(semantic_distribution, truth_table), f)
    print('TRAINING DONE!')
