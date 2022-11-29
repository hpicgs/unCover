from pathlib import Path
import pickle
from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPServer

from definitions import STANFORD_JARS
from stylometry.char_trigrams import char_trigrams
from stylometry.semantic_trigrams import sem_trigrams
from stylometry.logistic_regression import trigram_distribution, logistic_regression

from database.mock_database import DatabaseAuthorship

import sys

if __name__ == "__main__":
    #https://www.theguardian.com/profile/martin-chulov
    #https://www.theguardian.com/profile/leyland-cecco
    #https://www.theguardian.com/profile/hannah-ellis-petersen
    authors = [
        "https://www.theguardian.com/profile/martin-chulov",
        "https://www.theguardian.com/profile/leyland-cecco",
        "https://www.theguardian.com/profile/hannah-ellis-petersen"
        ]
    training_data = []
    Path("models/stylometrie").mkdir(parents=True, exist_ok=True)

    for author in authors:
        full_article_list = [(article["text"], author) for article in DatabaseAuthorship.get_articles_by_author(author)]
        training_data += full_article_list[:int(len(full_article_list)*0.6)]

    print(len(training_data))

    char_grams = [char_trigrams(article_tuple[0]) for article_tuple in training_data]
    parser = CoreNLPDependencyParser(url='http://localhost:9000')
    sem_grams = [sem_trigrams(article_tuple[0], parser) for article_tuple in training_data]
    
    character_distribution = trigram_distribution(char_grams)
    semantic_distribution = trigram_distribution(sem_grams)
    print(character_distribution)
    print(semantic_distribution)

    for author in authors:
        truth_table = [1 if article_tuple[1] == author else 0 for article_tuple in training_data]
        with open('models/stylometrie/'+author.replace('/', '_')+'_char.pickle', 'wb') as f:
            pickle.dump(logistic_regression(character_distribution, truth_table), f)
        with open('models/stylometrie/'+author.replace('/', '_') + '_char.pickle', 'wb') as f:
            pickle.dump(logistic_regression(semantic_distribution, truth_table), f)
    print('TRAINING DONE!')
