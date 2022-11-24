import argparse

from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPServer

from definitions import STANFORD_JARS
from stylometry.char_trigrams import char_trigrams
from stylometry.logistic_regression import trigram_distribution
from stylometry.semantic_trigrams import sem_trigrams

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('text_files', type=str, nargs='*')
    args = argparser.parse_args()

    texts = list[str]()
    for text_file in args.text_files:
        with open(text_file, 'r') as fp:
            texts.append(fp.read())

    char_grams = [char_trigrams(text) for text in texts]
    with CoreNLPServer(*STANFORD_JARS):
        parser = CoreNLPDependencyParser()
        sem_grams = [sem_trigrams(text, parser) for text in texts]

    print(trigram_distribution(sem_grams))
    print(trigram_distribution(char_grams))
