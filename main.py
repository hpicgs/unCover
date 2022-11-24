import os
import sys

from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPServer
from nltk.tokenize import sent_tokenize

from nlp.helpers import normalize
from trigrams.char_trigrams import add_char_trigrams
from trigrams.extraction import trigram_distribution
from trigrams.semantic_trigrams import add_sem_trigrams, dep_tree

if __name__ == '__main__':
    from definitions import ROOT_DIR
    stanford_dir = os.path.join(ROOT_DIR, 'models', 'stanford-corenlp-4.5.1')
    jars = (
       os.path.join(stanford_dir, 'stanford-corenlp-4.5.1.jar'),
       os.path.join(stanford_dir, 'stanford-corenlp-4.5.1-models.jar'),
    )

    text = ''.join(sys.stdin.readlines())
    sentences = [normalize(sent) for sent in sent_tokenize(text)]

    char_grams = dict[str, int]()
    add_char_trigrams(''.join(sentences), char_grams)

    with CoreNLPServer(*jars):

        parser = CoreNLPDependencyParser()
        parsed = parser.raw_parse_sents(sentences)

        sem_grams = dict[tuple, int]()

        for sent in parsed:
            tree = dep_tree(next(sent))
            if tree:
                add_sem_trigrams(tree, sem_grams)

    print(trigram_distribution(sem_grams))
    print(trigram_distribution(char_grams))
