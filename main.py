import os
import sys

from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPServer
from nltk.parse.dependencygraph import DependencyGraph
from nltk.tokenize import sent_tokenize

from nlp.helpers import normalize
from trigrams.char_trigrams import char_trigrams
from trigrams.logistic_regression import trigram_distribution
from trigrams.semantic_trigrams import dep_tree, get_text, sem_trigrams

if __name__ == '__main__':
    from definitions import ROOT_DIR
    stanford_dir = os.path.join(ROOT_DIR, 'models', 'stanford-corenlp-4.5.1')
    jars = (
       os.path.join(stanford_dir, 'stanford-corenlp-4.5.1.jar'),
       os.path.join(stanford_dir, 'stanford-corenlp-4.5.1-models.jar'),
    )

    text = ''.join(sys.stdin.readlines())
    sentences = [normalize(sent) for sent in sent_tokenize(text)]

    with CoreNLPServer(*jars):

        parser = CoreNLPDependencyParser()
        parsed = parser.raw_parse_sents(sentences)
        sem_grams = list[dict[tuple, int]]()
        char_grams = list[dict[str, int]]()

        for sent in parsed:
            graph: DependencyGraph = next(sent)
            tree = dep_tree(graph)
            if tree:
                sem_grams.append(sem_trigrams(tree))
                char_grams.append(char_trigrams(get_text(graph)))

    sem_dist = trigram_distribution(sem_grams)
    print(sem_dist)
    print(trigram_distribution(char_grams))
