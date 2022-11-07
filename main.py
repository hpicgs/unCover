import os

from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPServer, DependencyGraph

from trigrams.semantic_trigrams import dep_tree, sem_trigrams

if __name__ == '__main__':
    from definitions import ROOT_DIR
    stanford_dir = os.path.join(ROOT_DIR, 'models', 'stanford-corenlp-4.5.1')
    jars = (
       os.path.join(stanford_dir, 'stanford-corenlp-4.5.1.jar'),
       os.path.join(stanford_dir, 'stanford-corenlp-4.5.1-models.jar'),
    )

    with CoreNLPServer(*jars):
        parser = CoreNLPDependencyParser()
        dep_graph: DependencyGraph = next(parser.raw_parse('Use StanfordParser to parse multiple sentences'))
        tree = dep_tree(dep_graph)
        if tree:
            tree.pretty_print()
            print(sem_trigrams(tree))
