import os

from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPServer, DependencyGraph

from trigrams.showcase.dep_colors import print_dep_colors

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

        dep_graph.tree().pretty_print()
        print_dep_colors(dep_graph)
