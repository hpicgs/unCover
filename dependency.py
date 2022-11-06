import os
from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPServer, DependencyGraph
# from nltk.parse.dependencygraph import dot2img

stanford_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'models',
    'stanford-corenlp-4.5.1'
)
jars = (
   os.path.join(stanford_dir, 'stanford-corenlp-4.5.1.jar'),
   os.path.join(stanford_dir, 'stanford-corenlp-4.5.1-models.jar'),
)

with CoreNLPServer(*jars):
    parser = CoreNLPDependencyParser()
    dep_graph: DependencyGraph = next(parser.raw_parse('Use StanfordParser to parse multiple sentences'))

    # dot_string = dep_graph.to_dot()
    # dot2img(dot_string, t='png')

    dep_graph.tree().pretty_print()
