from itertools import chain
import os
from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPServer, DependencyGraph
from nltk.tree.tree import Tree

def dep_tree(graph: DependencyGraph):
    def _tree(address: int):
        node = graph.get_by_address(address)
        word = node['ctag']
        deps = sorted(chain.from_iterable(node['deps'].values()))
        if deps:
            return Tree(word, [_tree(dep) for dep in deps])
        else:
            return word

    node = graph.root
    if not node: return None

    tag = node['ctag']
    deps = sorted(chain.from_iterable(node['deps'].values()))
    return Tree(tag, [_tree(dep) for dep in deps])

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
    tree = dep_tree(dep_graph)
    if tree:
        tree.pretty_print()
