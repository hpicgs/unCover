from itertools import chain

from nltk.parse.corenlp import DependencyGraph
from nltk.tree.tree import Tree

def word_pos_tree(graph: DependencyGraph):
    def _tree(address: int):
        node = graph.get_by_address(address)
        word = f'({node["word"]}, {node["ctag"]})'
        deps = sorted(chain.from_iterable(node['deps'].values()))
        if deps:
            return Tree(word, [_tree(dep) for dep in deps])
        else:
            return word

    node = graph.root
    if not node: return None

    tag = f'({node["word"]}, {node["ctag"]})'
    deps = sorted(chain.from_iterable(node['deps'].values()))
    return Tree(tag, [_tree(dep) for dep in deps])
