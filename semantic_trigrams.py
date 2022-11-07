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

def depth_search(tree: Tree, fun, is_root: bool = True):
    if is_root: fun(tree)
    for subtree in tree:
        fun(subtree)
        if type(subtree) is not Tree: continue
        depth_search(subtree, fun, False)

def sem_trigrams(tree: Tree) -> dict[tuple, int]:
    trigrams = dict[tuple, int]()
    def _add_or_increment(tree: tuple):
        if tree in trigrams:
            trigrams[tree] += 1
        else:
            trigrams[tree] = 1

    def _str_label(node) -> str:
        return node.label() if type(node) is Tree else node

    def _process_node(node):
        if type(node) is not Tree: return
        for n, a in enumerate(node):
            a_label = _str_label(a)
            # horizontal (node, (a, b))
            if n < len(node) - 1:
                tree = (node.label(), frozenset((a_label, _str_label(node[n+1]))))
                _add_or_increment(tree)
            # vertical (node (a, (c)))
            if type(a) is not Tree: continue
            for c in a:
                # single tuple `(x)` is resolved to `x` so we use `(x,)`
                tree = (node.label(), (a_label, (_str_label(c),)))
                _add_or_increment(tree)

    depth_search(tree, _process_node)
    return trigrams

if __name__ == '__main__':
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
            print(sem_trigrams(tree))
