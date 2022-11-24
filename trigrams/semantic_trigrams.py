from itertools import chain

from nltk.parse.corenlp import DependencyGraph
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

def get_text(graph: DependencyGraph) -> str:
    return ' '.join([
        graph.get_by_address(i)['word'] for i in range(1, len(graph.nodes))
    ])

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
