from itertools import chain

from nltk.parse.corenlp import CoreNLPDependencyParser, DependencyGraph
from nltk.tree.tree import Tree
from nltk.tokenize import sent_tokenize

from nlp.helpers import lower_alnum

def _dep_tree(graph: DependencyGraph):
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

def _depth_search(tree: Tree, fun, is_root: bool = True):
    if is_root: fun(tree)
    for subtree in tree:
        fun(subtree)
        if type(subtree) is not Tree: continue
        _depth_search(subtree, fun, False)

def _add_sem_trigrams(tree: Tree, trigrams: dict[tuple, int]):
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

    _depth_search(tree, _process_node)

def sem_trigrams(text: str, parser: CoreNLPDependencyParser) -> dict[tuple, int]:
    sentences = [lower_alnum(sent) for sent in sent_tokenize(text)]
    parsed = parser.raw_parse_sents(sentences)

    trigrams = dict[tuple, int]()
    for sentence in parsed:
        graph: DependencyGraph = next(sentence)
        tree = _dep_tree(graph)
        if tree:
            _add_sem_trigrams(tree, trigrams)

    return trigrams
