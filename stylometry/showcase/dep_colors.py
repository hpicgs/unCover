import colorsys
from itertools import chain

from nltk.parse.dependencygraph import DependencyGraph
from nltk.tree.tree import Tree


def index_tree(graph: DependencyGraph):
    def _tree(address: int):
        node = graph.get_by_address(address)
        tag = str(node['address'])
        deps = sorted(chain.from_iterable(node['deps'].values()))
        if deps:
            return Tree(tag, [_tree(dep) for dep in deps])
        else:
            return tag

    node = graph.root
    if not node: return None

    deps = sorted(chain.from_iterable(node['deps'].values()))
    return Tree(str(node['address']), [_tree(dep) for dep in deps])

def color_string(text: str, hsv: tuple[float, float, float], hl: bool=False):
    rgb = colorsys.hsv_to_rgb(*hsv)
    rgb = [int(c * 255) for c in rgb]
    return ''.join([
        f'\033[1;{"4;" if hl else ""}38;2;{rgb[0]};{rgb[1]};{rgb[2]}m',
        text,
        '\033[0m'
    ])

def print_dep_colors(dep_graph: DependencyGraph):
    tree = index_tree(dep_graph)
    if not tree: return

    root_address = dep_graph.root['address']
    colors: dict[int, tuple[float, float, float]] = {
        root_address: (0, 0, 0)
    }

    def _find_colors(tree: Tree, hue_range: tuple[float, float] = (0, 1), sat: float = 0.8, decline: float = 1):
        left, right = hue_range
        interval = float(right - left) / len(tree)
        for n, c in enumerate(tree):
            idx = int(c.label() if type(c) is Tree else c)
            l = left + interval * n
            r = l + interval
            colors[idx] = ((l + r) / 2, sat, 0.9)
            if type(c) is Tree:
                _find_colors(c, (l, r), sat * decline)

    _find_colors(tree, decline=0.7)
    print(' '.join([
        color_string(dep_graph.get_by_address(i)['word'], colors[i], hl=i == root_address)
    for i in range(1, len(colors) + 1)]))
