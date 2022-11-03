import os
jars_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'jars'
)
os.environ['STANFORD_PARSER'] = jars_path
os.environ['STANFORD_MODELS'] = jars_path

from nltk.parse.stanford import StanfordParser
from nltk.tree import Tree

def adjust_tree(tree: Tree) -> Tree:
    positions = tree.treepositions('leaves')
    parent_nodes = { (0,): Tree('ROOT', []) }
    word_node_positions = list()

    for pos in positions:
        word_node_pos = pos[:-1]
        word_node = tree[word_node_pos]
        # filter out ,. etc
        if not word_node.label().isalpha(): continue
        # "absorbe parent until before we have multiple children"
        adjusted_pos = word_node_pos
        while len(tree[adjusted_pos]) == 1 and len(adjusted_pos) > 0:
            adjusted_pos = adjusted_pos[:-1]
        # when multiple nodes end up with the same adjusted_pos, they
        # never have children so overwriting this in the dict doesn't matter
        # (left to the reader to prove :p)
        parent_nodes[adjusted_pos] = word_node
        word_node_positions.append((adjusted_pos, word_node))
    
    for pos, node in word_node_positions:
        parent_pos = pos[:-1]
        # same "absorbing irrelevant parents"
        while parent_pos not in parent_nodes and len(parent_pos) > 0:
            parent_pos = parent_pos[:-1]
        if parent_pos not in parent_nodes:
            continue
        parent_nodes[parent_pos].append(node)

    # return ROOT from above
    return parent_nodes[(0,)]

if __name__ == '__main__':
    sentence = 'Use StanfordParser to parse multiple sentences.'
    text = 'Use StanfordParser to parse multiple sentences. The StandfordParser takes multiple sentences as a list where each sentence is a list of words. Each sentence will be automatically tagged with this StanfordParser instance’s tagger. If whitespaces exists inside a token, then the token will be treated as separate tokens.'

    parser = StanfordParser(model_path=os.path.join(jars_path, 'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'))
    tree: Tree = next(parser.raw_parse(sentence))
    print('ORIGINAL TREE ------------------------------------------------------------------')
    tree.pretty_print()
    adjusted = adjust_tree(tree)
    print('ADJUSTED TREE ------------------------------------------------------------------')
    adjusted.pretty_print()
