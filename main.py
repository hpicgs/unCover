def char_trigrams(text: str) -> dict[str, int]:
    normal = ''.join([
        char.lower()
    for char in text if char.isalnum() or char.isspace()])

    ret = dict[str, int]()
    for i in range(len(normal) - 2):
        trigram = normal[i:i + 3]
        ret[trigram] = 1 if trigram not in ret else ret[trigram] + 1

    return ret

# install stanford parser:
# https://stackoverflow.com/a/22269678/17614473
import os
jars_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'jars'
)
os.environ['STANFORD_PARSER'] = jars_path
os.environ['STANFORD_MODELS'] = jars_path

from nltk.parse.stanford import StanfordParser
from nltk.tree import Tree

def syntactic_trigrams(text: str):
    parser = StanfordParser(model_path=os.path.join(jars_path, 'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'))
    tree: Tree = next(parser.raw_parse(text))
    print('ORIGINAL TREE ------------------------------------------------------------------')
    tree.pretty_print()
    positions = tree.treepositions('leaves')

    parent_nodes = {
        (0,): Tree('ROOT', [])
    }
    word_node_positions = list()
    for pos in positions:
        leaf_node_pos = pos[:-1]
        leaf_node = tree[leaf_node_pos]
        if not leaf_node.label().isalpha(): continue
        new_pos = leaf_node_pos
        while len(tree[new_pos]) == 1 and len(new_pos) > 0:
            new_pos = new_pos[:-1]
        parent_nodes[new_pos] = leaf_node
        word_node_positions.append((new_pos, leaf_node))
    
    for w in word_node_positions:
        parent_pos = w[0][:-1]
        while parent_pos not in parent_nodes and len(parent_pos) > 0:
            parent_pos = parent_pos[:-1]
        if parent_pos not in parent_nodes:
            continue
        parent_nodes[parent_pos].append(w[1])

    print('ADJUSTED TREE ------------------------------------------------------------------')
    parent_nodes[(0,)].pretty_print()

sentence = 'Use StanfordParser to parse multiple sentences.'
text = 'Use StanfordParser to parse multiple sentences. The StandfordParser takes multiple sentences as a list where each sentence is a list of words. Each sentence will be automatically tagged with this StanfordParser instance’s tagger. If whitespaces exists inside a token, then the token will be treated as separate tokens.'

syntactic_trigrams(sentence)



'''
                         S                                 
                     ____|_______________________________   
                    VP                                   | 
  __________________|____                                |  
 |                       S                               | 
 |         ______________|____                           |  
 |        |                   VP                         | 
 |        |          _________|_____                     |  
 |        |         |               VP                   | 
 |        |         |     __________|______              |  
 |        NP        |    |                 NP            | 
 |        |         |    |           ______|______       |  
 VB      NNP        TO   VB         JJ           NNS     . 
 |        |         |    |          |             |      |  
Use StanfordParser  to parse     multiple     sentences  . 

    (VB, Use)
        |________________________
        |                        |
(NNP, StanfordParser)         (TO, to)
                                 |
                            (VB, parse)
                           ______|__________      
                          |                 |
                  (JJ, multiple)    (NNS, sentences)
'''
