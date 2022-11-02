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
    
    def _is_word_node(node: Tree) -> bool:
        return len(node.leaves()) == 1 and node.height() == 2

    def _first_word_node(tree: Tree) -> Tree | None:
        for subtree in tree:
            if type(subtree) is not Tree: continue
            if _is_word_node(subtree): return subtree
            else:
                recursion = _first_word_node(subtree)
                if recursion: return recursion
        return None

    #def _build_trigrams(tree: Tree, trigrams: dict[Tree, int]):
    #    for subtree in tree:

    filtered = [
        line[:-1] for line in str(next(parser.raw_parse(text))).split('\n')
     if len(line) > 0 and line[-1] == ')']
    for f in filtered:
        print(f)

sentence = 'Use StanfordParser to parse multiple sentences.'
text = 'Takes multiple sentences as a list where each sentence is a list of words. Each sentence will be automatically tagged with this StanfordParser instance’s tagger. If whitespaces exists inside a token, then the token will be treated as separate tokens.'
# x = char_trigrams()
# 
# print(x)

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
