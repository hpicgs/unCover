def char_trigrams(text: str) -> dict[str, int]:
    normal = ''.join([
        char.lower()
    for char in text if char.isalnum() or char.isspace()])

    ret = dict[str, int]()
    for i in range(len(normal) - 2):
        trigram = normal[i:i + 3]
        ret[trigram] = 1 if trigram not in ret else ret[trigram] + 1

    return ret

# https://stackoverflow.com/a/22269678/17614473
import os
jars_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'jars'
)
os.environ['STANFORD_PARSER'] = jars_path
os.environ['STANFORD_MODELS'] = jars_path

from nltk.parse.stanford import StanfordParser

def syntactic_trigrams(text: str):
    parser = StanfordParser(model_path=os.path.join(jars_path, 'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'))
    for tree in parser.raw_parse(text):
        print(tree)

sentence = 'Use StanfordParser to parse multiple sentences.'
text = 'Takes multiple sentences as a list where each sentence is a list of words. Each sentence will be automatically tagged with this StanfordParser instance’s tagger. If whitespaces exists inside a token, then the token will be treated as separate tokens.'
# x = char_trigrams()
# 
# print(x)

syntactic_trigrams(sentence)
