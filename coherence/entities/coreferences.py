from stanza.server import CoreNLPClient
from stanza.server.client import StartServer

from stylometry.showcase.dep_colors import color_string

def coreference():
    text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people. The people love him for giving them oranges."
    with CoreNLPClient(start_server=StartServer.DONT_START, annotators=['tokenize','pos','lemma', 'ner', 'coref'], timeout=30000) as client:
        ann = client.annotate(text)
        coref_chains = [
            { (mention.sentenceIndex, idx) for mention in chain.mention for idx in range(mention.beginIndex, mention.endIndex) }
        for chain in ann.corefChain]
        coref_colors = [1 / len(coref_chains) * n for n in range(len(coref_chains))]

        pretty_tokens = list()
        for n, sentence in enumerate(ann.sentence):
            for m, token in enumerate(sentence.token):
                color_index = next((i for i, positions in enumerate(coref_chains) if (n, m) in positions), None)
                if color_index is None:
                    pretty_tokens.append(token.originalText)
                    continue
                pretty_tokens.append(color_string(token.originalText, (coref_colors[color_index], 0.8, 1)))

        print(' '.join(pretty_tokens).replace(' .', '.'))
