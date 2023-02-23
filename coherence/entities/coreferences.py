from stanza.server import CoreNLPClient
from stanza.server.client import StartServer
from nlp.helpers import normalize_quotes

from stylometry.showcase.dep_colors import color_string
from visualization.text_chart import stacked_bar

def coref_annotation(text: str):
    with CoreNLPClient(start_server=StartServer.DONT_START, annotators=['tokenize','pos','lemma', 'ner', 'coref'], timeout=30000) as client:
        return client.annotate(normalize_quotes(text))

def coref_colored(annotation):
    coref_chains = [
        { (mention.sentenceIndex, idx) for mention in chain.mention for idx in range(mention.beginIndex, mention.endIndex) }
    for chain in annotation.corefChain]
    coref_colors = [1 / len(coref_chains) * n for n in range(len(coref_chains))]

    pretty_tokens = list()
    for n, sentence in enumerate(annotation.sentence):
        for m, token in enumerate(sentence.token):
            color_index = next((i for i, positions in enumerate(coref_chains) if (n, m) in positions), None)
            if color_index is None:
                pretty_tokens.append(token.originalText)
                continue
            pretty_tokens.append(color_string(token.originalText, (coref_colors[color_index], 0.8, 1)))

    return ' '.join(pretty_tokens).replace(' .', '.')

def coref_diagram(annotation):
    def text_for_mention_chain(chain):
        def text_for_mention(mention):
            sentence = annotation.sentence[mention.sentenceIndex]
            tokens = sentence.token[mention.beginIndex:mention.endIndex]
            return ' '.join([token.originalText for token in tokens])
        mentions = [text_for_mention(mention) for mention in chain.mention]
        mentions.sort(key=lambda m: len(m))
        return mentions[-1]

    def text_for_sentence(i):
        sentence = annotation.sentence[i]
        return annotation.text[sentence.characterOffsetBegin:sentence.characterOffsetEnd]

    sentence_range = range(len(annotation.sentence))
    sentence_windows = [
        [i for i in [n-1, n, n+1] if i in sentence_range] for n in sentence_range
    ]

    entity_occurences = {
        text_for_mention_chain(chain): [
            len([
                1 for mention in chain.mention if mention.sentenceIndex in rs# and mention.animacy == 'ANIMATE'
            ])
        for rs in sentence_windows]
    for chain in annotation.corefChain}

    chart, legend = stacked_bar([
        (text_for_sentence(n), [float(values[n]) for values in entity_occurences.values()])
    for n in sentence_range], [k for k in entity_occurences.keys()])

    return chart, legend
