from stanza.server import CoreNLPClient
from stanza.server.client import StartServer
from pandas import DataFrame
import pandas_bokeh
pandas_bokeh.output_file("plot.html")

from stylometry.showcase.dep_colors import color_string

def coreference(text: str):
    with CoreNLPClient(start_server=StartServer.DONT_START, annotators=['tokenize','pos','lemma', 'ner', 'coref'], timeout=30000) as client:
        annotation = client.annotate(text)
        print(coref_colored(annotation))
        coref_diagram(annotation)

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

    def limit_line_length(text, max_length=160):
        words = text.split(' ')
        lines = ['']
        for word in words:
            if len(lines[-1]) + len(word) < max_length:
                lines[-1] += f' {word}'
            else:
                lines.append(word)
        return '\n'.join(lines)

    sentence_range = range(len(annotation.sentence))
    sentence_windows = [
        [i for i in [n-1, n, n+1] if i in sentence_range] for n in sentence_range
    ]

    entity_occurences = {
        text_for_mention_chain(chain): [
            len([
                1 for mention in chain.mention if mention.sentenceIndex in rs and mention.animacy == 'ANIMATE'
            ])
        for rs in sentence_windows]
    for chain in annotation.corefChain}
    
    sentence_labels = [limit_line_length(text_for_sentence(i)) for i in reversed(sentence_range)]
    df = DataFrame(
            { name: occurences for name, occurences in reversed(entity_occurences.items()) if any(occurences)
     }, index=sentence_labels)

    df.plot_bokeh.barh(
        stacked=True,
        xlabel='Number of entity occurences in given sentence and its neighbors',
        fontsize_ticks=15,
        fontsize_label=15,
        figsize=(2000, 1200),
        hovertool=False
    )
