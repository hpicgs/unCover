import argparse
import statistics

from tem.model import TopicEvolution
from tem.nlp import docs_from_period, merge_short_periods
from tem.process import get_topic_evolution

def te_analysis_data(te: TopicEvolution) -> dict[str, float]:
    node_count_by_id: dict[int, int] = {}
    for period in te.periods:
        for topic in period.topics:
            node_count_by_id[topic.id] = 1 if topic.id not in node_count_by_id else node_count_by_id[topic.id] + 1
    node_count = sum((count for count in node_count_by_id.values()))

    return {
        'n_ids/n_nodes': len(node_count_by_id) / node_count,
        'largest group / n_nodes': max((count for count in node_count_by_id.values())) / node_count,
        'mean n_words per topic': statistics.mean([len(words) for period in te.periods for topic in period.topics for words in topic.words])
    }

def te_analysis_img(text: str) -> bytes:
    corpus = [docs_from_period(line) for line in text.split('\n') if len(line) > 0]
    corpus = merge_short_periods(corpus, min_docs=2)
    te = get_topic_evolution(
        corpus,
        c=0.5,
        alpha=0,
        beta=-1,
        gamma=0,
        delta=1,
        theta=2,
        mergeThreshold=100,
        evolutionThreshold=100
    )

    graph = te.graph()
    graph.attr(label='''<<FONT POINT-SIZE="48" COLOR="white">
        <TABLE BGCOLOR="red" ALIGN="left" BORDER="0" CELLBORDER="0" CELLSPACING="40">
            {rows}
        </TABLE>
    </FONT>>'''.format(rows='\n'.join(['''
        <TR>
            <TD ALIGN="left">{key}</TD>
            <TD ALIGN="right">{value}</TD>
        </TR>
    '''.format(key=key, value=value) for key, value in te_analysis_data(te).items()])))

    return graph.pipe(format='png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TEM analyzer')
    parser.add_argument('file', help='input text file')
    args = parser.parse_args()

    with open(args.file, 'r') as fp:
        text = fp.read()
    with open(args.file + ".png", 'wb') as fp:
        fp.write(te_analysis_img(text))
