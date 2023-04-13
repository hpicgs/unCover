import argparse
import os
import statistics
import yaml

from tem.model import TopicEvolution
from tem.nlp import docs_from_period, merge_short_periods
from tem.process import get_topic_evolution

def te_analysis_data(te: TopicEvolution) -> dict[str, float] | None:
    if len(te.periods) < 2: return None

    node_count_by_id: dict[int, int] = {}
    for period in te.periods:
        for topic in period.topics:
            node_count_by_id[topic.id] = 1 if topic.id not in node_count_by_id else node_count_by_id[topic.id] + 1
    node_count = sum((count for count in node_count_by_id.values()))
    if node_count == 0: return None

    period_topic_ids = [
            { topic.id for topic in period.topics }
    for period in te.periods]
    period_has_incoming = [
            n > 0 and any((i in period_topic_ids[n - 1] for i in ids))
    for n, ids in enumerate(period_topic_ids)]

    longest_period_path = max((
        len({ n for n, ids in enumerate(period_topic_ids) if topic_id in ids })
    for topic_id in node_count_by_id.keys()))

    return {
        'abs(1 - n_ids/n_nodes)': abs(1 - len(node_count_by_id) / node_count),
        'largest group / n_nodes': max((count for count in node_count_by_id.values())) / node_count,
        'n_{periods with incoming} / (n_periods - 1)': sum(period_has_incoming) / (len(te.periods) - 1),
        'n_{longest connected periods} / (n_periods)': longest_period_path / len(te.periods),
        'median n_words per topic': statistics.median([len(words) for period in te.periods for topic in period.topics for words in topic.words]),
    }

def te_analysis_img(text: str) -> bytes | None:
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

    data = te_analysis_data(te)
    if not data: return None

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
    '''.format(key=key, value=value) for key, value in data.items()])))

    return graph.pipe(format='png')

def analyze_db(data: list[dict[str, str]]):
    for n, item in enumerate(data):
        if not item['author'] or not item['text']: continue

        author = 'human' if item['author'].startswith('http') else item['author']
        directory = os.path.join('out', author)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f'{n}.png')

        print(f'\r\033[Kanalyzing text {n + 1} of {len(data)}. destination: {path}', end='')
        img = te_analysis_img(item['text'])
        if not img: continue

        with open(path, 'wb') as fp:
            fp.write(img)

    print('\ndone!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TEM analyzer')
    parser.add_argument('file', help='input text or yaml database file')
    args = parser.parse_args()

    if args.file.endswith('.txt'):
        with open(args.file, 'r') as fp:
            text = fp.read()
        with open(args.file + ".png", 'wb') as fp:
            fp.write(te_analysis_img(text))
    elif args.file.endswith('.yaml'):
        print('reading yaml database')
        with open(args.file, 'r') as fp:
            data = yaml.safe_load(fp.read())
            analyze_db(data)
    else:
        parser.error('Unknown file type')
