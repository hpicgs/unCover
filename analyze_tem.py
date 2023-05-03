import argparse
import csv
import os
import statistics

import yaml

from tem.model import TopicEvolution
from tem.process import get_default_te

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

def te_annotated_img(te: TopicEvolution, data: dict[str, float]) -> bytes:
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

def analyze_db(db: list[dict[str, str]], db_name: str):
    handles = list()
    csv_by_author = dict()

    for n, item in enumerate(db):
        if not item['author'] or not item['text']: continue

        author = 'human' if item['author'].startswith('http') else item['author']
        print(f'\r\033[Kanalyzing text by {author} ({n + 1} of {len(db)})', end='')

        te = get_default_te(item['text'])
        data = te_analysis_data(te)
        if not data: continue

        directory = os.path.join('out', author)

        if author not in csv_by_author:
            os.makedirs(directory, exist_ok=True)
            fp = open(os.path.join(directory, '_stats.csv'), 'w')
            handles.append(fp)
            csv_by_author[author] = csv.writer(fp)
            csv_by_author[author].writerow(['source'] + list(data.keys()))
        
        item_name = f'{db_name}-{n}'
        csv_by_author[author].writerow([item_name] + list(data.values()))

        img_path = os.path.join(directory, f'{item_name}.png')
        img = te_annotated_img(te, data)
        with open(img_path, 'wb') as fp:
            fp.write(img)

    for fp in handles: fp.close()

    print('\ndone!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TEM analyzer')
    parser.add_argument('file', help='input yaml database file')
    args = parser.parse_args()

    if args.file.endswith('.yaml'):
        print('reading yaml database')
        with open(args.file, 'r') as fp:
            data = yaml.safe_load(fp.read())
            analyze_db(data, args.file.split('/')[-1].split('.')[0])
    else:
        parser.error('Unknown file type')
