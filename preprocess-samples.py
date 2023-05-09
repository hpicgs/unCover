import argparse
import base64
import csv
import os
import random
from typing import Literal

import dominate
# pyright: reportWildcardImportFromLibrary=false
from dominate.tags import *
from dominate.util import raw
import yaml

from coherence.entities.coreferences import coref_annotation, coref_diagram
from main import get_prediction
from nlp.helpers import normalize_quotes
from stylometry.logistic_regression import predict_author
from tem.model import TopicEvolution
from tem.process import get_default_te
from train_tem_metrics import predict_from_tem_metrics

def html_results(
    text: str,
    author: Literal[0, 1, -1],
    te: TopicEvolution,
    entity_diagram: tuple[div, div],
    title: str = 'unBlock Analysis',
) -> str:
    te_img_data = base64.encodebytes(te.graph().pipe(format='png')).decode('ascii')

    doc = dominate.document(title=title)
    with doc.head:
        # raw prevents escaping of `>` character
        # CSS `#container > * + *` is like tailwind's `space-y` class
        style(raw('''\
            body { padding: 0 4rem; }
            h2 {
                border-top: 1px solid lightgray;
                padding-top: 1rem
            }

            #container { display: flex; }
            #container > * + * { margin-left: 4rem; }

            #left { flex-basis: 33.33333333%; }
            #right { flex-basis: 66.66666667%; }
            .col > * + * { margin-top: 2rem; }

        '''))

    with doc:
        h1(title)
        container = div(id='container')

        left = container.add(div(id='left', className='col'))
        left.add(h2('Full text'))
        text_container = left.add(div())
        for paragraph in text.split('\n'):
            # dominate doesn't do well with unicode characters so we change the
            # most frequent ones (quotes) into their ascii equivalents
            text_container.add(p(raw(normalize_quotes(paragraph))))

        right = container.add(div(id='right', className='col'))
        right.add(h2('Prediction based on Topic Evolution & stylometry markers'))
        right.add(p([
            'We are not sure whether this text was written by a human or generated by a machine.',
            'This text was likely generated by a machine.',
            'This text was likely written by a human.',
        ][author]))

        right.add(h2('Topic Evolution analysis'))
        right.add(img(
            src=f'data:image/png;base64,{te_img_data}',
            style='width: 100%'
        ))

        right.add(h2('Entity occurrence analysis'))
        right.add(entity_diagram[0])
        right.add(h3('Legend'))
        right.add(entity_diagram[1])

    return doc.render()

def analyze_samples(databases: list[tuple[str, list[dict[str, str]]]], sets: int, samples: int):
    directory = os.path.join('samples')
    os.makedirs(directory)

    def draw() -> tuple[str, dict[str, str]]:
        if len(databases) == 0:
            raise IndexError('Not enough samples in databases.')
        i = random.choice(range(len(databases)))
        name, db = databases[i]
        j = random.choice(range(len(db)))
        sample = db.pop(j)
        if len(db) == 0:
            databases.pop(i)
        return (name, sample)

    sources_fp = open(os.path.join(directory, '.sources.csv'), 'w')
    sources_writer = csv.writer(sources_fp)
    sources_writer.writerow(['set', 'sample', 'source'])

    total = sets * samples
    for set_id in range(1, sets + 1):
        directory_i = os.path.join(directory, str(set_id))
        os.makedirs(directory_i)
        sampled = 0
        while sampled < samples:
            text_id = sampled + 1
            progress = (set_id - 1) * samples + text_id
            print(f'\r\033[Kanalyzing sample {progress}/{total}', end='')
            source, sample = draw()
            text = sample['text']
            if not text: continue

            try:
                style_prediction = predict_author(text)
                te = get_default_te(text)
                te_prediction = predict_from_tem_metrics(te)
                author = get_prediction(style_prediction, te_prediction)
                entity_diagram = coref_diagram(coref_annotation(text))
            except: continue

            sources_writer.writerow([set_id, text_id, source])
            with open(os.path.join(directory_i, f'{text_id}.html'), 'w') as fp:
                fp.write(html_results(text, author, te, entity_diagram, title=f'unBlock Analysis for text {text_id}'))

            sampled += 1

    sources_fp.close()
    print('\ndone!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Analysis preprocessor')
    parser.add_argument('files', type=str, nargs='+', help='input yaml database files to uniformly draw samples from')
    parser.add_argument('--samples', '-m', type=int, required=True, help='number of samples per set')
    parser.add_argument('--sets', '-n', type=int, required=True, help='number of sample sets')
    args = parser.parse_args()

    databases: list[tuple[str, list[dict[str, str]]]] = list()
    for i, db in enumerate(args.files):
        name = db.split('/')[-1].split('.')[0]
        print(f'\r\033[Kreading yaml database "{name}" ({i + 1}/{len(args.files)})', end='')
        with open(db, 'r') as fp:
            data = yaml.safe_load(fp.read())
            databases.append((name, data))
    print('\ndone reading')

    analyze_samples(databases, args.sets, args.samples)
