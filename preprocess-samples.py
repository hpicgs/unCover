import argparse
import os
import random

import yaml

from main import entity_occurrence_diagram, get_prediction
from stylometry.logistic_regression import predict_author
from tem.process import get_default_te
from train_tem_metrics import predict_from_tem_metrics

def analyze_samples(db: list[dict[str, str]], db_name: str, samples: int):
    sampled = 0
    while sampled < samples and len(db) > 0:
        print(f'\r\033[Kanalyzing random sample {sampled + 1} of {samples}', end='')

        i = random.choice(range(len(db)))
        text = db.pop(i)['text']
        if not text: continue

        try:
            style_prediction = predict_author(text)
            te = get_default_te(text)
            te_prediction = predict_from_tem_metrics(te)
            entity_html = entity_occurrence_diagram(text)
            author = get_prediction(style_prediction, te_prediction)
        except: continue

        directory = os.path.join('samples', db_name, str(i))
        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, 'topic_evolution.png'), 'wb') as fp:
            fp.write(te.graph().pipe(format='png'))
        with open(os.path.join(directory, 'stilometry.html'), 'w') as fp:
            fp.write(entity_html)
        with open(os.path.join(directory, 'text.txt'), 'w') as fp:
            fp.write(text)
        with open(os.path.join(directory, 'prediction.txt'), 'w') as fp:
            fp.write(['Not sure', 'Human', 'Machine'][author])

        sampled += 1

    print('\ndone!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Analysis preprocessor')
    parser.add_argument('file', type=str, help='input yaml database file')
    parser.add_argument('--samples', '-n', type=int, required=True, help='number of samples to draw from the database')
    args = parser.parse_args()

    if args.file.endswith('.yaml'):
        print('reading yaml database')
        with open(args.file, 'r') as fp:
            data = yaml.safe_load(fp.read())
            analyze_samples(data, args.file.split('/')[-1].split('.')[0], args.samples)
    else:
        parser.error('Unknown file type')
