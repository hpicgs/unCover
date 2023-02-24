import argparse

import dominate
# pyright: reportWildcardImportFromLibrary=false
from dominate.tags import *

from coherence.entities.coreferences import coref_annotation, coref_diagram

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('text', type=str)
    argparser.add_argument('-o', '--out', type=str, default='plot.html')
    args = argparser.parse_args()

    with open(args.text, 'r') as fp:
        annotation = coref_annotation(fp.read())
    chart, legend = coref_diagram(annotation)

    title = f'Entity occurrances for {args.text}'
    doc = dominate.document(title=title)
    with doc:
        container = div(style='max-width: 900px; margin: auto')
        with container:
            h1(title)
            h2('Text')
        container.add(chart)
        container.add(h2('Legend'))
        container.add(legend)

    with open(args.out, 'w') as fp:
        fp.write(doc.render())
