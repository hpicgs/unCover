import argparse

import pandas_bokeh

from coherence.entities.coreferences import coref_annotation, coref_diagram

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('text_files', type=str, nargs='+')
    argparser.add_argument('-o', '--out', type=str, default='plot.html')
    args = argparser.parse_args()

    pandas_bokeh.output_file(args.out)

    plots = list()
    for tf in args.text_files:
        with open(tf, 'r') as fp:
            annotation = coref_annotation(fp.read())
        plots.append(coref_diagram(annotation))
    pandas_bokeh.plot_grid([plots], width=1250)
