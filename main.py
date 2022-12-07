import argparse

from coherence.entities.coreferences import coreference

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('text_files', type=str, nargs='*')
    args = argparser.parse_args()

    coreference()
