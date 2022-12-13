import argparse

from coherence.entities.coreferences import coreference
from database.mock_database import DatabaseAuthorship

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('text_files', type=str, nargs='*')
    args = argparser.parse_args()

    coreference(DatabaseAuthorship.get_articles_by_author('https://www.theguardian.com/profile/martin-chulov')[0]['text'])
