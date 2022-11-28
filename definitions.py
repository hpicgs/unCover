import os
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

STANFORD_DIR = os.path.join(ROOT_DIR, 'models', 'stanford-corenlp-4.5.1')
STANFORD_JARS = (
    os.path.join(STANFORD_DIR, 'stanford-corenlp-4.5.1.jar'),
    os.path.join(STANFORD_DIR, 'stanford-corenlp-4.5.1-models.jar'),
)

DATABASE_AUTHORS_PATH = os.path.join(ROOT_DIR, 'database/files', '.mock-authors.yaml')

QUERIES = os.getenv("QUERIES", "").split(", ")
