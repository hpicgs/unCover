import os
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

MODELS_DIR = os.path.join(ROOT_DIR, 'models')
STYLOMETRY_DIR = os.path.join(MODELS_DIR, 'stylometry')
# also adjust existing stanford dir to use new variable
STANFORD_DIR = os.path.join(MODELS_DIR, 'stanford-corenlp-4.5.1')
STANFORD_JARS = (
    os.path.join(STANFORD_DIR, 'stanford-corenlp-4.5.1.jar'),
    os.path.join(STANFORD_DIR, 'stanford-corenlp-4.5.1-models.jar'),
)

DATABASE_FILES_PATH = os.path.join(ROOT_DIR, 'database', 'files')
DATABASE_AUTHORS_PATH = os.path.join(DATABASE_FILES_PATH, '.mock-authors.yaml')
DATABASE_MACHINES_PATH = os.path.join(DATABASE_FILES_PATH, '.mock-machines.yaml')

QUERIES = os.getenv("QUERIES", "").split(", ")
