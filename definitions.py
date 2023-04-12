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
CHAR_MACHINE_CONFIDENCE = float(os.getenv("CHAR_MACHINE_CONFIDENCE", ""))   #tried & tested value: 0.253
CHAR_HUMAN_CONFIDENCE = float(os.getenv("CHAR_HUMAN_CONFIDENCE", ""))       #tried & tested value: 0.1304
SEM_MACHINE_CONFIDENCE = float(os.getenv("SEM_MACHINE_CONFIDENCE", ""))     #0.251
SEM_HUMAN_CONFIDENCE = float(os.getenv("SEM_HUMAN_CONFIDENCE", ""))         #0.13