import os

from dotenv import load_dotenv
import numpy as np

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ENV_DIR = os.getenv("CONDA_PREFIX", os.getenv("HOME", '/'))

TEM_PARAMS = np.array([
    float(os.getenv('TEM_PARAM_C', '')),
    float(os.getenv('TEM_PARAM_ALPHA', '')),
    float(os.getenv('TEM_PARAM_BETA', '')),
    float(os.getenv('TEM_PARAM_GAMMA', '')),
    float(os.getenv('TEM_PARAM_DELTA', '')),
    float(os.getenv('TEM_PARAM_THETA', '')),
    float(os.getenv('TEM_PARAM_MERGE', '')),
    float(os.getenv('TEM_PARAM_EVOLV', ''))
])

MODELS_DIR = os.path.join(ENV_DIR, 'models')
STYLOMETRY_DIR = os.path.join(MODELS_DIR, 'stylometry')
# also adjust existing stanford dir to use new variable
STANFORD_DIR = os.path.join(MODELS_DIR, 'stanford-corenlp-4.5.1')
STANFORD_JARS = (
    os.path.join(STANFORD_DIR, 'stanford-corenlp-4.5.1.jar'),
    os.path.join(STANFORD_DIR, 'stanford-corenlp-4.5.1-models.jar'),
)
NLTK_DATA = os.path.join(ENV_DIR, 'nltk_data')

DATABASE_FILES_PATH = os.path.join(ROOT_DIR, '.database')
DATABASE_AUTHORS_PATH = os.path.join(DATABASE_FILES_PATH, 'authors.yaml')
DATABASE_GEN_PATH = os.path.join(DATABASE_FILES_PATH, 'gen.yaml')
DATABASE_TEST_PATH = os.path.join(DATABASE_FILES_PATH, 'test.yaml')

GPT_KEY = os.getenv("GPT_KEY", "")
OPENAI_ORGA = os.getenv("OPENAI_ORGA", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
WINSTON_API_KEY = os.getenv("WINSTON_API_KEY", "")

CHAR_MACHINE_CONFIDENCE = float(os.getenv("CHAR_MACHINE_CONFIDENCE", ""))
CHAR_HUMAN_CONFIDENCE = float(os.getenv("CHAR_HUMAN_CONFIDENCE", ""))
SEM_MACHINE_CONFIDENCE = float(os.getenv("SEM_MACHINE_CONFIDENCE", ""))
SEM_HUMAN_CONFIDENCE = float(os.getenv("SEM_HUMAN_CONFIDENCE", ""))
STYLE_MACHINE_CONFIDENCE = float(os.getenv("STYLE_MACHINE_CONFIDENCE", ""))
STYLE_HUMAN_CONFIDENCE = float(os.getenv("STYLE_HUMAN_CONFIDENCE", ""))
