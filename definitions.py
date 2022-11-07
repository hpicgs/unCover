import os
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


DATABASE_AUTHORS_PATH = os.path.join(ROOT_DIR, 'database', '.mock-authors.yaml')

QUERIES = os.getenv("QUERIES", "").split(", ")