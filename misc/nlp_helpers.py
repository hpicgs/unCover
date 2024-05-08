import re
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from definitions import NLTK_DATA


def handle_nltk_download(e: LookupError):
    message = e.args[0]
    resource_match = re.search(r"nltk\.download\('([^']+)'\)", message)
    if not resource_match: raise e
    resource = resource_match.group(1)
    download(resource, download_dir=NLTK_DATA)


def lower_alnum(doc: str) -> str:
    characters = [
        char.lower() for char in doc if char.isalnum() or char.isspace()
    ]
    return ''.join(characters)


def preprocess_article(doc):
    paragraph = re.sub("[ \t\r\f]+", ' ', doc)  # \s without \n
    return paragraph


# https://stackoverflow.com/a/62723088
def normalize_quotes(doc: str) -> str:
    # single quotes
    doc = re.sub(r'[\u02BB\u02BC\u066C\u2018-\u201A\u275B\u275C]', '\'', doc)
    # double quotes
    doc = re.sub(r'[\u201C-\u201E\u2033\u275D\u275E\u301D\u301E]', '"', doc)
    # apostrophes
    doc = re.sub(
        r'[\u0027\u02B9\u02BB\u02BC\u02BE\u02C8\u02EE\u0301\u0313\u0315\u055A\u05F3\u07F4\u07F5\u1FBF\u2018\u2019\u2032\uA78C\uFF07]',
        '\'', doc)
    return doc


# word-tokenizes document and removes all non-alphanumeric characters
# optionally clears stopwords and stems word
try:  # check if nltk is installed and download if it is not
    __stopwords = set(stopwords.words('english'))
except LookupError as e:
    handle_nltk_download(e)
    __stopwords = set(stopwords.words('english'))
finally:
    __ps = PorterStemmer()


def normal_tokens(doc: str, clear_stopwords=True, stem=True) -> list[str]:
    return [
        __ps.stem(token) if stem else token
        for token in word_tokenize(lower_alnum(doc))
        if not clear_stopwords or token not in __stopwords
    ]
