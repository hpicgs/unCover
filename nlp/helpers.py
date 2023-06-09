import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def lower_alnum(doc: str) -> str:
    characters = [
        char.lower() for char in doc if char.isalnum() or char.isspace()
    ]
    return ''.join(characters)

# https://stackoverflow.com/a/62723088
def normalize_quotes(doc: str) -> str:
    # single quotes
    doc = re.sub(r'[\u02BB\u02BC\u066C\u2018-\u201A\u275B\u275C]', '\'', doc)
    # double quotes
    doc = re.sub(r'[\u201C-\u201E\u2033\u275D\u275E\u301D\u301E]', '"', doc)
    # apostrophes
    doc = re.sub(r'[\u0027\u02B9\u02BB\u02BC\u02BE\u02C8\u02EE\u0301\u0313\u0315\u055A\u05F3\u07F4\u07F5\u1FBF\u2018\u2019\u2032\uA78C\uFF07]', '\'', doc)
    return doc

# word-tokenizes document and removes all non-alphanumeric characters
# optionally clears stopwords and stems word
__stopwords = set(stopwords.words('english'))
__ps = PorterStemmer()
def normal_tokens(doc: str, clear_stopwords=True, stem=True) -> list[str]:
    return [
        __ps.stem(token) if stem else token
        for token in word_tokenize(lower_alnum(doc))
        if not clear_stopwords or token not in __stopwords
    ]
