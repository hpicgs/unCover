import re

from nltk.tokenize import sent_tokenize

from nlp.helpers import normal_tokens

# dash needs to be escaped
# other chars are en-/em-dash
__doc_separators = ',:;\\-–—'
def docs_from_period(period: str) -> list[list[str]]:
    with_seps = re.sub(f'[{__doc_separators}]', '.', period)
    return [
        normal_tokens(doc)
        for doc in sent_tokenize(with_seps)
    ]
