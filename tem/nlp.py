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

# merge period into predecessor if number of docs < min_docs
def merge_short_periods(corpus: list[list[list[str]]], min_docs = 2) -> list[list[list[str]]]:
    if len(corpus) == 0: return []
    merged = [corpus[0]]
    for period in corpus[1:]:
        if len(merged[-1]) >= min_docs:
            merged.append(period)
        else:
            merged[-1] += period
    return merged
