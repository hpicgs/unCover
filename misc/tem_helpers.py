import os
import sys
import numpy as np
import numpy.typing as npt
from typing import Optional
from definitions import ROOT_DIR, TEM_PARAMS
from misc.logger import printProgressBar

sys.path.append(os.path.join(ROOT_DIR, 'tem', 'script'))
from tem.script.model import TopicEvolution
from tem.script.nlp import get_structured_corpus
from tem.script.process import TEM
from tem.script.visualization import graph


def _params(params: Optional[npt.NDArray] = None):
    return TEM_PARAMS if params is None else params


def get_tecm(texts: list[str], tem_params: Optional[npt.NDArray] = None, drop_invalids=True, preprocess=True) \
        -> npt.NDArray[np.float64]:
    if preprocess:
        corpus = list[str | None]()
        for i, text in enumerate(texts):
            if len(texts) > 1:
                printProgressBar(i, len(texts) - 1, fill='█')
            try:
                corpus.append(get_structured_corpus(text))
            except ValueError:
                corpus.append('')
    else:  # already preprocessed
        corpus = texts
    model = TEM.from_param_list(_params(tem_params), metrics=True)
    metrics = model.get_metrics(corpus)

    if drop_invalids:
        print("dropping invalids...")
        mask = np.all(np.isnan(metrics), axis=1)
        return metrics[~mask]
    return metrics


def get_te_graph(text: str, tem_params: Optional[npt.NDArray] = None):
    corpus = get_structured_corpus(text)
    model = TEM.from_param_list(_params(tem_params))
    return graph(TopicEvolution(model.get_outputs([corpus])[0]))


def preprocess_database(database):
    articles = database.get_all_articles()
    for i, article in enumerate(articles):
        printProgressBar(i, len(articles) - 1, fill='█')
        try:
            article['text'] = get_structured_corpus(article['text'])
        except ValueError:
            article['text'] = ''
        article['label'] = article['label'][0]
    database.replace_data(articles)
