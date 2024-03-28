import os
import sys

from definitions import ROOT_DIR, TEM_PARAMS

sys.path.append(os.path.join(ROOT_DIR, 'tem', 'script'))

import numpy as np
import numpy.typing as npt

from tem.script.model import TopicEvolution
from tem.script.nlp import get_structured_corpus
from tem.script.process import TEM
from tem.script.visualization import graph


def get_default_tecm(texts: list[str], tem_params: npt.NDArray | None) -> npt.NDArray[np.float64]:
    corpus = []
    for text in texts:
        try:
            corpus.append(get_structured_corpus(text))
        except ValueError:
            continue
    if tem_params is None:
        model = TEM.from_param_list(TEM_PARAMS, metrics=True)
    else:
        model = TEM.from_param_list(tem_params, metrics=True)
    return model.get_metrics(corpus)


def get_default_te_graph(text: str):
    corpus = get_structured_corpus(text)
    model = TEM.from_param_list(TEM_PARAMS)
    return graph(TopicEvolution(model.get_outputs([corpus])[0]))
