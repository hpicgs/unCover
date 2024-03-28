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

def _params(params: npt.NDArray | None = None):
    return TEM_PARAMS if params is None else params

def get_tecm(texts: list[str], tem_params: npt.NDArray | None = None, drop_invalids = True) -> npt.NDArray[np.float64]:
    corpus = list[str | None]()
    for text in texts:
        try:
            corpus.append(get_structured_corpus(text))
        except ValueError:
            corpus.append(None)

    model = TEM.from_param_list(_params(tem_params), metrics=True)
    metrics = model.get_metrics(corpus)

    if drop_invalids:
        mask = np.all(np.isnan(metrics), axis=1)
        return metrics[~mask]
    return metrics


def get_te_graph(text: str, tem_params: npt.NDArray | None = None):
    corpus = get_structured_corpus(text)
    model = TEM.from_param_list(_params(tem_params))
    return graph(TopicEvolution(model.get_outputs([corpus])[0]))
