import os
import sys
import numpy as np
import numpy.typing as npt
from misc.definitions import ROOT_DIR, TEM_PARAMS
from misc.logger import printProgressBar

sys.path.append(os.path.join(ROOT_DIR, 'tem', 'script'))
from tem.script.model import TopicEvolution
from tem.script.nlp import get_structured_corpus
from tem.script.process import TEM
from tem.script.visualization import graph


def _params(params: npt.NDArray | None = None):
    return TEM_PARAMS if params is None else params


def get_tecm(texts: list[str], tem_params: npt.NDArray | None = None, drop_invalids=True, german=False) \
        -> npt.NDArray[np.float64]:
    corpus = list[str | None]()
    for i, text in enumerate(texts):
        printProgressBar(i, len(texts), fill='█')
        try:
            corpus.append(get_structured_corpus(text))
        except ValueError:
            corpus.append("")
    print("running tem...")
    model = TEM.from_param_list(_params(tem_params), metrics=True, german=german)
    metrics = model.get_metrics(corpus)

    if drop_invalids:
        print("dropping invalids...")
        mask = np.all(np.isnan(metrics), axis=1)
        return metrics[~mask]
    return metrics


def get_te_graph(text: str, tem_params: npt.NDArray | None = None):
    corpus = get_structured_corpus(text)
    model = TEM.from_param_list(_params(tem_params))
    return graph(TopicEvolution(model.get_outputs([corpus])[0]))
