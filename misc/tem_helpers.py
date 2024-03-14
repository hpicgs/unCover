import os
import sys

from definitions import ROOT_DIR

sys.path.append(os.path.join(ROOT_DIR, 'tem', 'script'))

import numpy as np
import numpy.typing as npt

from tem.script.model import TopicEvolution
from tem.script.nlp import get_structured_corpus
from tem.script.process import TEM
from tem.script.visualization import graph


tem_default_params = np.array([
    0.5, # c
    0, # alpha
    0, # beta
    0, # gamma
    1, # delta
    0.35, # theta
    0.35, # merge_threshold
    0.6 # evolution_threshold
])

def get_default_tecm(text: str) -> npt.NDArray[np.float64]:
    corpus = get_structured_corpus(text)
    model = TEM.from_param_list(tem_default_params, metrics=True)
    return model.get_metrics([corpus])[0]

def get_default_te_graph(text: str):
    corpus = get_structured_corpus(text)
    model = TEM.from_param_list(tem_default_params)
    return graph(TopicEvolution(model.get_outputs([corpus])[0]))
