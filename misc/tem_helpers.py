import numpy as np
import numpy.typing as npt

from tem.script.nlp import get_structured_corpus
from tem.script.process import TEM


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
