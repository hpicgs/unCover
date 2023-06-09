import subprocess
import yaml

from tem.model import TopicEvolution
from tem.nlp import docs_from_period, merge_short_periods

def get_topic_evolution(
    corpus: list[list[list[str]]], # periods, docs, words
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    theta: int,
    mergeThreshold: float,
    evolutionThreshold: float,
) -> TopicEvolution:
    structured_text = '\n\n'.join([
        '\n'.join([
            ' '.join(word for word in doc)
        for doc in period])
    for period in corpus])

    p = subprocess.Popen([
            'tem/topic-evolution-model/.build.out/out',
            '--c', str(c),
            '--alpha', str(alpha),
            '--beta', str(beta),
            '--gamma', str(gamma),
            '--delta', str(delta),
            '--theta', str(theta),
            '--mergeThreshold', str(mergeThreshold),
            '--evolutionThreshold',str(evolutionThreshold) 
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out = p.communicate(input=structured_text.encode())[0].decode()

    return TopicEvolution(yaml.safe_load(out))

def get_default_te(text: str) -> TopicEvolution:
    corpus = [docs_from_period(line) for line in text.split('\n') if len(line) > 0]
    corpus = merge_short_periods(corpus, min_docs=2)
    return get_topic_evolution(
        corpus,
        c=0.5,
        alpha=0,
        beta=-1,
        gamma=0,
        delta=1,
        theta=2,
        mergeThreshold=100,
        evolutionThreshold=100
    )
