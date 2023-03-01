import subprocess
import yaml

from tem.model import TopicEvolution

def get_topic_evolution(
    text: str,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    theta: int,
    mergeThreshold: float,
    evolutionThreshold: float,
) -> TopicEvolution:
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
    out = p.communicate(input=text.encode())[0].decode()
    return TopicEvolution(yaml.safe_load(out))
