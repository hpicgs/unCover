import subprocess
import yaml

from tem.model import TopicEvolution

def get_topic_evolution(text: str) -> TopicEvolution:
    p = subprocess.Popen(
        'tem/topic-evolution-model/.build.out/out',
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    out = p.communicate(input=text.encode())[0].decode()
    return TopicEvolution(yaml.safe_load(out))
