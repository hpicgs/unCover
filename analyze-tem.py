import argparse
import statistics

from tem.model import TopicEvolution
from tem.nlp import docs_from_period, merge_short_periods
from tem.process import get_topic_evolution

def te_analysis_data(te: TopicEvolution) -> dict[str, float]:
    node_count_by_id: dict[int, int] = {}
    for period in te.periods:
        for topic in period.topics:
            node_count_by_id[topic.id] = 1 if topic.id not in node_count_by_id else node_count_by_id[topic.id] + 1
    node_count = sum((count for count in node_count_by_id.values()))

    return {
        'n_ids/n_nodes': len(node_count_by_id) / node_count,
        'largest group / n_nodes': max((count for count in node_count_by_id.values())) / node_count,
        'mean n_words per topic': statistics.mean([len(words) for period in te.periods for topic in period.topics for words in topic.words])
    }
