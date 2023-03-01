import graphviz

class TopicEvolution:
    class Topic:
        def __init__(self, data):
            self.words: list[str] = data['topic']
            self.id: int = data['id']
            self.health: float = data['health']

    class Period:
        def __init__(self, data):
            self.topics = [
                TopicEvolution.Topic(topic)
            for topic in data]

    def __init__(self, data):
        self.periods = [
            TopicEvolution.Period(period)
        for period in data]

    def graph(self) -> graphviz.Digraph:
        g = graphviz.Digraph()
        g.attr(rankdir='TB')

        # RANK NODE FOR EACH PERIOD (i.e. "Period n" labels)
        with g.subgraph() as s:
            s.attr('node', shape='box')
            s.attr('edge', style='invis')
            for n, _ in enumerate(self.periods):
                s.node(str(n), label=f'Period {n}')
                if n < len(self.periods) - 1:
                    s.edge(str(n), str(n + 1))

        # TOPIC NODES FOR EACH PERIOD
        # topic id -> id of node in previous period
        previous_topics: dict[int, str] = {}
        for n, period in enumerate(self.periods):
            current_topics: dict[int, str] = {}
            with g.subgraph() as s:
                s.attr(rank='same')
                s.node(str(n))
                for m, topic in enumerate(period.topics):
                    node_id = f'{n}-{m}'
                    current_topics[topic.id] = node_id
                    s.node(node_id, label=', '.join(topic.words))
                    if topic.id in previous_topics:
                        g.edge(previous_topics[topic.id], node_id)
            previous_topics = current_topics

        return g
