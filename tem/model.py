import graphviz

class TopicEvolution:
    class Topic:
        def __init__(self, data):
            self.words: list[str] = [str(word) for word in data['topic']]
            self.id: int = data['id']
            self.health: float = data['health']

        def label(self, max_width=32):
            if len(self.words) == 0: return ''
            lines = [self.words[0]]
            for word in self.words[1:]:
                if len(lines[-1]) < max_width:
                    lines[-1] += ' ' + word
                else:
                    lines.append(word)
            return '\n'.join(lines)

    class Period:
        def __init__(self, data):
            if not data:
                self.topics = []
                return
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
        # topic id -> ids of (graph) nodes in previous period
        previous_topics: dict[int, list[str]] = {}
        # helper for dictionary
        def append_or_set(d: dict[int, list[str]], k: int, v: str):
            if k in d: d[k].append(v)
            else: d[k] = [v]
        # create nodes & edges
        for n, period in enumerate(self.periods):
            current_topics: dict[int, list[str]] = {}
            with g.subgraph() as s:
                s.attr(rank='same')
                s.node(str(n))
                for m, topic in enumerate(period.topics):
                    node_id = f'{n}-{m}'
                    append_or_set(current_topics, topic.id, node_id)
                    s.node(node_id, label=topic.label())
                    if topic.id in previous_topics:
                        for predecessor in previous_topics[topic.id]:
                            g.edge(predecessor, node_id)
            previous_topics = current_topics

        return g
