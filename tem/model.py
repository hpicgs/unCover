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

        # create rank node for each period
        with g.subgraph() as s:
            s.attr('node', shape='box')
            s.attr('edge', style='invis')
            for n, _ in enumerate(self.periods):
                s.node(str(n), label=f'Period {n}')
                if n < len(self.periods) - 1:
                    s.edge(str(n), str(n + 1))

        return g
