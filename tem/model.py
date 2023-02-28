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
