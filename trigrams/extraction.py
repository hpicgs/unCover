def trigram_distribution(trigrams, max_features: int = 10):
    items = list(trigrams.items())
    items.sort(key=lambda e: e[1], reverse=True)

    count = sum(c for _, c in items)
    return [(gram, n / count) for gram, n in items[:max_features]]
