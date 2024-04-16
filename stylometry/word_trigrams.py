def word_trigrams(text: str) -> dict[str, int]:
    trigrams = dict[str, int]()
    words = text.lower().split()

    for i in range(len(words) - 2):
        trigram = ' '.join(words[i:i + 3])
        trigrams[trigram] = 1 if trigram not in trigrams else trigrams[trigram] + 1

    return trigrams
