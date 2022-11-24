def char_trigrams(text: str) -> dict[str, int]:
    trigrams = dict[str, int]()
    normal = ''.join([
        char.lower()
    for char in text if char.isalnum() or char.isspace()])

    for i in range(len(normal) - 2):
        trigram = normal[i:i + 3]
        trigrams[trigram] = 1 if trigram not in trigrams else trigrams[trigram] + 1
        
    return trigrams
