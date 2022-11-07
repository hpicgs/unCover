def char_trigrams(text: str) -> dict[str, int]:
    normal = ''.join([
        char.lower()
    for char in text if char.isalnum() or char.isspace()])

    ret = dict[str, int]()
    for i in range(len(normal) - 2):
        trigram = normal[i:i + 3]
        ret[trigram] = 1 if trigram not in ret else ret[trigram] + 1

    return ret
