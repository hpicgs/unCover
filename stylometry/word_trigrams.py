import spacy

en = spacy.load("en_core_web_sm")
de = spacy.load("de_core_news_sm")

def word_trigrams(text: str, german: bool) -> dict[str, int]:
    trigrams = dict[str, int]()
    nlp = de if german else en
    doc = nlp(text.lower())
    words = [token.text for token in doc if not token.is_stop]

    for i in range(len(words) - 2):
        trigram = ' '.join(words[i:i + 3])
        trigrams[trigram] = 1 if trigram not in trigrams else trigrams[trigram] + 1

    return trigrams
