import spacy
import re

en = spacy.load("en_core_web_md")
de = spacy.load("de_core_news_md")


def word_trigrams(text: str, german: bool) -> dict[str, int]:
    trigrams = dict[str, int]()
    nlp = de if german else en
    doc = nlp(text.lower())
    sentences = list(doc.sents)

    for sentence in sentences:
        # Remove special characters from each sentence
        clean_sentence = re.sub(r'\W', ' ', sentence.text).replace('_', ' ')
        words = [token.text for token in nlp(clean_sentence) if not token.is_stop]
        for i in range(len(words) - 2):
            trigram = ' '.join(words[i:i + 3])
            trigrams[trigram] = 1 if trigram not in trigrams else trigrams[trigram] + 1

    return trigrams
