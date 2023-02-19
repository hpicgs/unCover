import openai
from definitions import GPT_KEY, OPENAI_ORGA
def generate_gpt3_news_from_original(doc):
    openai.organization = OPENAI_ORGA
    openai.api_key = GPT_KEY
    doc += "\n\nWrite a news article about this topic."
    response = openai.Completion.create(model="text-davinci-003", prompt=doc, max_tokens=500, temperature=0.4)
    return response.choices[0].text
