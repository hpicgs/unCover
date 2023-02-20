import openai
from definitions import GPT_KEY, OPENAI_ORGA


def generate_gpt3_news_from_original(doc):
    openai.organization = OPENAI_ORGA
    openai.api_key = GPT_KEY
    doc += "\n\nWrite a news article about this topic."
    #long docs can throw error but try catch will not work because of long response time, need to find other solution
    response = openai.Completion.create(model="text-davinci-003", prompt=doc, max_tokens=800, temperature=0.4)
    return response.choices[0].text
