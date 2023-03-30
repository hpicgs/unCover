import openai
from definitions import GPT_KEY, OPENAI_ORGA


def generate_gpt3_news_from_original(doc):
    print("Starting GPT Request")
    openai.organization = OPENAI_ORGA
    openai.api_key = GPT_KEY
    doc += "\n\nWrite a news article about this topic."
    response = None
    while response is None:
        try:
            response = openai.Completion.create(model="text-davinci-003", prompt=doc, max_tokens=800, temperature=0.4)
        except openai.error.APIConnectionError or openai.error.Timeout:
            continue
        except openai.error.InvalidRequestError:
            break

    print("GPT finished")
    return response.choices[0].text if response else None
