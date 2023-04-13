import openai
from transformers import pipeline, set_seed
from definitions import GPT_KEY, OPENAI_ORGA


def generate_gpt3_news_from(doc, size=1000):
    print("Starting GPT Request")
    openai.organization = OPENAI_ORGA
    openai.api_key = GPT_KEY
    doc += "\n\nWrite a long news article about this topic."
    response = None
    while response is None:
        try:
            response = openai.Completion.create(model="text-davinci-003", prompt=doc, max_tokens=size, temperature=0.4)
        except openai.error.APIConnectionError or openai.error.Timeout:
            continue
        except openai.error.InvalidRequestError:
            break

    print("GPT finished")
    return response.choices[0].text if response else None


def generate_gpt2_news_from(doc, size=1000):
    print("Starting GPT2 Generation")
    generator = pipeline('text-generation', model='gpt2-large')
    set_seed(42)
    # doc += "Russia and Ukraine: Who will win this war? It is unlikely a 'third force,' will prevail: A former Soviet politician. In general War is"
    doc += ". I would write a news article about the topic like this:"
    result = generator(doc, max_length=size, num_return_sequences=1)[0]["generated_text"]
    print("GPT2 Finished")
    print(result)
    # return result.rsplit(".", 1)[0] + "."
    return (result.rsplit(".", 1)[0] + ".").split("I would write a news article about the topic like this:", 1)[1]

