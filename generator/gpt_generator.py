import openai
from transformers import pipeline, set_seed
from definitions import GPT_KEY, OPENAI_ORGA


def generate_gpt3_news_from(doc, size=1000):
    print("Starting GPT3 Request")
    client = openai.OpenAI(
        api_key=GPT_KEY,
        organization=OPENAI_ORGA
    )
    doc += "\n\nWrite a long news article about this topic."
    response = None
    while response is None:
        try:
            response = client.completions.create(model="gpt-3.5-turbo-instruct", prompt=doc, max_tokens=size,
                                                 temperature=0.4)
        except openai.APIConnectionError or openai.Timeout:
            continue
        except openai.BadRequestError:
            break

    print("GPT3 finished")
    return response.choices[0].text if response else None


def generate_gpt4_news_from(doc, size=1000):
    print("Starting GPT 4 Request")
    client = openai.OpenAI(
        api_key=GPT_KEY,
        organization=OPENAI_ORGA
    )
    doc += "\n\nWrite a long news article about this topic."
    response = None
    while response is None:
        try:
            response = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": doc}],
                                                      max_tokens=size, temperature=0.4)
        except openai.APIConnectionError or openai.Timeout:
            continue
        except openai.RateLimitError:
            break

    print("GPT4 finished")
    return response.choices[0].message.content if response else None


def generate_gpt2_news_from(doc, size=1000):
    print("Starting GPT2 Generation")
    doc_prompt = "I would write a news article about the topic like this:"
    generator = pipeline('text-generation', model='gpt2-large')
    set_seed(42)
    doc += ". " + doc_prompt
    result = generator(doc, max_length=size, num_return_sequences=1)[0]["generated_text"]
    print("GPT2 Finished")
    return (result.rsplit(".", 1)[0] + ".").split(doc_prompt, 1)[1]
