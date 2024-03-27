from definitions import GOOGLE_API_KEY
import google.generativeai as genai


def generate_gemini_news_from(doc, german=False):
    print("Starting Gemini Request")
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    if not german:
        doc += "\n\nWrite a long news article about this topic."
    else:
        doc += "\n\nSchreibe einen langen Nachrichten Artikel über das Thema."
    response = model.generate_content(doc)
    try:
        return response.text
    except ValueError as e:
        print(f"Error while generating gemini article as none was generated: {e}")
        print(response.prompt_feedback)
        return None


