from bs4 import BeautifulSoup

class PageProcessor:
    def __init__(self, html):
        self.__soup = BeautifulSoup(html, features="html.parser")

    def get_fulltext(self, separator=" "):
        return separator.join([element.text for element in self.__soup.find_all("p")])

    def get_title(self):
        return " ".join([element.text for element in self.__soup.find_all("h1")])
    
    def get_author(self):
        author_elements = [meta_element for meta_element in self.__soup.find_all("meta") if any(["article:author" in meta_element.get(attribute) for attribute in meta_element.attrs])]
        if len(author_elements) == 1:
            return author_elements[0].get("content")
        else:
            return ""

    def get_all_paragraphs(self):
        return [element.text for element in self.__soup.find_all("p")]

