from definitions import DATABASE_AUTHORS_PATH, DATABASE_FILES_PATH, DATABASE_GEN_PATH, DATABASE_TEST_PATH
import yaml, os


class MockDatabase:
    def __init__(self, path):
        self.path = path

    def write_data(self, data):
        os.makedirs(DATABASE_FILES_PATH, exist_ok=True)
        with open(self.path, "w") as mock_db:
            yaml.dump(data, mock_db, default_flow_style=False)

    def get_data(self):
        try:
            with open(self.path, "r") as mock_db:
                return yaml.safe_load(mock_db.read())
        except:
            return list()


class DatabaseAuthorship:
    __db = MockDatabase(DATABASE_AUTHORS_PATH)

    @staticmethod
    def get_authors():
        articles = DatabaseAuthorship.__db.get_data()
        authors = list()
        for article in articles:
            for author in article["author"].split(","):
                if author not in authors and article["text"] is not None:
                    authors.append(author)
        return authors

    @staticmethod
    def get_articles_by_author(author):
        articles = DatabaseAuthorship.__db.get_data()
        return [
            {
                "author" : article["author"].split(","),
                "source" : article["source"],
                "text" : article["text"]
            }
            for article in articles if author in article["author"].split(",") and article["text"] is not None
            ]

    @staticmethod
    def insert_article(text, source, author):
        data = DatabaseAuthorship.__db.get_data()
        if data == [] or data is None:
            data = []
            data.append({"source": source, "text": text, "author": author})
            DatabaseAuthorship.__db.write_data(data)
        elif not any([source == article["source"] for article in data]):
            data.append({"source": source, "text": text, "author": author})
            DatabaseAuthorship.__db.write_data(data)


class DatabaseGenArticles:
    __db = MockDatabase(DATABASE_GEN_PATH)

    @staticmethod
    def get_methods():
        articles = DatabaseGenArticles.__db.get_data()
        return {
            method for article in articles
            for method in article['method'].split(',') if article['text'] is not None}

    @staticmethod
    def get_articles_by_method(method):
        articles = DatabaseGenArticles.__db.get_data()
        return [article for article in articles if
                method in article["method"].split(",") and article["text"] is not None]

    @staticmethod
    def insert_article(text, source, method):
        data = DatabaseGenArticles.__db.get_data()
        if data == [] or data is None:
            data = []
        data.append({"source": source, "text": text, "method": method})
        DatabaseGenArticles.__db.write_data(data)


class TestDatabase:
    __db = MockDatabase(DATABASE_TEST_PATH)

    @staticmethod
    def get_all_articles_sorted_by_methods():
        articles = TestDatabase.__db.get_data()
        return {label: [article['text'] for article in articles if
                        label in article["label"].split(",") and article["text"] is not None]
                for label in [l for a in articles for l in a['label'].split(',')]
                }

    @staticmethod
    def insert_article(text, source, label):
        data = TestDatabase.__db.get_data()
        if data == [] or data is None:
            data = []
        data.append({"source": source, "text": text, "label": label})
        TestDatabase.__db.write_data(data)
