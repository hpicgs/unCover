from definitions import DATABASE_AUTHORS_PATH, DATABASE_MACHINES_PATH, DATABASE_FILES_PATH
import yaml, os

class DatabaseMachines:
    @staticmethod
    def __write_data(data):
        os.makedirs(DATABASE_FILES_PATH, exist_ok=True)
        with open(DATABASE_MACHINES_PATH, "w") as mock_db:
            yaml.dump(data, mock_db, default_flow_style=False)

    @staticmethod
    def __get_data():
        try:
            with open(DATABASE_MACHINES_PATH, "r") as mock_db:
                return yaml.safe_load(mock_db.read())
        except:
            return list()

    @staticmethod
    def get_authors():
        articles = DatabaseMachines.__get_data()
        authors = list()
        for article in articles:
            for author in article["author"].split(","):
                if author not in authors and article["text"] is not None:
                    authors.append(author)
        return authors

    @staticmethod
    def get_articles_by_author(author):
        articles = DatabaseMachines.__get_data()
        return [article for article in articles if author in article["author"].split(",") and article["text"] is not None]


    @staticmethod
    def insert_article(text, source, author):
        data = DatabaseMachines.__get_data()
        if data == [] or data is None:
            data = []
            data.append({"source":source, "text":text, "author":author})
            DatabaseMachines.__write_data(data)
        elif not any([source == article["source"] for article in data]):
            data.append({"source":source, "text":text, "author":author})
            DatabaseMachines.__write_data(data)


class DatabaseAuthorship:
    @staticmethod
    def __write_data(data):
        os.makedirs(DATABASE_FILES_PATH, exist_ok=True)
        with open(DATABASE_AUTHORS_PATH, "w") as mock_db:
            yaml.dump(data, mock_db, default_flow_style="|")
    

    @staticmethod
    def __get_data():
        try:
            with open(DATABASE_AUTHORS_PATH, "r") as mock_db:
                return yaml.safe_load(mock_db.read())
        except:
            return list()
    

    @staticmethod
    def get_authors():
        articles = DatabaseAuthorship.__get_data()
        authors = list()
        for article in articles:
            for author in article["author"].split(","):
                if author not in authors and article["text"] is not None:
                    authors.append(author)
        return authors

    @staticmethod
    def get_articles_by_author(author):
        articles = DatabaseAuthorship.__get_data()
        return [
            {
                "author" : article["author"].split(","),
                "source" : article["source"],
                "text" : article["text"].replace("#customdelimiter#", "\n")
            }
            for article in articles if author in article["author"].split(",") and article["text"] is not None
            ]


    @staticmethod
    def insert_article(text, source, author):
        data = DatabaseAuthorship.__get_data()
        if data == [] or data is None:
            data = []
            data.append({"source":source, "text":text, "author":author})
            DatabaseAuthorship.__write_data(data)
        elif not any([source == article["source"] for article in data]):
            data.append({"source":source, "text":text, "author":author})
            DatabaseAuthorship.__write_data(data)
