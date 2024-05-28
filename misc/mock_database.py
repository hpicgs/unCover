from definitions import DATABASE_FILES_PATH
import yaml
import os


class MockDatabase:
    def __init__(self, path):
        self.path = path

    def write_data(self, data):
        os.makedirs(DATABASE_FILES_PATH, exist_ok=True)
        with open(self.path, 'w') as mock_db:
            yaml.dump(data, mock_db, default_flow_style=False)

    def get_data(self):
        try:
            with open(self.path, 'r') as mock_db:
                return yaml.safe_load(mock_db.read())
        except FileNotFoundError:
            return list()


class Database:
    def __init__(self, path):
        self.__db = MockDatabase(path)

    def get_labels(self):
        articles = self.__db.get_data()
        return {label for article in articles
                for label in article['label'].split(',') if article['text'] is not None}

    def get_all_articles(self):
        articles = self.__db.get_data()
        return [article for article in articles if article['text'] is not None]

    def get_articles_by_label(self, label):
        articles = self.__db.get_data()
        return [article for article in articles if
                label in article['label'].split(',') and article['text'] is not None]

    def get_all_articles_sorted_by_labels(self):
        articles = self.__db.get_data()
        return {label: [article['text'] for article in articles if
                        label in article['label'].split(',') and article['text'] is not None]
                for label in [l for a in articles for l in a['label'].split(',')]}

    def get_all_sources(self):
        articles = self.__db.get_data()
        return {article['source'] for article in articles if article['source'] is not None}

    def insert_article(self, text, source, label):
        data = self.__db.get_data()
        if data == [] or data is None:
            data = [{'source': source, 'text': text, 'label': label}]
            self.__db.write_data(data)
        elif not any([source == article['source'] and label == article['label'] for article in data]):
            data.append({'source': source, 'text': text, 'label': label})
            self.__db.write_data(data)

    def replace_data(self, data):
        self.__db.write_data(data)
