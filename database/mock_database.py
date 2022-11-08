from definitions import DATABASE_AUTHORS_PATH
import yaml


class DatabaseAuthorship:
    @staticmethod
    def __write_data(data):
        with open(DATABASE_AUTHORS_PATH, "a") as mock_db:
            yaml.dump(data, mock_db, default_flow_style=False)
    

    @staticmethod
    def __get_data():
        try:
            with open(DATABASE_AUTHORS_PATH, "r") as mock_db:
                return yaml.safe_load(mock_db.read())
        except:
            return list()
    
    
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