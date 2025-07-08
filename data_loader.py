import csv

class DataLoader:
    @staticmethod
    def load_csv(file_path):
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            return list(reader)
