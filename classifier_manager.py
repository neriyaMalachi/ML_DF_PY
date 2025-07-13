from data_loader import DataLoader
from naive_bayes import NaiveBayesClassifier
from user_interface import UserInterface

class ClassifierManager:
    def __init__(self):
        self.ui = UserInterface()
        self.classifier = NaiveBayesClassifier()

    def run(self):
        self.ui.show_welcome()
        train_path = self.ui.ask_csv_path("\nהזן את הנתיב לקובץ האימון: ")

        data = DataLoader.load_csv(train_path)
        target_column = input("\nמה שם עמודת הסיווג (Label)? ")

        split_index = int(len(data) * 0.7)
        self.train_data = data[:split_index]
        self.test_data = data[split_index:]

        self.classifier.fit(self.train_data, target_column)
        print("\nהמודל נבנה בהצלחה!")

        acc = self.classifier.evaluate(test_data, target_column)
        self.ui.show_accuracy(acc)

        while True:
            choice = input("\nהאם תרצה לבצע סיווג לרשומה בודדת? (y/n): ")
            if choice.lower() != 'y':
                break
            row = self.ui.ask_for_single_prediction(self.classifier.features)
            label = self.classifier.predict(row)
            self.ui.show_prediction(label)
