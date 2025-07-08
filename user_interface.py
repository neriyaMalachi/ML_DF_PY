class UserInterface:
    def show_welcome(self):
        print("\nברוך הבא למסווג Naive Bayes!")

    def ask_csv_path(self, message):
        return input(message)

    def show_accuracy(self, accuracy):
        print(f"\nאחוז הדיוק של המודל הוא: {accuracy * 100:.2f}%")

    def ask_for_single_prediction(self, features):
        print("\nהזן ערכים לרשומה חדשה:")
        row = {}
        for f in features:
            row[f] = input(f"{f}: ")
        return row

    def show_prediction(self, label):
        print(f"\nהסיווג של הרשומה הוא: {label}")