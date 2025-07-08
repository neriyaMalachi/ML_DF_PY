import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = defaultdict(int)
        self.feature_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.class_counts = defaultdict(int)
        self.total_count = 0
        self.features = []

    def fit(self, data, target_column):
        self.features = [f for f in data[0] if f != target_column]
        self.total_count = len(data)

        for row in data:
            label = row[target_column]
            self.class_counts[label] += 1
            for feature in self.features:
                value = row[feature]
                self.feature_probs[feature][value][label] += 1

        for label in self.class_counts:
            self.class_probs[label] = self.class_counts[label] / self.total_count

    def predict(self, row):
        probs = {}
        for label in self.class_probs:
            log_prob = math.log(self.class_probs[label])
            for feature in self.features:
                value = row.get(feature, '')
                value_count = self.feature_probs[feature][value][label] + 1
                total_label_count = self.class_counts[label] + len(self.feature_probs[feature])
                log_prob += math.log(value_count / total_label_count)
            probs[label] = log_prob
        return max(probs, key=probs.get)

    def evaluate(self, test_data, target_column):
        correct = 0
        for row in test_data:
            true_label = row[target_column]
            predicted_label = self.predict(row)
            if true_label == predicted_label:
                correct += 1
        return correct / len(test_data)