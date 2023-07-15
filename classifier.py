import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns

class EmailClassifier:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.y_pred = None
        self.console = Console()

    def load_dataset(self, dataset_path):
        self.df = pd.read_csv(dataset_path).dropna()
        self.X = self.df['EmailText']
        self.y = self.df['Label']

    def preprocess_data(self):
        vectorizer = TfidfVectorizer()
        self.X = vectorizer.fit_transform(self.X)

    def split_dataset(self, test_size, random_state):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def train_model(self):
        self.model = svm.SVC()
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        self.y_pred = self.model.predict(self.X_test)

    def display_results(self):
        table = Table(title="Model Accuracy")
        table.add_column("Model")
        table.add_column("Method")
        table.add_column("Accuracy")

        accuracy = accuracy_score(self.y_test, self.y_pred)
        table.add_row("SVM", "TF-IDF", f"{accuracy:.4f}")
        self.console.print(table)

        report = classification_report(self.y_test, self.y_pred)
        print(report)

    def plot_confusion_matrix(self):
        classes = np.unique(self.y_test)
        cm = confusion_matrix(self.y_test, self.y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        plt.xticks(ticks=range(len(classes)), labels=classes)
        plt.yticks(ticks=range(len(classes)), labels=classes)

        plt.show()
