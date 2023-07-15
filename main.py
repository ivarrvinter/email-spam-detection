from classifier import EmailClassifier

def main():
    classifier = EmailClassifier()

    classifier.load_dataset('/kaggle/input/email-classification/emails.csv')
    classifier.preprocess_data()

    classifier.split_dataset(test_size=0.2, random_state=123)

    classifier.train_model()

    classifier.evaluate_model()

    classifier.display_results()

    classifier.plot_confusion_matrix()

if __name__ == '__main__':
    main()
