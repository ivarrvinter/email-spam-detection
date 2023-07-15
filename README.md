# Email Classification using Machine Learning

This repository contains code for classifying emails using machine learning techniques. The project utilizes a dataset of emails and applies a support vector machine (SVM) model for classification. The classification is performed based on the content of the emails using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

## Features

- Preprocessing: The code preprocesses the email text data, including tokenization, TF-IDF vectorization, and splitting the dataset into training and testing sets.
- SVM Classification: The SVM model is trained on the preprocessed data to classify emails into relevant categories.
- Evaluation: The model's performance is evaluated using accuracy metrics, a classification report, and a confusion matrix.
- Visualization: The code generates visualizations, including a confusion matrix plot, to provide further insights into the classification results.

## Dataset

The code expects an email dataset in CSV format with two columns: 'EmailText' and 'Label'. The 'EmailText' column contains the email content, and the 'Label' column specifies the corresponding email category or class.

## Usage

1. Ensure you have Python and the necessary libraries (pandas, numpy, scikit-learn, rich, matplotlib, seaborn) installed.
2. Clone this repository: `git clone https://github.com/ivarrvinter/email-spam-detection.git`
3. Navigate to the project directory: `cd email-spam-detection`
4. Install the required dependencies
5. Replace the dataset file (`emails.csv`) with your own email dataset, ensuring it follows the required format.
6. Run the main.py file: `python main.py`
7. Explore the classification results, including accuracy metrics, classification reports, and visualization of the confusion matrix.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
