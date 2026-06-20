# Spam Detection Model Comparison

A Jupyter Notebook project for spam detection using Natural Language Processing and classical machine learning models.

The project compares multiple text classification models for detecting spam messages using TF-IDF-based feature extraction, scikit-learn pipelines, and GridSearchCV.

## Overview

This project tests different machine learning models for binary spam classification.

The main goal is to compare model performance using the same text preprocessing, vectorization, and evaluation workflow.

Models tested:

- Multinomial Naive Bayes
- Logistic Regression
- Random Forest Classifier
- Linear Support Vector Classifier

The best results in this experiment were obtained using the **Random Forest Classifier**.

## Features

- Text preprocessing for spam classification
- CountVectorizer-based tokenization
- TF-IDF feature transformation
- scikit-learn Pipeline usage
- GridSearchCV hyperparameter tuning
- Comparison of multiple machine learning classifiers
- Evaluation using precision, recall, F1-score, and accuracy
- Jupyter Notebook experimentation workflow

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- scikit-learn
- CountVectorizer
- TfidfTransformer
- Pipeline
- GridSearchCV

## Machine Learning Pipeline

The project follows this workflow:

```text
Raw text messages
        ↓
CountVectorizer
        ↓
TfidfTransformer
        ↓
Machine Learning Classifier
        ↓
Spam / Not-Spam prediction
```

## Text Vectorization

Machine learning models cannot directly understand raw text, so the messages are converted into numerical features.

The notebook uses:

```python
CountVectorizer()
```

to convert raw text into token counts.

Then it uses:

```python
TfidfTransformer()
```

to convert token counts into TF-IDF features.

TF-IDF helps represent how important a word is in a message compared to the whole dataset.

## Model Comparison

The notebook compares four classifiers.

### Multinomial Naive Bayes

A common baseline model for text classification and spam detection.

### Logistic Regression

A linear model often used for binary classification tasks.

### Random Forest Classifier

An ensemble model based on multiple decision trees.

In this experiment, Random Forest produced the best overall test performance, reaching **97% accuracy**.

### Linear SVC

A linear Support Vector Machine classifier commonly used for text classification problems.

## GridSearchCV

The project uses GridSearchCV to test different parameter combinations and find the best configuration for each model.

General workflow:

```text
Pipeline + parameter grid
        ↓
GridSearchCV
        ↓
Best estimator
        ↓
Evaluation on test set
```

## Results

The best performing model in this experiment was the **Random Forest Classifier**.

### Evaluation on Test Set

```text
--- Evaluation on Test Set ---

              precision    recall  f1-score   support

Not-Spam          0.97      1.00      0.98       966
Spam              1.00      0.79      0.88       149

accuracy                              0.97      1115
macro avg         0.98      0.89      0.93      1115
weighted avg      0.97      0.97      0.97      1115
```

### Interpretation

The Random Forest model achieved an overall accuracy of **97%** on the test set.

For the `Not-Spam` class, the model achieved:

```text
precision: 0.97
recall:    1.00
f1-score:  0.98
```

This means the model was very strong at correctly identifying normal messages.

For the `Spam` class, the model achieved:

```text
precision: 1.00
recall:    0.79
f1-score:  0.88
```

The spam precision of **1.00** means that when the model predicted a message as spam, it was correct in this evaluation.

The spam recall of **0.79** means that the model detected 79% of spam messages, but some spam messages were still missed.

This is a common trade-off in spam detection. Higher spam recall would catch more spam, but it may also increase the number of normal messages incorrectly marked as spam.

## Example Prediction Flow

A message such as:

```text
Congratulations, you won a free prize. Click here now!
```

is transformed into numerical features using CountVectorizer and TF-IDF.

The trained model then predicts whether the message is:

```text
Spam
```

or:

```text
Not-Spam
```

## Project Structure

```text
spam-detection-model-comparison/
├── spamuri.ipynb
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/chrispsk/spam-detection-model-comparison.git
cd spam-detection-model-comparison
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If there is no `requirements.txt`, install the main dependencies manually:

```bash
pip install pandas numpy scikit-learn jupyter
```

## Usage

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
spamuri.ipynb
```

Run the notebook cells from top to bottom.

## Recommended `requirements.txt`

```text
pandas
numpy
scikit-learn
jupyter
```

## Possible Improvements

- Add a confusion matrix visualization
- Export the best model with `joblib`
- Add a Streamlit interface
- Add cross-validation results table
- Add model persistence and inference example
- Test additional models
- Add ROC-AUC or PR-AUC evaluation
