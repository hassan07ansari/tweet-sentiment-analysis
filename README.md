# Twitter Sentiment Analysis Using Machine Learning

## Project Overview

This project explores sentiment classification of tweets using Natural Language Processing (NLP) and machine learning. The objective is to classify tweets into three sentiment categories: positive, negative, and neutral.

The project uses text preprocessing and CountVectorizer to convert text into numerical features. Multiple machine learning classifiers are being evaluated to compare their performance on the same sentiment classification task.

## Dataset

* Total records: 27,480 tweets
* Input text: `text`
* Target column: `sentiment`
* Processed text: `clean_text`
* Classification type: Multiclass classification
* Classes: Positive, Negative, Neutral

## Text Preprocessing and Feature Extraction

The project includes text cleaning and preprocessing to prepare tweets for classification.

CountVectorizer is used to convert the cleaned text into a sparse bag-of-words representation. The vectorizer is fitted on the training data and then applied to the test data to avoid data leakage during feature extraction.

## Machine Learning Models

The following classifiers have been explored:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Multinomial Naive Bayes
* Decision Tree Classifier
* Random Forest Classifier
* Linear Support Vector Machine (Linear SVM)

TruncatedSVD was also used with KNN to reduce the dimensionality of the text features.

## Initial Results

The following results are from the completed experiments recorded so far.

| Model                   | Accuracy | Macro F1-score |
| ----------------------- | -------: | -------------: |
| Logistic Regression     |   68.14% |           0.68 |
| Linear SVM              |   65.82% |           0.66 |
| Multinomial Naive Bayes |   64.40% |           0.64 |
| Decision Tree           |   64.00% |           0.64 |
| KNN with TruncatedSVD   |   47.01% |           0.46 |

Random Forest evaluation and additional model experiments remain in progress.

## Evaluation

Model performance is evaluated using classification accuracy, precision, recall, F1-score, classification reports, and confusion matrices.

Macro F1-score is useful for comparing performance across all three sentiment classes.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLP text preprocessing
* CountVectorizer
* TruncatedSVD
* Machine Learning Classification

## Planned Improvements

* Complete Random Forest evaluation
* Experiment with XGBoost
* Compare additional feature extraction techniques, including TF-IDF and Word2Vec
* Develop a Word2Vec + LSTM model
* Explore CBOW and other embedding approaches
* Implement a Transformer-based sentiment classifier
* Compare models using consistent evaluation metrics
* Analyze model errors and improve classification performance

## Project Status

The initial CountVectorizer-based machine learning experiments have been developed, and model comparison is ongoing. Further experiments with feature representations, deep learning, and transformer architectures are planned.
