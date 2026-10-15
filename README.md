# Twitter Sentiment Analysis Using Machine Learning

## Project Overview

This project explores sentiment classification of tweets using Natural Language Processing (NLP), traditional machine learning, and deep learning.

The objective is to classify tweets into three sentiment categories: **positive, negative, and neutral**.

The project compares multiple classification algorithms using text features extracted with CountVectorizer and explores Word2Vec embeddings with an LSTM network for sequence-based sentiment classification.

## Dataset

- Total records in the original dataset: 27,480 tweets
- Records after cleaning empty processed text: 27,476 tweets
- Input text column: `text`
- Target column: `sentiment`
- Processed text column: `clean_text`
- Classification type: Multiclass classification
- Classes: Positive, Negative, Neutral

The data was split into training and testing sets using a stratified 50/50 split with `random_state=42`.

- Training samples: 13,738
- Testing samples: 13,738

## Text Preprocessing and Feature Extraction

Text preprocessing was performed to prepare tweets for sentiment classification.

**CountVectorizer** was used to convert cleaned tweets into numerical bag-of-words features for the traditional machine learning experiments. The vectorizer was fitted on the training data and then applied to the test data to avoid data leakage.

**TruncatedSVD** was used with KNN to reduce the dimensionality of the text feature space.

For the deep learning experiment, **Word2Vec** was used to learn 100-dimensional word embeddings from the training text. These embeddings were supplied as sequences to an LSTM network.

## Machine Learning Models

The following models have been implemented and evaluated:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVC (Support Vector Classification)
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost

## Deep Learning Model

### Word2Vec + LSTM

A Word2Vec model was trained on the training tweets to generate 100-dimensional word embeddings. The resulting word vectors were arranged into sequences and passed to an LSTM model for three-class sentiment classification.

The LSTM architecture consists of:

- Masking layer for zero-padding
- LSTM layer with 64 units
- Dense layer with 32 units and ReLU activation
- Dropout layer with a rate of 0.3
- Softmax output layer with three classes

The model was trained using the Adam optimizer and sparse categorical crossentropy loss.

## Model Performance Comparison

The table below summarizes the recorded test-set results.

| Model | Test Accuracy | Macro F1-score |
|---|---:|---:|
| Logistic Regression | 68.30% | 0.68 |
| Linear SVC | 65.64% | 0.66 |
| XGBoost | 65.05% | 0.64 |
| Multinomial Naive Bayes | 63.87% | 0.64 |
| Decision Tree | 63.43% | 0.63 |
| Word2Vec + LSTM | 60.58% | 0.60 |
| Random Forest | 58.71% | 0.54 |
| KNN with TruncatedSVD | 48.08% | 0.47 |

*Macro F1-scores are rounded to two decimal places.*

These results reflect the current experiments. Logistic Regression achieved the highest recorded test accuracy among the evaluated models.

## Evaluation Metrics

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Macro and weighted averages
- Classification reports
- Confusion matrices

Macro F1-score is useful for comparing performance across the three sentiment classes, while confusion matrices help identify which sentiments are frequently misclassified.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- NLTK
- Gensim (Word2Vec)
- TensorFlow / Keras
- CountVectorizer
- TruncatedSVD
- XGBoost
- Jupyter Notebook

## Key Observations

- Logistic Regression achieved the highest test accuracy among the models evaluated so far.
- Linear SVC and XGBoost also produced competitive results.
- KNN achieved the lowest test accuracy among these experiments.
- Random Forest showed differences between precision and recall across sentiment classes.
- Word2Vec + LSTM provided a comparison between traditional machine learning and sequence-based deep learning.
- Evaluating class-level metrics alongside accuracy helps provide a more complete picture of model performance.

## Planned Improvements

- Explore transformer architectures and fine-tune a pretrained BERT-family model in a separate experiment.
- Investigate model errors using confusion matrices and per-class metrics.
- Explore hyperparameter tuning and methods to improve generalization.
- Compare traditional machine learning, deep learning, and transformer-based approaches.

## Project Status

The initial machine learning experiments and the Word2Vec + LSTM experiment have been completed. Model comparison is documented using test accuracy and macro F1-score.

Further work will focus on interpreting model performance, understanding classification errors, and exploring transformer-based sentiment analysis.
