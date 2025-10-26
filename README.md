# 5 AI Projects Compilation (Based on NLP CV and Forecasting)
A curated collection of applied AI and Machine Learning projects. This portfolio showcases practical skills in NLP and Computer Vision, featuring a sentiment analyzer, chatbot, emotion detector, cryptocurrency forecaster, and an ANPR system. Demonstrates end-to-end model development from dataset to deployment-ready solution.

## Overview

This repository showcases five complete Machine Learning and Deep Learning projects covering Natural Language Processing (NLP), Time Series Forecasting, and Computer Vision domains.  
Each project demonstrates end-to-end implementation — including data ingestion, preprocessing, feature engineering, model training, evaluation, and performance optimization.

All projects are implemented in Python (Google Colab) using popular libraries such as Scikit-learn, TensorFlow, Keras, NLTK, Pandas, and OpenCV.

---

## Project List

| Sl. No.      | Title                              | Dataset                                                                                                               |
|--------------|------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 1            | Hotel Reviews Sentiment Analysis   | [Kaggle Dataset](https://www.kaggle.com/code/jonathanoheix/sentiment-analysis-with-hotel-reviews/notebook#Conclusion) |
| 2            | Cryptocurrency Price Prediction    | [Kaggle Dataset](https://www.kaggle.com/code/taniaj/cryptocurrency-price-forecasting/comments)                        |
| 3            | Emotion Detector                   | [Kaggle Dataset](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)                                    |
| 4            | ChatBot                            | [Kaggle Dataset](https://www.kaggle.com/code/ahmedmoabdelkader/my-chatbot/notebook)                                   |
| 5            | Automatic Number Plate Recognition | [Kaggle Dataset](https://www.kaggle.com/code/sarthakvajpayee/license-plate-recognition-using-cnn/notebook)            |

---

## 1. Hotel Reviews Sentiment Analysis

### Objective  
Analyze hotel reviews and classify them as **positive** or **negative** based on sentiment.

### Preprocessing  
- Lowercasing, stopword removal, tokenization  
- TF-IDF and Bag of Words vectorization  

### Models  
- Logistic Regression, Naive Bayes, Random Forest, SVM

### Evaluation  
Accuracy, Precision, Recall, F1-Score

### Best Model  
Logistic Regression (~90% accuracy)

---

## 2. Cryptocurrency Price Prediction

### Objective  
Predict cryptocurrency price trends using historical data.

### Features  
- Moving averages, rolling standard deviation, lag features

### Models  
- Linear Regression, Decision Tree, Random Forest, ARIMA, LSTM

### Tuning  
GridSearchCV, Keras Tuner (learning rate, neurons, dropout)

### Metrics  
MAE, RMSE, R² Score

### Best Model  
LSTM — lowest RMSE

---

## 3. Emotion Detection

### Objective  
Classify facial expressions into emotions like Happy, Sad, Angry, etc.

### Preprocessing  
- Grayscale, resize (48x48), normalization, data augmentation

### Model  
- CNN (Conv2D, MaxPooling, Dropout, Dense, Softmax)

### Training  
Optimizer: Adam | Epochs: 50 | Accuracy: ~92%

### Future Work  
ResNet/VGG transfer learning, web deployment

---

## 4. ChatBot

### Objective  
Develop an intent-based chatbot using NLP and neural networks.

### Preprocessing  
Tokenization, Lemmatization, Bag of Words

### Model  
Feedforward Neural Network (Dense layers + Softmax)

### Accuracy  
~88% on intent classification

### Future Work  
Add contextual memory using BERT or GPT embeddings

---

## 5. Automatic Number Plate Recognition (ANPR)

### Objective  
Detect and recognize vehicle number plates using CNN and OpenCV.

### Steps  
- Grayscale → Edge Detection → Contour Extraction → Character Segmentation

### Model  
CNN (Conv2D, Pooling, Flatten, Dense)

### Accuracy  
~95% character recognition

### Future Work  
Use YOLO/SSD for real-time detection, Raspberry Pi deployment

---

## Technologies Used

| Category      | Tools / Libraries     |
|---------------|-----------------------|
| Languages     | Python                |
| Data Handling | Pandas, NumPy         |
| Visualization | Matplotlib, Seaborn   |
| NLP           | NLTK, spaCy, Word2Vec |
| ML            | Scikit-learn          |
| DL            | TensorFlow, Keras     |
| CV            | OpenCV                |
| Deployment    | Flask, Streamlit      |

---

## Performance Summary

| Project           | Domain      | Best Algorithm      | Metric   | Score  |
|-------------------|-------------|---------------------|----------|--------|
| Hotel Sentiment   | NLP         | Logistic Regression | Accuracy | ~90%   |
| Crypto Prediction | Time Series | LSTM                | RMSE     | Lowest |
| Emotion Detection | CV          | CNN                 | Accuracy | ~92%   |
| ChatBot           | NLP         | NN                  | Accuracy | ~88%   |
| ANPR              | CV          | CNN                 | Accuracy | ~95%   |

---

# Brief on the Projects

1. Hotel Review Sentiment Analysis

This project aims to analyze customer hotel reviews and classify them as positive or negative sentiments. The system will use Natural Language Processing (NLP) and Machine Learning algorithms to process text data, clean it, and identify underlying sentiment trends. The problem addresses the challenge of understanding large-scale customer feedback efficiently for hospitality businesses to improve services. Previous works in sentiment analysis include approaches like Naïve Bayes, Logistic Regression, and Deep Learning models such as LSTM. Our approach will focus on text preprocessing (tokenization, stopword removal, lemmatization), feature engineering (TF-IDF, Word2Vec), and classification models (Logistic Regression, Random Forest, and LSTM).

2. Cryptocurrency Price Prediction

The project focuses on predicting cryptocurrency prices (e.g., Bitcoin, Ethereum) using historical data and time-series forecasting models. The problem of cryptocurrency volatility has significant financial implications for investors and traders. We will analyze time-series datasets containing daily open, high, low, close prices, and volumes. The model will be trained to forecast future prices using Machine Learning (ML) and Deep Learning (DL) methods such as ARIMA, LSTM, and Random Forest. The motivation is to explore the feasibility of accurate price prediction in volatile markets through advanced ML techniques.

3. Emotion Detection from Text

The goal of this project is to classify emotions such as joy, sadness, anger, fear, and surprise from text input. It uses NLP techniques and emotion-labeled datasets to understand human feelings expressed in textual form. This project helps enhance applications like chatbots, customer service, and mental health monitoring. Our approach includes preprocessing, word embeddings (TF-IDF, Word2Vec), and classification using ML algorithms like Logistic Regression, SVM, and Random Forest.

4. Chatbot for General Query Response

This project involves designing an intelligent chatbot capable of answering basic user queries conversationally. Using NLP and intent recognition, the chatbot identifies user intent and provides appropriate responses. The chatbot integrates a knowledge base or corpus of predefined responses. We will use NLP techniques like tokenization, lemmatization, and intent classification models such as Logistic Regression or Neural Networks.

5. Automatic Number Plate Recognition (ANPR)

This project aims to detect and recognize vehicle number plates from images using image processing and computer vision techniques. The motivation stems from the need for automation in vehicle identification for parking systems, tolls, and security. The system captures vehicle images, detects the license plate using edge detection or YOLO models, and recognizes characters using Optical Character Recognition (OCR).

3. Deliverables of the Projects

For all projects, the deliverables include:

 - Complete data preprocessing and feature engineering pipeline.

 - Model training, testing, and performance evaluation reports.

 - Visualization of results.

 - Final trained models and documentation.

Questions the Models Aim to Answer

 - Hotel Review Analysis: What is the sentiment of customer feedback — positive or negative?

 - Crypto Prediction: What will be the predicted closing price of Bitcoin for the next day?

 - Emotion Detection: Which emotion does a text express?

 - Chatbot: Can the system provide relevant answers to user questions?

 - ANPR: Can the model detect and accurately recognize a vehicle’s license plate?

Details, Findings, and Expected Outcome

 - Hotel Review Analysis: Expect accuracy above 85% using TF-IDF and Logistic Regression.

 - Crypto Prediction: Forecast future cryptocurrency prices with minimum RMSE.

 - Emotion Detection: Achieve F1-scores above 0.80 with Word2Vec + Random Forest.

 - Chatbot: Enable smooth response flow for predefined question categories.

 - ANPR: Accurately detect and read vehicle plates under different lighting conditions.


# Milestones

| Milestone	                        | Description
|---------------------------------------|----------------------------------------------------------------|
|Define a Problem	                |Identify and describe the business problem for each project.    |
|Understanding the Business Problem	|Determine real-world context and motivation.                    |
|Get the Data	                        |Acquire datasets from Kaggle or public sources.                 |
|Explore and Pre-process Data	        |Handle missing values, normalize, and clean data.               |
|Choosing the Python Platform	        |Select Jupyter/Colab for development.                           |
|Create Features	                |Perform feature extraction (TF-IDF, embeddings, image features).|
|Exploratory Data Analysis (EDA)	|Visualize data distribution and patterns.                       |
|Create Model	                        |Implement ML and DL algorithms.                                 |
|Model Evaluation	                |Evaluate using metrics like Accuracy, F1-score, RMSE.           |
|Report Writing	                        |Summarize approach, results, and findings.                      |
|Project Submission	                |Submit final notebook, report, and documentation.               |

---

## Author

**Dr. Pushkar Srivastava**  
AI & LIMS Specialist  
Germany
