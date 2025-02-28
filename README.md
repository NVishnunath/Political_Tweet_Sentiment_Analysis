# From Traditional to Advanced Machine Learning: A Comparative Study of Political Tweet Sentiment Analysis

## Project Overview:
This research project explores the effectiveness of traditional machine learning models versus deep learning approaches in sentiment analysis of political tweets. With the increasing role of social media in shaping political discourse, this study focuses on analyzing tweets related to Indian political figures, Narendra Modi and Rahul Gandhi. The research evaluates the performance of Support Vector Machine (SVM) and Long Short-Term Memory (LSTM) models in classifying tweets into sentiment categories. The project follows a structured data processing pipeline, including data collection, preprocessing, model training, evaluation, and comparative analysis.

## Technology Used:
- **Programming Language:** Python
- **Libraries & Frameworks:**
    - NLP: NLTK, SpaCy
    - Machine Learning: Scikit-learn
    - Deep Learning: TensorFlow, Keras
    - Data Processing: Pandas, NumPy
    - Visualization: Matplotlib, Seaborn
- **Dataset:** Political tweet dataset sourced from Kaggle
- **Pre-trained Models:** Hugging Face Transformers for sentiment annotation

## System Design & Workflow:
1. **Data Collection & Transformation:**
     - Tweets related to Indian political elections extracted from Kaggle.
     - Preprocessed dataset to remove noise (URLs, hashtags, punctuation).
     - Tokenized and labeled tweets for sentiment classification (positive, negative).
2. **Data Processing & Feature Engineering:**
     - Text normalization (lowercasing, lemmatization, stopword removal).
     - Sentiment annotation using Hugging Face Transformer models.
     - TF-IDF feature extraction for SVM and tokenization for LSTM.
3. **Model Implementation:**
     - **Support Vector Machine (SVM):** Applied TF-IDF features and trained using L1 regularization for sentiment classification.
     - **Long Short-Term Memory (LSTM):** Implemented for sequential text processing with embedding layers and dropout for regularization.
4. **Model Evaluation:**
     - **Metrics used:** Accuracy, Precision, Recall, F1-score.
     - **SVM Accuracy:** 86%
     - **LSTM Accuracy:** 87% (demonstrated superior performance in capturing text dependencies).
5. **Comparative Analysis:**
     - SVM is computationally efficient but lacks context awareness.
     - LSTM effectively captures contextual dependencies but requires higher computational resources.


## Results:
- **Best Performing Model:** LSTM with an accuracy of 87%.
- **SVM Performance:** 86% accuracy, making it a viable alternative for lower-resource environments.
- **Sentiment Distribution:**
    - 74.5% of positive sentiments favor BJP.
    - 25.5% of positive sentiments favor INC.
- **Data Imbalance:** Addressed using SMOTE (Synthetic Minority Oversampling Technique).
- **Visualization Tools Used:** Confusion Matrix, ROC Curve, Accuracy Plots.

### Conclusion:
This study highlights the strengths and limitations of both traditional and deep learning approaches for sentiment analysis. While SVM offers computational efficiency, LSTM provides better contextual understanding, making it ideal for political tweet analysis. The findings demonstrate that deep learning techniques, specifically LSTM, outperform traditional ML methods in sentiment classification tasks.

### Note:
1. Please keep the dataset (IndianElection19TwitterData) in the Downloads for Data Preprocessing and Visualization file.
2. Use (cleaned_tweets) data for the model development file (the cleaned_tweets CSV automatically generated from the first Python file).
