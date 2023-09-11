# CodeSoft
CodeSoft internship projects (01 september 2023 to 30 september 2023)
# Project 1 :- Creating a machine learning model that can predict the genre of a movie based on its textual information.
# Movie Genre Classification using Natural Language Processing

## Overview

This program is designed to classify movie genres based on movie summaries using Natural Language Processing (NLP) techniques. It utilizes the Multinomial Naive Bayes classifier to predict the genre of a movie based on its summary text. The program also performs data preprocessing, feature extraction using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, and provides performance metrics for the classification.

## Dependencies

Before running the program, make sure you have the following Python packages installed:

- pandas
- numpy
- nltk
- re
- scikit-learn

You can install these packages using `pip`:

```bash
pip install pandas numpy nltk scikit-learn

1.Data Preparation

### Place your movie data in a text file named "train_data.txt" with the following format:
- S_No:::movie_name:::genre:::summary
- 1:::Movie Title 1 (Year):::Genre 1:::Movie summary 1
- 2:::Movie Title 2 (Year):::Genre 2:::Movie summary 2
- ...
2. Run the following script to preprocess the data and convert it to a CSV file:

- python preprocess_data.py
3. Feature Extraction

- The program uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to extract features from movie summaries. This step is done automatically during the model training.
4. Training and Testing the Model

- Split the data into training and testing sets using the following command:
- python train_test_split.py
- Train the Multinomial Naive Bayes classifier:

- python train_model.py
- Test the model and view classification results:

- python test_model.py
## Results
- The program provides performance metrics such as confusion matrix and classification report to evaluate the Multinomial Naive Bayes classifier's accuracy in predicting movie genres.

## Note
- Make sure to customize the program according to your specific dataset and requirements.
- Feel free to reach out if you have any questions or need further assistance!





