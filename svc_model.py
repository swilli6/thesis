# Author: Salla Williams <salla.williams@gmail.com>
# Date: April 2023
# Description: This script trains a machine learning model for binary classification
#              on text data and evaluates performance using accuracy, precision,
#              recall, and F1 score.
#              Optionally, it also applies the SMOTE algorithm for oversampling, and plots coefficients.
# Input: train_AB.xlsx and test_AB.xlsx (or train_A.xlsx and test_A.xlsx)
# Output: performance evaluation metrics and a confusion matrix

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter
import time
import numpy as np
import matplotlib.pyplot as plt

# Import training and test data from Excel, extract strings into a separate list
train_df = pd.read_excel("train_AB.xlsx", sheet_name=0)
test_df = pd.read_excel("test_AB.xlsx", sheet_name=0)
train_strings = list(train_df[2]) 
test_strings = list(test_df[2]) 

# Create an instance of the TfidfVectorizer class
vectorizer = TfidfVectorizer(stop_words='english') # Toggle with the vectorizer below to include stop words
#vectorizer = TfidfVectorizer()

# Fit the vectorizer on the training data
vectorizer.fit(train_strings)

# Transform the text data into vectors
train_vectors = vectorizer.transform(train_strings)
test_vectors = vectorizer.transform(test_strings)

# Convert target labels to binary format
train_df[6] = train_df[6].astype('category')
test_df[6] = test_df[6].astype('category')

encode_map = {
    'H': 1,
    '0': 0
}

train_df[6].replace(encode_map, inplace=True)
test_df[6].replace(encode_map, inplace=True)

# Define X (strings in vectorized form) and y (corresponding gold-standard tags)
X_train = train_vectors.toarray()
X_test = test_vectors.toarray()
y_train = list(train_df[6])
y_test = list(test_df[6]) 

# Initialize the sequence
print('Welcome to the hostility detecting SVC model!')
print()
print('There are', len(X_train), 'samples in the training set and', len(X_test), 'samples in the test set.')
print()

# Create the SVC model
svc_model = SVC(kernel='linear')

# Train the model on the training data
print('Training the SVC model...')
print()
svc_model.fit(X_train, y_train)

# Predict the classes of the test data
print('Predicting the labels of the test set...')
print()
import time
start = time.time()
y_pred = svc_model.predict(X_test)
end = round(time.time()-start,2)
print("The prediction process for", len(X_test), "samples took",end,"seconds.")
print("This is the equivalent of",end/(len(X_test)),"seconds per sample.")
print()

# Evaluate accuracy, precision, recall, F1 score
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    print('Accuracy: {:.2%}'.format(accuracy))
    print('Precision: {:.2%}'.format(precision), '(How many samples labelled hostile were actually hostile)')
    print('Recall: {:.2%}'.format(recall), '(How many of the hostile samples in the data set were found)')
    print('F1 score: {:.2%}'.format(f1))
    print()
    print(classification_report(y_test, y_pred))
    print()
    print(confusion_matrix(y_test, y_pred))
    print()

evaluate(svc_model, X_test, y_test)

# SMOTE oversampling

smoteinput = input('Do you wish to oversample the training set using SMOTE? Y/N ')
if smoteinput == 'Y' or smoteinput == 'y':
   print()
   sstrategy = input('Please input sampling strategy (0.15-1.0) ')
   print()
   # Count original samples
   counter = Counter(y_train)
   print('Original samples: ', counter)
   print()
   # Introduce SMOTE oversampling algorithm
   smt = SMOTE(sampling_strategy=float(sstrategy))
   X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
   # Count new samples including SMOTE
   counter = Counter(y_train_sm)
   print('Samples including synthetic samples: ', counter)
   print()
   # Train the model on the SMOTE training data
   svc_model.fit(X_train_sm, y_train_sm)
   # Predict the classes of the test data
   y_pred = svc_model.predict(X_test)
   # Evaluate accuracy, precision, recall
   accuracy = accuracy_score(y_test, y_pred)
   precision = precision_score(y_test, y_pred, pos_label=1)
   recall = recall_score(y_test, y_pred, pos_label=1)
   f1 = f1_score(y_test, y_pred, pos_label=1)
   print('Accuracy: {:.2%}'.format(accuracy))
   print('Precision: {:.2%}'.format(precision), '(How many samples labelled hostile were actually hostile)')
   print('Recall: {:.2%}'.format(recall), '(How many of the hostile samples in the data set were found)')
   print('F1 score: {:.2%}'.format(f1))
   print()
   print(classification_report(y_test, y_pred))
   print()
   print(confusion_matrix(y_test, y_pred))
   print()
else:
   pass

# Plotting coefficients

# Refit the model with original samples (without SMOTE)
svc_model.fit(X_train, y_train)

# Plot coefficients in a graph to reveal most salient words
def plot_coefficients(classifier, feature_names, top_features=15):
   coef = classifier.coef_.ravel()
   top_positive_coefficients = np.argsort(coef)[-top_features:]
   top_negative_coefficients = np.argsort(coef)[:top_features]
   top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
   plt.figure(figsize=(15, 5))
   colors = ['green' if c < 0 else 'red' for c in coef[top_coefficients]]
   plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
   feature_names = np.array(feature_names)
   plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
   plt.show()

plotinput = input('Do you wish to plot the top 15 coefficients (most salient words)? Y/N ')
if plotinput == 'Y' or plotinput == 'y':
   plot_coefficients(svc_model, vectorizer.get_feature_names_out())
else:
   print('Thank you for using the hostility detecting SVC model!')

