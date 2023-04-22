# Author: Salla Williams <salla.williams@gmail.com>
# Date: April 2023
# Description: This script trains a deep neural network for binary classification
#              on text data and evaluates performance using accuracy, precision,
#              recall, and F1 score.
# Input: train_AB.xlsx and test_AB.xlsx (or train_A.xlsx and test_A.xlsx)
# Output: performance evaluation metrics and a confusion matrix

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import time
import torch
import torch.nn as nn
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import numpy as np

# Import training and test data from Excel, extract strings into a separate list
train_df = pd.read_excel("train_AB.xlsx", sheet_name=0)
test_df = pd.read_excel("test_AB.xlsx", sheet_name=0)
train_strings = list(train_df[2]) 
test_strings = list(test_df[2]) 

# Create the vectorizer
vectorizer = CountVectorizer(stop_words='english') # Toggle with the vectorizer below to include stop words
# vectorizer = CountVectorizer()
 
# Transform the corpus data into vectors
X = vectorizer.fit_transform(train_strings)
 
# Prepare data frame for MLP model
# Features
CountVectorizedData=pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
# Target labels
CountVectorizedData['Hostility']=train_df[6]

# Load the word vectors from Word2Vec model
GoogleModel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True,)

# List the vocabulary present in the corpus matrix
WordsVocab=CountVectorizedData.columns[:-1]

# Create one vector for each sentence
def Text2Vec(input_data):
    # Convert the text to numeric data
    X = vectorizer.transform(input_data)
    CountVecData=pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    # Create empty dataframe for vectors
    W2V_Data=pd.DataFrame()
    # Loop through each row for the data
    for i in range(CountVecData.shape[0]):
        # Initiate a sentence with all zeros
        Sentence = np.zeros(300)
        # Loop through each word in the sentence and store the Word2Vec vector if present
        for word in WordsVocab[CountVecData.iloc[i , :]>=1]:
            if word in GoogleModel.key_to_index.keys():    
                Sentence = Sentence + GoogleModel[word]
            # If word is not found in Word2Vec, an empty vector is stored
            else:
                Sentence = Sentence
        # Concatenate the vector to the dataframe
        W2V_Data=pd.concat([W2V_Data, pd.DataFrame([Sentence])], ignore_index=True)
    return(W2V_Data)

# Vectorize training and test sets
train_w2v=Text2Vec(train_strings)
test_w2v=Text2Vec(test_strings)

# Convert dataframe to Numpy array, define X_train and X_test
X_train = train_w2v.values
X_test = test_w2v.values

# Convert target labels to binary format
train_df[6] = train_df[6].astype('category')
test_df[6] = test_df[6].astype('category')

encode_map = {
    'H': 1,
    '0': 0
}

train_df[6].replace(encode_map, inplace=True)
test_df[6].replace(encode_map, inplace=True)

# Define y_train and y_test
y_train = list(train_df[6])
y_test = list(test_df[6])

# Define MLP training parameters
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Establish training data architecture
class TrainData(Dataset):
    
    def __init__(self, X_data, y_data):
        assert len(X_data) == len(y_data)
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

train_data = TrainData(torch.FloatTensor(X_train), 
                       torch.FloatTensor(y_train))

# Establish test data architecture 
class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
test_data = TestData(torch.FloatTensor(X_test))

# Load training and test sets
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=1)

# Define MLP model
class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Define input, hidden, and output layers
        input_feat = len(X_train[0])
        self.layer_1 = nn.Linear(input_feat, 128) 
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(32)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

# Define whether CPU or GPU should be used for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BinaryClassification()
model.to(device)

# Define optimizer and loss function
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.NAdam(model.parameters(), lr=LEARNING_RATE)

# Define performance metric
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

# Train MLP model
model.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

start = time.time()

# Predict labels for test set samples
y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

# Calculate time taken to predict label for each sample
end = round(time.time()-start,2)
print()
print("The prediction process for", len(X_test), "samples took",end,"seconds.")
print("This is the equivalent of",end/(len(X_test)),"seconds per sample.")
print()

# Evaluate accuracy, precision, recall, F1 score
accuracy = accuracy_score(y_test, y_pred_list)
precision = precision_score(y_test, y_pred_list, pos_label=1)
recall = recall_score(y_test, y_pred_list, pos_label=1)
f1 = f1_score(y_test, y_pred_list, pos_label=1)

print('Accuracy: {:.2%}'.format(accuracy))
print('Precision: {:.2%}'.format(precision), '(How many samples labelled hostile were actually hostile)')
print('Recall: {:.2%}'.format(recall), '(How many of the hostile samples in the data set were found)')
print('F1 score: {:.2%}'.format(f1))
print()
print(classification_report(y_test, y_pred_list))
print()
print(confusion_matrix(y_test, y_pred_list))
print()
