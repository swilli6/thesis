# Author: Salla Williams <salla.williams@gmail.com>
# Date: April 2023
# Description: This script takes an Excel sheet of message data, shuffles the
#              rows randomly, splits the first 30% into a test set and the
#              remaining data (70%) into a training set.
# Input: preli_set_A.xlsx (or preli_set_AB.xlsx)
# Output: test_A.xlsx and train_A.xlsx (or test_AB.xlsx and train_AB.xlsx)

import pandas as pd

# Read Excel file
df = pd.read_excel("preli_set_AB.xlsx", sheet_name=0)

# Shuffle samples
df = df.sample(frac=1)

# Form test (30%) and training (70%) sets
test = df.values[:(int(df.shape[0]*.3))]
train = df.values[(int(df.shape[0]*.3)):]

# Convert list back to a DataFrame
test_set = pd.DataFrame(test)
train_set = pd.DataFrame(train)

# Write to an Excel file
test_set.to_excel('test_AB.xlsx')
train_set.to_excel('train_AB.xlsx')
