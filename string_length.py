# Author: Salla Williams <salla.williams@gmail.com
# Date: April 2023
# Description: This script takes an Excel sheet of message data, tokenizes the
#              strings, and counts the number of words per string, adding this
#              adding this information into a new column.
# Input: preli_set_AB.xlsx (or preli_set_A.xlsx)
# Output: word_lengths.csv

import pandas as pd
from nltk.tokenize import TweetTokenizer

tk = TweetTokenizer()

# Read Excel file
df = pd.read_excel("preli_set.xlsx", sheet_name=0)

# Convert DataFrame to list of lists
data = df.values.tolist()

for i in data:
    words = tk.tokenize(i[2])
    # Update nested list
    i.append(len(words))

# Convert list of lists back to a DataFrame
dataset = pd.DataFrame(data, columns=df.columns.tolist() + ['string_length'])

# Write to a CSV file
dataset.to_csv('word_lengths.csv', index=False)
