# Author: Salla Williams <salla.williams@gmail.com>
# Date: April 2023
# Description: This script takes an Excel sheet of message data, detects the
#              language of each strings, extracts the English-language strings
#              and performs VADER sentiment analysis on these. A tag of "H" or "0"
#              is added to each sample.
# Input: raw_data_A.xlsx (or raw_data_B.xlsx)
# Output: preli_set_A.xlsx (or preli_set_B.xlsx)

import pandas as pd
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Import raw data from Excel, form samples into a list of lists
df = pd.read_excel('raw_data_B.xlsx')
data = df.values.tolist()

# Perform language indentification and extract English strings into a separate list
english_strings = []
for i in data:
    try:
        language = detect(str(i[1]))
        i.append(language)
        if language == 'en':
            english_strings.append(i)
    except:
        pass

# Perform preliminary hostility tagging using VADER
def sentiment_score(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    if sentiment_dict['compound'] <= -0.05 :
        return 'H'
    else:
        return '0'

# Add predicted hostility label (H/0) to each sample
for i in english_strings:
    i.append(sentiment_score(i[1]))

# Export the results into Excel for manual checking
dataset = pd.DataFrame(english_strings)
dataset.to_excel('preli_set_B.xlsx')