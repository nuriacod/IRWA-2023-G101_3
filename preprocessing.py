
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import FreqDist
from wordcloud import WordCloud
from array import array
from csv import reader
import matplotlib.pyplot as plt
import pandas as pd
#import tkinter as Tk
import numpy as np
import math
import json
import string
import re
import datetime
import csv
from collections import defaultdict
import collections 
from numpy import linalg as la
import nltk
nltk.download('stopwords')

def rep(m):
    """
    Description: Splits a hashtag into a list of words based on capitalization.
    """
    s=m.group(1)
    return ' '.join(re.split(r'(?=[A-Z])', s))

def preprocessing(text, word_dist, distr):
    """
    Description: This function preprocesses text data for natural language processing tasks 
    and updates the dictionary of words of word frequency . Then returns the preprocessed text.
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    # Hashtag processing: split the hashtag into a list of words. #ThisIsAnExample --> # This Is An Example
    text = re.sub(r'#(\w+)', rep, text)

    # Transform in lowercase
    text = text.lower()

    # Remove non latin characters
    text = re.sub(r'[^\x00-\x7f]',r'', text)

    # Remove urls from text in case it is a quote tweet
    text = re.sub(r'http\S+', '', text)

    # Remove words with numbers but no hashtags 
    pattern = r'\b(?!\w*#\w+)\w*\d+\w*\b'
    text = re.sub(pattern, '', text)


    #remove punctuation
    our_punct = string.punctuation #.replace('#', '')

    tablePunt = str.maketrans("","",our_punct) 

    text = text.translate(tablePunt)

    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese characters
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                           "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub(r'', text)

    #Remove one letter words
    text = re.sub(r'\b\w{1,1}\b', '', text)

    # Tokenize the text to get a list of terms
    text = text.split()

    # Eliminate the stopwords
    text = [word for word in text if not word in stop_words] 

    # Add unique words to vocabulary
    if distr:
        word_dist.update(text)
    
    # Perform stemming
    text = [stemmer.stem(word) for word in text]
    
    text = ' '.join(text)

    return text

def get_fields(line,doc_id,word_dist):
    """
    Extracts and formats various Twitter data fields from a JSON object and returns them 
    as a concatenated string.
    """
    
    # Transform each line of the json file into a python dictionary
    if len(line['entities']["urls"]) > 0:
        our_url = line['entities']['urls'][0]['expanded_url']
    else:
        our_url = 'https://twitter.com/'+str(line['user']['screen_name'])+'/status/'+str(line['id'])

    ht_list = ''
    for element in line['entities']['hashtags']:
        ht_list += ' '+element['text']

    
    our_str = str(doc_id) + ' | ' + str(line['id']) + ' | ' + preprocessing(line['full_text'],word_dist, True)+ ' | ' + \
            str(line['created_at']) + ' | ' + ht_list + ' | ' + str(line['favorite_count']) + ' | ' + \
            str(line['retweet_count']) + ' | ' + our_url + ' | '+ str(line['user']['followers_count']) + ' | ' + str(line['user']['verified'])
    
    text = line['full_text']
    return our_str, text