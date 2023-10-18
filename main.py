from collections import defaultdict
from array import array
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import FreqDist
import math
import numpy as np
import collections
from numpy import linalg as la
import json
import string
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as Tk
import datetime
import csv
from csv import reader


import nltk
nltk.download('stopwords')


def rep(m):
    s=m.group(1)
    return ' '.join(re.split(r'(?=[A-Z])', s))

def preprocessing(text, word_dist):
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

    # Perform stemming
    # text = [stemmer.stem(word) for word in text]
    word_dist.update(text)

    text = ' '.join(text)

    return text

def temporal(list_of_tweets):

    # Create a dictionary to store the tweet counts per day
    tweet_counts = {}

    # Assuming your 'created_at' field is a list of timestamps
    for tweet in list_of_tweets:
        created_at = tweet.split('|')[3].strip()
        print(created_at)

        created_at = datetime.datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')

        date = created_at.date()
        
        # Update the tweet count for this date
        tweet_counts[date] = tweet_counts.get(date, 0) + 1

    # Sort the dictionary by date
    sorted_tweet_counts = sorted(tweet_counts.items())

    # Extract the dates and corresponding tweet counts
    dates, counts = zip(*sorted_tweet_counts)

    # Create a line plot to visualize tweet activity over time
    plt.figure(figsize=(10, 6))
    plt.plot(dates, counts, marker='o')
    plt.title('Temporal Distribution of Tweets')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.grid()
    plt.show()

def csv_to_dict(filepath):
    d = {}
    with open(filepath, 'r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')  # Use tab as the delimiter
        for row in reader:
            if len(row) == 2:
                key, value = row
                d[str(value)] = key
    return d

def main():
    docs_path = './IRWA_data_2023/Rus_Ukr_war_data.json'
    dict_path = './IRWA_data_2023/Rus_Ukr_war_data_ids.csv'
    # Initialize dictionary for word count
    word_dist = FreqDist()
    list_of_tweets= []
    docid_tweetid = csv_to_dict(dict_path)
    with open(docs_path) as fp:
        for i, line in enumerate(fp):
            # Transform each line of the json file into a python dictionary
            json_line = json.loads(line)
            our_docid = docid_tweetid[str(json_line['id'])]
            if len(json_line['entities']["urls"]) > 0:
                our_url = json_line['entities']['urls'][0]['expanded_url']
            else:
                our_url = 'https://twitter.com/'+str(json_line['user']['screen_name'])+'/status/'+str(json_line['id'])

            ht_list = ''
            for element in json_line['entities']['hashtags']:
                ht_list += ' '+element['text']

            
            our_str = str(our_docid) + ' | ' + str(json_line['id']) + ' | ' + preprocessing(json_line['full_text'],word_dist) + ' | ' + str(json_line['created_at']) + ' | ' + ht_list + ' | ' + str(json_line['favorite_count']) + ' | ' + str(json_line['retweet_count']) + ' | ' + our_url
            list_of_tweets.append(our_str)

    # print(word_dist.keys())
    #print(len(list_of_tweets[1][2]))

    '''for i in range(10):
        print(list_of_tweets[i+30]+'\n')'''
    
    '''DE MOMENT FALTA FER LO DE tweet_document_ids_map QUE NO ENTENEM COM FER MAP (QUÈ SON ELS DOCUMENTS EN AQUEST CAS ?)'''

    #font_path = '/path/to/your/truetype/font.ttf'

    #WORDCLOUD
    wordcloud = WordCloud(width = 800, height = 800,
            background_color ='white').generate(' '.join(word_dist.keys()))
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig('./results/wordcloud.jpg')
    plt.close()


    #LENGTH DISTRIBUTION

    text_lengths = []

    for tweet in list_of_tweets:
        #get the second element -> tweet text
        full_text = tweet.split('|')[2].strip()
        words = full_text.split()
        # Append the word count to the word_counts array
        text_lengths.append(len(words))

    # Create a histogram plot
    plt.hist(text_lengths, bins=20, edgecolor='black')

    # Set labels and title
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Full Text Lengths')

    # Display the plot 
    plt.savefig('./results/full_length_distribution.jpg')
    plt.close()

    #VOCABULARY SIZE (unique words):

    total_count = 0

    for tweet in list_of_tweets:
        #get the second element -> tweet text
        full_text = tweet.split('|')[2].strip()
        words = full_text.split()
        total_count+= len(words)

    print('\nNUMBER OF UNIQUE WORDS:', len(word_dist.keys()))
    print('NUMBER OF TOTAL WORDS:', total_count)  

    #RANKING OF MOST RETWEETED TWEETS
    def get_retweet_count(tweet):
        tweet_parts = tweet.split('|')
        retweet_count = int(tweet_parts[6].strip())  # Assuming retweet_count is the second-to-last field
        return retweet_count

    # Sort the list of tweets based on retweet count in descending order (highest retweet count first)
    sorted_tweets = sorted(list_of_tweets, key=get_retweet_count, reverse=True)
    print('\n- Top 5 most retweeted tweets:')
    print(sorted_tweets[:5])

    #TEMPORAL ANALYSIS
    
    # temporal(list_of_tweets)


if __name__ == '__main__':
    main()

    
