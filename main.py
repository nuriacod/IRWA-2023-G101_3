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

import nltk
nltk.download('stopwords')


def rep(m):
    """
    Description: Splits a hashtag into a list of words based on capitalization.
    """
    s=m.group(1)
    return ' '.join(re.split(r'(?=[A-Z])', s))


def preprocessing(text, word_dist):
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
    word_dist.update(text)
    
    # Perform stemming
    text = [stemmer.stem(word) for word in text]
    
    text = ' '.join(text)

    return text


def csv_to_dict(filepath):
    """
    Converts a tab-delimited CSV file into a dictionary with values from the second 
    column as keys and values from the first column as values
    """
    d = {}
    with open(filepath, 'r', newline='') as file:
        # Use tab as the delimiter
        reader = csv.reader(file, delimiter='\t')  
        for row in reader:
            if len(row) == 2:
                key, value = row
                d[str(value)] = key
    return d

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

    
    our_str = str(doc_id) + ' | ' + str(line['id']) + ' | ' + preprocessing(line['full_text'],word_dist)+ ' | ' + \
            str(line['created_at']) + ' | ' + ht_list + ' | ' + str(line['favorite_count']) + ' | ' + \
            str(line['retweet_count']) + ' | ' + our_url
            
    return our_str
    
def get_retweet_count(tweet):
    """
    Retrieves the retweet count from a formatted tweet string.
    """
    tweet_parts = tweet.split('|')
    # Given retweet_count is the second-to-last field
    retweet_count = int(tweet_parts[6].strip())  
    return retweet_count
    
def update_count(tweet,total_count):
    """
    Updates a running word count based on the full text of a tweet and returns the new total word count.
    """
    full_text = tweet.split('|')[2].strip()
    words = full_text.split()
    total_count+= len(words)
    return total_count



def temporal_plot(list_of_tweets):
    """
    Description: Generates a temporal plot of tweet activity over time based on a list of tweets. 
    The function parses the creation dates of tweets, counts the tweets per day, and 
    visualizes the temporal distribution in a line plot.
    """

    # Tweet counts per day
    tweet_counts = {}

    for tweet in list_of_tweets:
        # Get timeStamp field and convert to date type 
        created_at = tweet.split('|')[3].strip()
        created_at = datetime.datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')
        date = created_at.date()
        
        # Update the tweet count
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
    plt.savefig('./results/temporalPlot.jpg')
    plt.close()
    
def word_cloud(word_dist):
    """
    Creates a word cloud visualization based on a dictionary of word frequency distribution. 
    The word cloud displays words with sizes relative to their frequencies in the input data.
    """
    wordcloud = WordCloud(width = 800, height = 800,
            background_color ='white').generate(' '.join(word_dist.keys()))
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig('./results/wordcloud.jpg')
    plt.close()
    
def text_length_distribution(list_of_tweets):
    """
    Plots a histogram representing the distribution of text lengths in a list of tweets. 
    The function counts the number of words in each tweet and generates a histogram to 
    show the frequency of various text lengths.
    """
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

    # Save the plot 
    plt.savefig('./results/full_length_distribution.jpg')
    plt.close()
    
            
def main():
    docs_path = './IRWA_data_2023/Rus_Ukr_war_data.json'
    dict_path = './IRWA_data_2023/Rus_Ukr_war_data_ids.csv'
    
    # Initialize dictionary for word count
    word_dist = FreqDist()
    list_of_tweets= []
    docid_tweetid = csv_to_dict(dict_path)
    total_count = 0
    with open(docs_path) as fp:
        for i, line in enumerate(fp):
            json_line = json.loads(line)
            our_docid = docid_tweetid[str(json_line['id'])]
            our_str = get_fields(json_line,our_docid,word_dist)
            total_count = update_count(our_str,total_count)
            list_of_tweets.append(our_str)



    #WORDCLOUD
    word_cloud(word_dist)

    #LENGTH DISTRIBUTION
    text_length_distribution(list_of_tweets)
    
    #TEMPORAL ANALYSIS
    temporal_plot(list_of_tweets)
    
    #VOCABULARY SIZE AND TOTAL SIZE:
    print('\nNUMBER OF UNIQUE WORDS:', len(word_dist.keys()))
    print('NUMBER OF TOTAL WORDS:', total_count)  

    #RANK MOST RETWEETED TWEETS
    sorted_tweets = sorted(list_of_tweets, key=get_retweet_count, reverse=True)
    print('\n- Top 5 most retweeted tweets:')
    print(sorted_tweets[:5])

   

if __name__ == '__main__':
    main()

    
