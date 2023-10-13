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


import nltk
nltk.download('stopwords')



def preprocessing(text, word_dist):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    
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
    our_punct = string.punctuation.replace('#', '')

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
      
    word_dist.update(text)

    # Eliminate the stopwords
    text = [word for word in text if not word in stop_words] 

    # Perform stemming
    text = [stemmer.stem(word) for word in text]

    text = ' '.join(text)

    return text


def main():
    docs_path = './IRWA_data_2023/Rus_Ukr_war_data.json'
    # Initialize dictionary for word count
    word_dist = FreqDist()
    list_of_tweets= []
    docid_tweetid = {}
    with open(docs_path) as fp:
        for i, line in enumerate(fp):
            # Transform each line of the json file into a python dictionary
            json_line = json.loads(line)
            docid_tweetid[json_line['id']] = i

            our_url = ''
            if len(json_line['entities']["urls"]) > 0:
                our_url = json_line['entities']['urls'][0]['expanded_url']

            ht_list = ''
            for element in json_line['entities']['hashtags']:
                ht_list += ' '+element['text']

            
            our_str = str(i) + ' | ' + preprocessing(json_line['full_text'],word_dist) + ' | ' + str(json_line['created_at']) + ' | ' + ht_list + ' | ' + str(json_line['favorite_count']) + ' | ' + str(json_line['retweet_count']) + ' | ' + our_url
            list_of_tweets.append(our_str)

    print(word_dist.keys())
    '''for i in range(10):
        print(list_of_tweets[i+30]+'\n')'''
    
    '''DE MOMENT FALTA FER LO DE tweet_document_ids_map QUE NO ENTENEM COM FER MAP (QUÈ SON ELS DOCUMENTS EN AQUEST CAS ?)'''

    #font_path = '/path/to/your/truetype/font.ttf'

    wordcloud = WordCloud(width = 800, height = 800,
            background_color ='white').generate(' '.join(word_dist.keys()))
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()

if __name__ == '__main__':
    main()

    #%%  
    from wordcloud import WordCloud, STOPWORDS

    wordcloud = WordCloud(width = 800, height = 800,
    background_color ='white',
    stopwords = stopwords,
    min_font_size = 10).generate(list_of_tweets)

    
