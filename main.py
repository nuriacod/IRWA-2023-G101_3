from collections import defaultdict
from array import array
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import numpy as np
import collections
from numpy import linalg as la
import json
import string

import nltk
nltk.download('stopwords')

def preprocessing(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    
    # Transform in lowercase
    text = text.lower()

    #remove punctuation
    tablePunt = str.maketrans("","",string.punctuation) 
    text = text.translate(tablePunt)
    
    text= text.split() 
    # Tokenize the text to get a list of terms
    
    text= [word for word in text if not word in stop_words] 
    # Eliminate the stopwords (HINT: use List Comprehension)
    text= [stemmer.stem(word) for word in text]

    return text


def main():
    docs_path = '.\IRWA_data_2023\Rus_Ukr_war_data.json' # canviar \ per / en mac i linux
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

            
            our_str = str(i) + ' | ' + json_line['full_text'] + ' | ' + str(json_line['created_at']) + ' | ' + ht_list + ' | ' + str(json_line['favorite_count']) + ' | ' + str(json_line['retweet_count']) + ' | ' + our_url
            list_of_tweets.append(our_str)

    for i in range(10):
        print(list_of_tweets[i+10]+'\n')

             

    

if __name__ == '__main__':
    main()

