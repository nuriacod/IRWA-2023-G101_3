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

    
    our_str = str(doc_id) + ' | ' + str(line['id']) + ' | ' + preprocessing(line['full_text'],word_dist, True)+ ' | ' + \
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

def create_inverted_index(list_of_tweets):

    inv_idx = defaultdict(list)
    

    for tweet in list_of_tweets:
        tweet_arr = tweet.split("|") # we get the fields of each tweet
        doc_id = tweet_arr[0] # doc_xxxx
        tweet_text = tweet_arr[2].split(" ")
        tweet_text = [word for word in tweet_text if word != ""]
        
        current_page_index = {}

        for position, term in enumerate(tweet_text):
            try:
                # if the term is already in the index for the current page (current_page_index)
                # append the position to the corresponding list

        ## START CODE
                current_page_index[term][1].append(position)
            except:
                # Add the new term as dict key and initialize the array of positions and add the position
                current_page_index[term]=[doc_id, array('I',[position])] #'I' indicates unsigned int (int in Python)

        #merge the current page index with the main index


        for term_page, posting_page in current_page_index.items():
            inv_idx[term_page].append(posting_page)

        ## END CODE

    return inv_idx

def create_index_tfidf(lines, num_documents):
    """
    Implement the inverted index and compute tf, df and idf

    Argument:
    lines -- collection of Wikipedia articles
    num_documents -- total number of documents

    Returns:
    index - the inverted index (implemented through a Python dictionary) containing terms as keys and the corresponding
    list of document these keys appears in (and the positions) as values.
    tf - normalized term frequency for each term in each document
    df - number of documents each term appear in
    idf - inverse document frequency of each term
    """

    index = defaultdict(list)
    tf = defaultdict(list)  # term frequencies of terms in documents (documents in the same order as in the main index)
    df = defaultdict(int)  # document frequencies of terms in the corpus
    idf = defaultdict(float)

    for tweet in lines:
        tweet_arr = tweet.split("|") # we get the fields of each tweet
        doc_id = tweet_arr[0] # doc_xxxx
        tweet_text = tweet_arr[2].split(" ")
        tweet_text = [word for word in tweet_text if word != ""]

        ## ===============================================================
        ## create the index for the *current page* and store it in current_page_index
        ## current_page_index ==> { ‘term1’: [current_doc, [list of positions]], ...,‘term_n’: [current_doc, [list of positions]]}

        ## Example: if the curr_doc has id 1 and his text is
        ##"web retrieval information retrieval":

        ## current_page_index ==> { ‘web’: [1, [0]], ‘retrieval’: [1, [1,4]], ‘information’: [1, [2]]}

        ## the term ‘web’ appears in document 1 in positions 0,
        ## the term ‘retrieval’ appears in document 1 in positions 1 and 4
        ## ===============================================================

        current_page_index = {}

        for position, term in enumerate(tweet_text):  ## terms contains page_title + page_text
            try:
                # if the term is already in the dict append the position to the corresponding list
                current_page_index[term][1].append(position)
            except:
                # Add the new term as dict key and initialize the array of positions and add the position
                current_page_index[term]=[doc_id, array('I',[position])] #'I' indicates unsigned int (int in Python)

        #normalize term frequencies
        # Compute the denominator to normalize term frequencies (formula 2 above)
        # norm is the same for all terms of a document.
        norm = 0
        for term, posting in current_page_index.items():
            # posting will contain the list of positions for current term in current document.
            # posting ==> [current_doc, [list of positions]]
            # you can use it to infer the frequency of current term.
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)

        # calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term, posting in current_page_index.items():
            # append the tf for current term (tf = term frequency in current doc/norm)
            tf[term].append(np.round(len(posting[1])/norm,4)) ## SEE formula (1) above
            #increment the document frequency of current term (number of documents containing the current term)
            df[term] += 1 # increment DF for current term

        #merge the current page index with the main index
        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)

        # Compute IDF following the formula (3) above. HINT: use np.log
        for term in df:
            idf[term] = np.round(np.log(float(num_documents/df[term])), 4)

    return index, tf, df, idf

def rank_documents(terms, docs, index, idf, tf):
    """
    Perform the ranking of the results of a search based on the tf-idf weights

    Argument:
    terms -- list of query terms
    docs -- list of documents, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies

    Returns:
    Print the list of ranked documents
    """

    # I'm interested only on the element of the docVector corresponding to the query terms
    # The remaining elements would became 0 when multiplied to the query_vector
    doc_vectors = defaultdict(lambda: [0] * len(terms))
    # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.
    # Example: collections.Counter(["hello","hello","world"]) --> Counter({'hello': 2, 'world': 1})
    # HINT: use when computing tf for query_vector

    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index:
            continue

        ## Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex]=query_terms_count[term]/query_norm * idf[term]

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index[term]):
            # Example of [doc_index, (doc, postings)]
            # 0 (26, array('I', [1, 4, 12, 15, 22, 28, 32, 43, 51, 68, 333, 337]))
            # 1 (33, array('I', [26, 33, 57, 71, 87, 104, 109]))
            # term is in doc 26 in positions 1,4, .....
            # term is in doc 33 in positions 26,33, .....

            #tf[term][0] will contain the tf of the term "term" in the doc 26
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]  # TODO: check if multiply for idf

    # Calculate the score of each doc
    # compute the cosine similarity between queyVector and each docVector:
    # HINT: you can use the dot product because in case of normalized vectors it corresponds to the cosine similarity
    # see np.dot

    doc_scores=[[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items() ]
    print('DOC SCORES', doc_scores)
    
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]

    if len(result_docs) == 0:
        print("No results found, try again")
        exit()
    #print ('\n'.join(result_docs), '\n')
    return result_docs



def search_tf_idf(query, inv_idx, idf, tf):
    """
    output is the list of documents that contain any of the query terms.
    So, we will get the list of documents for each query term, and take the union of them.
    """
    if 'and' in query: conj = 1
    else: conj = 0

    query = preprocessing(query, {}, False)
    docs = set()
    for term in query:
        try:
            # store in term_docs the ids of the docs that contain "term"
            term_docs=[posting[0] for posting in inv_idx[term]]

            if conj == 1:
                # intersection --> the documents must contain ALL the words in the query
                docs = docs & set(term_docs)
            elif conj == 0:
                # docs = docs Union term_docs
                docs |= set(term_docs)
        except:
            #term is not in index
            pass
    docs = list(docs)

    ranked_docs = rank_documents(query, docs, inv_idx, idf, tf)
    return ranked_docs

    
            
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
    # word_cloud(word_dist)

    #LENGTH DISTRIBUTION
    # text_length_distribution(list_of_tweets)
    
    #TEMPORAL ANALYSIS
    # temporal_plot(list_of_tweets)
    
    #VOCABULARY SIZE AND TOTAL SIZE:
    print('\nNUMBER OF UNIQUE WORDS:', len(word_dist.keys()))
    print('NUMBER OF TOTAL WORDS:', total_count)  

    #RANK MOST RETWEETED TWEETS
    sorted_tweets = sorted(list_of_tweets, key=get_retweet_count, reverse=True)
    print('\n- Top 5 most retweeted tweets:')
    print(sorted_tweets[:5])

    inverted = create_inverted_index(list_of_tweets)
    print("Number of words in 'inverted'",len(inverted.keys()))
    # print(inverted)
    # 
    # print("First 10 Index results for the term 'ukrain': \n{}".format(inverted['ukrain'][:10]))

    num_documents = len(list_of_tweets)
    global tf, idf
    tf_idf_index, tf, df, idf = create_index_tfidf(list_of_tweets, num_documents)

    query = 'russia war'
    ranked_docs = search_tf_idf(query, tf_idf_index, idf, tf)
    top = 10


    print("\n======================\nTop {} results out of {} for the searched query:\n".format(top, len(ranked_docs)))
    for d_id in ranked_docs[:top]:
        print("page_id= {}".format(d_id))



   

if __name__ == '__main__':
    main()

    
