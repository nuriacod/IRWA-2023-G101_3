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


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE


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
    
    text = line['full_text']
    return our_str, text
    
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
    # Term frequencies of terms in documents (documents in the same order as in the main index)
    tf = defaultdict(list)  
    # Document frequencies of terms in the corpus
    df = defaultdict(int)  
    idf = defaultdict(float)

    for tweet in lines:
        # we get the fields of each tweet
        tweet_arr = tweet.split("|") 
        # doc_xxxx
        doc_id = tweet_arr[0]
        tweet_text = tweet_arr[2].split(" ")
        tweet_text = [word for word in tweet_text if word != ""]

        ## create the index for the *current page* and store it in current_page_index


        current_page_index = {}

        for position, term in enumerate(tweet_text):  
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
    # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically 
    # added to the dictionary
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
        #print("index term",index[term])
        #print("DOCS",docs)
        for doc_index, (doc, postings) in enumerate(index[term]):
            #print("doc",doc,"in",docs)
            #print(doc)
            if doc.strip() in docs:
                #print("HELOOOOO",doc)
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]  
                # TODO: check if multiply for idf
                #print("doc loop",doc_vectors[doc][termIndex])
    # Calculate the score of each doc
    # compute the cosine similarity between queyVector and each docVector:
    # HINT: you can use the dot product because in case of normalized vectors it corresponds to the cosine similarity
    # see np.dot

    doc_scores=[[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items() ]
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]

    if len(result_docs) == 0:
        print("No results found, try again")
        exit()
    #print ('\n'.join(result_docs), '\n')
    return result_docs

def search_tf_idf(query, inv_idx):
    
    """
    output is the list of documents that contain any of the query terms.
    So, we will get the list of documents for each query term, and take the union of them.
    For conjunctive queries (containing the keyword 'AND') we take the intesection so that all words in the query
    appear in the retrieved documents.
    """

    if 'and' in query: conj = 1
    else: conj = 0

    query = preprocessing(query, {}, False)
    query=query.split()
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

def subset_search_tf_idf(query, inv_idx, subset):
    """
    Works in the same fashion as search_tf_idf, but only taking into account the subset of tagged documents for evaluation.
    """

    if 'and' in query: conj = 1
    else: conj = 0

    query = preprocessing(query, {}, False)
    query=query.split()
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
                docs = docs.union(set(term_docs))
        except:
            #term is not in index
            pass

    docs = [element.strip() for element in list(docs)]
    docs = set(docs).intersection(set(subset)) # only keep those documents that are in the input subset

    docs = list(docs)

    #print("docs",docs)

    ranked_docs = rank_documents(query, docs, inv_idx, idf, tf)
    return ranked_docs

def create_reverse_mapping(forward_mapping):
    reverse_mapping = {value: key for key, value in forward_mapping.items()}
    return reverse_mapping

def print_query(doc_id):
    id = doc_id.replace('doc_', '')
    text = tweet_fulltext[int(id)]
    print("{}:{}".format(id,text))
    return id,text
    #print(doc_id, '=>', text, '\n')


def docids_for_evaluation(query, df):
    cond_1 = df['our_query_id'] == query
    cond_2 = df['label'] == 1
    query_doc_ids = df[(cond_1) | (cond_2)]
    query_doc_ids_list = query_doc_ids['doc'].tolist()
        

    return query_doc_ids_list

def avg_precision_at_k(doc_score, y_score, k=10):
    """
    Parameters
    ----------
    doc_score: Ground truth (true relevance labels).
    y_score: Predicted scores.
    k : number of doc to consider.

    Returns
    -------
    average precision @k : float
    """
    gtp = np.sum(doc_score) # number of Ground Truth Positives --> relevant documents
    #order = np.argsort(y_score)[::-1]
    #doc_score = np.take(doc_score, order[:k])
    
    ## if all documents are not relevant
    if gtp == 0:
        return 0
    n_relevant_at_i = 0
    prec_at_i = 0

    for i in range(len(doc_score)):
        if doc_score[i] == 1:
        
            n_relevant_at_i += 1
            prec_at_i += n_relevant_at_i/(i+1)
    return prec_at_i/gtp

def map_at_k(search_res, k=10):
    """
    Parameters
    ----------
    search_res: search results dataset containing:
        query_id: query id.
        doc_id: document id.
        predicted_relevance: relevance predicted through LightGBM.
        doc_score: actual score of the document for the query (ground truth).

    Returns
    -------
    mean average precision @ k : float
    """
    avp = []
    for q in search_res['query_id'].unique():  # loop over all query id
        curr_data = search_res[search_res['query_id']==q]  # select data for current query
        avp.append(avg_precision_at_k(np.array(curr_data['is_relevant']),np.array(curr_data['predicted_relevance']),k))  #append average precision for current query
    return sum(avp)/len(avp),avp # return mean average precision

def rr_at_k(doc_score, y_score, k=10):
    """
    Parameters
    ----------
    doc_score: Ground truth (true relevance labels).
    y_score: Predicted scores.
    k : number of doc to consider.

    Returns
    -------
    Reciprocal Rank for qurrent query
    """

    #order = np.argsort(y_score)[::-1]  # get the list of indexes of the predicted score sorted in descending order.
    #doc_score = np.take(doc_score,order[:k]) # sort the actual relevance label of the documents based on predicted score(hint: np.take) and take first k.
    if sum(doc_score) == 0:  # if there are not relevant doument return 0
        return 0
    return 1/(np.argmax(doc_score==1)+1)  # hint: to get the position of the first relevant document use "np.argmax"

def dcg_at_k(doc_score, y_score, k=10):
    order = np.argsort(y_score)[::-1]  # get the list of indexes of the predicted score sorted in descending order.
    doc_score = np.take(doc_score, order[:k])  # sort the actual relevance label of the documents based on predicted score(hint: np.take) and take first k.
    gain = 2** doc_score-1  # Compute gain (use formula 7 above)
    discounts = np.log2(np.arange(len(doc_score)) + 2)  # Compute denominator
    return np.sum(gain / discounts)  #return dcg@k

def ndcg_at_k(doc_score, y_score, k=10):
    dcg_max = dcg_at_k(doc_score, doc_score, k) # Ideal dcg
    if not dcg_max:
        return 0
    return np.round(dcg_at_k(doc_score, y_score, k)/dcg_max, 4) # return ndcg@k
            
def main():
    docs_path = './IRWA_data_2023/Rus_Ukr_war_data.json'
    dict_path = './IRWA_data_2023/Rus_Ukr_war_data_ids.csv'
    evaluation_path = './IRWA_data_2023/Evaluation_gt.csv'
    our_query_path = './IRWA_data_2023/our_query_labels.csv'
    query_map_path = './IRWA_data_2023/queryId_queryText.csv'

    eval_df = pd.read_csv(evaluation_path)
    our_query_df = pd.read_csv(our_query_path)
    query_map = csv_to_dict(query_map_path)
    query_map = {v: k for k, v in query_map.items()}
    
    # Initialize dictionary for word count
    word_dist = FreqDist()
    global list_of_tweets, tweet_fulltext
    list_of_tweets = []
    tweet_fulltext = []

    tweet_to_doc = csv_to_dict(dict_path) # key = tweet_id --> value = doc_xxxx
    total_count = 0
    with open(docs_path) as fp:
        for i, line in enumerate(fp):
            json_line = json.loads(line)
            our_docid = tweet_to_doc[str(json_line['id'])]
            our_str, aux_text = get_fields(json_line,our_docid,word_dist)
            total_count = update_count(our_str,total_count)
            list_of_tweets.append(our_str)
            tweet_fulltext.append(aux_text)

    
    doc_to_tweet = create_reverse_mapping(tweet_to_doc) # key = doc_xxxx --> value = tweet_id

    #WORDCLOUD
    # word_cloud(word_dist)

    #LENGTH DISTRIBUTION
    # text_length_distribution(list_of_tweets)
    
    #TEMPORAL ANALYSIS
    # temporal_plot(list_of_tweets)
    
    #VOCABULARY SIZE AND TOTAL SIZE:
    #print('\nNUMBER OF UNIQUE WORDS:', len(word_dist.keys()))
    #print('NUMBER OF TOTAL WORDS:', total_count)  

    #RANK MOST RETWEETED TWEETS
    #sorted_tweets = sorted(list_of_tweets, key=get_retweet_count, reverse=True)
    #print('\n- Top 5 most retweeted tweets:')
    #print(sorted_tweets[:5])   

    inverted = create_inverted_index(list_of_tweets)
    print("Number of words in 'inverted'",len(inverted.keys()))
    # print(inverted)
    # 
    top = 10
    term = 'ukrain'
    print("\n======================\nFirst {} Index results for the term 'ukrain':".format(top))
    # inverted[term][:top]
    print('document_id ==> positions in the document')
    for i in range(top):
        print("{} ==> {}".format(inverted[term][i][0], inverted[term][i][1].tolist()))
   
    
    print('\nQUERY MAP ---->', query_map)
    num_documents = len(list_of_tweets)
    global tf, idf, df, tf_idf_index
    print('\nCreating tf-idf index...')
    tf_idf_index, tf, df, idf = create_index_tfidf(list_of_tweets, num_documents)
    
    '''
    query = 'russian war dictator'   
    print('\nSearching for relevant documents...')
    ranked_docs = search_tf_idf(query, tf_idf_index)
    top = 10
    
    print("\n======================\nTop {} results out of {} for the searched query '{}':\n".format(top, len(ranked_docs), query))
    for d_id in ranked_docs[:top]:
    #print("page_id = {}".format(d_id))
        print_query(d_id)

    '''
    
    averages = []
    rr = []
    #iterate over queries in our_query_df for evaluation
    for query in our_query_df.our_query_id.unique():
        print('\n***** Searching docs for {}... *****'.format(query))

        # creating the subset of documents that we have tagged as relevant (or not) in the csv file
        q_ids = docids_for_evaluation(query, our_query_df) 
        q_ranking = subset_search_tf_idf(query_map[query], tf_idf_index, q_ids)
        # Choose a number of documents: 
        top = 10

        print("======================\nTop {} results out of {} for the searched query '{}':".format(top, len(q_ranking), query_map[query]))
        doc_ids =[]
        map_doc_ids = {}
        for i, d_id in enumerate(q_ranking[:top]):
            print_query(d_id)
            doc_ids.append(d_id)
            map_doc_ids[d_id.strip()] = i+1
        print('\n',map_doc_ids)
        doc_ids = [s.strip() for s in doc_ids]
        
        query_df = our_query_df.copy()
        
        query_df['predicted'] = query_df.apply(lambda row: 1 if row['doc'] in doc_ids else 0, axis=1)
        query_df['order'] = query_df.apply(lambda row: map_doc_ids[row['doc']] if row['doc'] in doc_ids else 0, axis=1)
        
        #print(query_df[query_df["our_query_id"] == query])
        # Create new dataframe that only contains the rows for the current query  
        #filtered_df = query_df[query_df['our_query_id'] == query]
        #print(filtered_df)

        #query_x_df = query_df[query_df['doc'].isin(q_ids)]

        # print('query_x_df\n', query_x_df)
        
        # Create new dataframe that only contains the rows for the current query QX 
        #filtered_df = query_df[query_df['our_query_id'] == query]
        
        adapted_df = query_df.copy()

        for index, row in adapted_df.iterrows():
            if row['our_query_id'] != query:
                adapted_df.at[index, 'label'] = 0

        # PRECISION (P)
        precision_at_k = adapted_df[adapted_df ['predicted'] == 1]['label'].sum()/top
        print("\n* Precision of query {} is: {}".format(query,precision_at_k))

        # RECALL (R)
        TP = adapted_df[adapted_df ['predicted'] == 1]['label'].sum()
        FN = adapted_df[(adapted_df['label'] == 1) & (adapted_df['predicted'] == 0)]['label'].count()
        
        #relevant_size = query_df[query_df['predicted'] == 1]['label'].sum()
        #recall_at_k = filtered_df[filtered_df['predicted'] == 1]['label'].sum()/relevant_size # ??
        
        recall_at_k = TP/(TP+FN)
        print("* Recall of query {} is: {}".format(query,recall_at_k))
        
        # AVERAGE PRECISION (AP)
        # modify dataframe
        # 1. all labels that are not from the current query -> set to 0
        # 2. order by column order

        # adapted_df['label'] = 0 if row['our_query_id'] != query else row['label']

        '''print('number of non-zero labels:', adapted_df['label'].sum() )
        print('number of non-zero "predicted" rows:', adapted_df['predicted'].sum() )'''

        # adapted_df['label'] = adapted_df.apply(lambda row: 0 if row['our_query_id'] != query else row['label'])
        sorted_adapted_df = adapted_df.sort_values(by='order')


        ordered_df = sorted_adapted_df.copy()
        ordered_df = ordered_df[ordered_df['order'] != 0]

        average_precision_at_k = avg_precision_at_k(ordered_df['label'].tolist(),ordered_df['predicted'].tolist(), k=10)
        if query not in ['Q1', 'Q2', 'Q3']:
            averages.append(average_precision_at_k)
        
    
        print("Average precision of query {} is: {}".format(query, average_precision_at_k))

        # F1 SCORE
        f1_score = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
        print("F1 score of query {} is: {}".format(query,f1_score))
        
        # RECIPROCAL RANK (later we will compute the avg for all queries to obtain the MRR)
        ground_truth = ordered_df['label'].tolist()
        pred_scores = ordered_df['predicted'].tolist()

        rec_rank = rr_at_k(ground_truth, pred_scores, k=10)
        if query not in ['Q1', 'Q2', 'Q3']:
            rr.append(rec_rank)

        # NORMALIZED DISCOUNTED CUMULATIVE GAIN 
        ndcg_atk = np.round(ndcg_at_k(sorted_adapted_df['label'], sorted_adapted_df['predicted'], k=10), 4)
        print('Normalized Discounted Cumulative Gain:', ndcg_atk )

    # MEAN AVERAGE PRECISION (mAP)
    #is outside the loop because it is a metric for all the queries
    mAP  = sum(averages)/len(averages)
    print("Mean average precision for all queries is:", mAP)

    # MEAN RECIPROCAL RANK (MRR)
    # Compute the average of the RR for all the queries
    mrr = sum(rr)/len(rr)
    print("Mean Reciprocal Rank for all queries:", mrr)
    
    ## 2 DIMENSIONAL REPRESENTATION
    # Create a TF-IDF vectorizer

    doc_ids = our_query_df['doc'].tolist() # doc ids from evaluation subset
    processed_texts = []

    for t in list_of_tweets:
        if t.split('|')[0].strip() in doc_ids:
            text = t.split('|')[2].strip()
            processed_texts.append(text)


    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)

    # Apply t-SNE to the TF-IDF matrix
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(tfidf_matrix.toarray())

    # Plot the t-SNE representation
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.show()

    

if __name__ == '__main__':
    main()


    