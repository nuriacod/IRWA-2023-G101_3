import array
from collections import defaultdict
import math
import numpy as np
from preprocessing import preprocessing
from numpy import linalg as la
import string
from array import array
import collections
from sklearn.preprocessing import StandardScaler
import collections



def create_inverted_index(list_of_tweets):
    """
    Create an inverted index for a list of tweets.

    Parameters
    ----------
    list_of_tweets : list
        A list of tweets, where each tweet is represented as a string

    Returns
    -------
    inv_idx : defaultdict
        The inverted index (implemented through a Python dictionary) containing terms as keys and the corresponding
        list of documents where these keys appears in (and the positions) as values.
    """

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
    our_score - dictionary with the score we generated for each doc
    """
    index = defaultdict(list)
    # Term frequencies of terms in documents (documents in the same order as in the main index)
    tf = defaultdict(list)  
    # Document frequencies of terms in the corpus
    df = defaultdict(int)  
    idf = defaultdict(float)
    our_score = defaultdict(float)

        
    for tweet in lines:
        # we get the fields of each tweet
        tweet_arr = tweet.split("|") 
        # doc_xxxx
        doc_id = tweet_arr[0]
        tweet_text = tweet_arr[2].split(" ")
        tweet_text = [word for word in tweet_text if word != ""]
        tweet_likes = int(tweet_arr[5])
        tweet_rts = int(tweet_arr[6])
        follower_count = int(tweet_arr[8])
        verified = tweet_arr[9] == 'True'

        #Compute the score for the current doc based on tweet likes, retweets, etc.
        our_score[doc_id] = (tweet_likes+ tweet_rts)*0.3 + (follower_count+ verified)*0.2

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
            # ret

    return index, tf, df, idf, our_score


def rank_documents(terms, docs, index, idf, tf, our_score):
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

    query_vector = [0] * len(terms)

    # get the frequency of each term in the query.
    query_terms_count = collections.Counter(terms)  
    
    # compute the norm for the query tf
    query_norm = la.norm(list(query_terms_count.values()))
    


    for termIndex, term in enumerate(terms):  
        #termIndex is the index of the term in the query
        if term not in index:
            continue
        ## Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex]=query_terms_count[term]/query_norm * idf[term]

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index[term]):
            if doc.strip() in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]
 
    # Calculate the score of each doc(cosine similarity between queyVector and each docVector)
    doc_scores=[[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items() ]
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]
    #print('************** TF-IDF SCORES', doc_scores)

    #Normalize each of the values in the sum:
    min_dot_product = min([np.dot(curDocVec, query_vector) for curDocVec in doc_vectors.values()])
    max_dot_product = max([np.dot(curDocVec, query_vector) for curDocVec in doc_vectors.values()])

    percentile_to_consider = 90
    min_our_score = min([our_score[doc] for doc in our_score.keys()])
    max_our_score = np.percentile([our_score[doc] for doc in our_score.keys()], percentile_to_consider)
    
    #MIN-MAX NORMALIZATION
    our_doc_scores=[[0.6 * ((np.dot(curDocVec, query_vector) - min_dot_product) / (max_dot_product - min_dot_product)) +0.4 * np.clip(((our_score[doc] - min_our_score) / (max_our_score - min_our_score)), 0, 1),doc] for doc, curDocVec in doc_vectors.items() ]
    
    our_doc_scores.sort(reverse=True)
    #print('************** OUR SCORE', our_doc_scores)
    our_result_docs = [x[1] for x in our_doc_scores]

    if len(result_docs) == 0:
        print("No results found with tf-idf, try again")

    if len(our_result_docs) == 0:
        print("No results found with our scores, try again")
        exit()
    #print ('\n'.join(result_docs), '\n')
    return result_docs, our_result_docs

def search_tf_idf(query, inv_idx,idf,tf):
    
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
            if conj == 0:
                # docs = docs Union term_docs
                docs |= set(term_docs)
        except:
            #term is not in index
            pass
    docs = list(docs)
    ranked_docs = rank_documents(query, docs, inv_idx, idf, tf)
    return ranked_docs

def subset_search_tf_idf(query, inv_idx, subset,idf,tf,our_score):
    """
    Works in the same fashion as search_tf_idf, but only taking into account the subset of tagged documents for evaluation.
    """

    if 'and' in query: conj = 1
    else: conj = 0

    query = preprocessing(query, {}, False)
    query=query.split()
    docs = set()
    count = 0
    for term in query:
        try:
            # store in term_docs the ids of the docs that contain "term"

            term_docs=[posting[0] for posting in inv_idx[term]]

            if conj == 1:
                # intersection --> the documents must contain ALL the words in the query
                if count== 0:
                    docs = set(term_docs)
                    count = 1
                else:
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

    ranked_docs, our_rank_docs = rank_documents(query, docs, inv_idx, idf, tf, our_score)
    # our_ranked_docs = our_rank_documents() --> ranking de documents amb la nostra score
    return ranked_docs, our_rank_docs