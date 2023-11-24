# from modeling_indexing import create_inverted_index, create_index_tfidf, rank_documents, search_tf_idf,subset_search_tf_idf
from collections import defaultdict 
from array import array
import math 
import numpy as np
from myapp.search.load_corpus import preprocessing
from myapp.search.objects import ResultItem, Document
import collections
from numpy import linalg as la
import pickle


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
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]
 
    # Calculate the score of each doc(cosine similarity between queyVector and each docVector)
    doc_scores=[[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items() ]
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]
    #print('************** TF-IDF SCORES', doc_scores)

    if len(result_docs) == 0:
        print("No results found with tf-idf, try again")

    #print ('\n'.join(result_docs), '\n')
    return result_docs

def search_in_corpus(query, corpus, search_id):
    # 1. create create_tfidf_index
    

    # 2. apply ranking


    res = []
    size = len(corpus)
    ll = list(corpus.values())
    print('LL ---->',ll)

    index = defaultdict(list)
    # Term frequencies of terms in documents (documents in the same order as in the main index)
    tf = defaultdict(list)  
    # Document frequencies of terms in the corpus
    df = defaultdict(int)  
    idf = defaultdict(float)
    print('creating tfidf ....')
    for document in ll: 
        tweet_text = preprocessing(document.description,{},False)
        tweet_text = tweet_text.split(' ')
        tweet_text = [word for word in tweet_text if word != ""]   ## 
        # print(tweet_text)
        
        # doc_xxxx
        doc_id = document.id

        #Compute the score for the current doc based on tweet likes, retweets, etc.
        #our_score[doc_id] = (tweet_likes+ tweet_rts)*0.3 + (follower_count+ verified)*0.2

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
            idf[term] = np.round(np.log(float(size/df[term])), 4)
            # ret

    with open('index.pkl', 'wb') as fp:
        pickle.dump(idf,, fp)


    if 'and' in query: conj = 1
    else: conj = 0

    query = preprocessing(query, {}, False)
    query=query.split()
    print('QUERY -->', query)
    docs = set()
    for term in query:
        try:
            # store in term_docs the ids of the docs that contain "term"
            term_docs=[posting[0] for posting in index[term]]

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
    ranked_docs = rank_documents(query, docs, index, idf, tf)
    print(len(ranked_docs))
    print('RANKED DOCS ----->',ranked_docs[0])
    
    for idx, doc in enumerate(ranked_docs):
        item = corpus[doc]
        res.append(ResultItem(item.id, item.title, item.description, item.doc_date,
                            "doc_details?id={}&search_id={}&param2=2".format(item.id, search_id), idx))
                        

    return res