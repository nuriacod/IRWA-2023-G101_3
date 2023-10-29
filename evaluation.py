import numpy as np 


def docids_for_evaluation(query, df):
    """
    Extract a list of document IDs for evaluation based on query and labels.

    Parameters
    ----------
    query : str
        The query identifier for which you want to extract document IDs.

    df : DataFrame
        The DataFrame containing doc id, query id and label.

    Returns
    -------
    list
        A list of document IDs that match for the specified query.
    """
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
    # number of Ground Truth Positives --> relevant documents
    gtp = np.sum(doc_score) 
    
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
    # loop over all query id
    for q in search_res['query_id'].unique():  
         # select data for current query
        curr_data = search_res[search_res['query_id']==q] 
        #append average precision for current query
        avp.append(avg_precision_at_k(np.array(curr_data['is_relevant']),np.array(curr_data['predicted_relevance']),k))  
    # return mean average precision
    return sum(avp)/len(avp),avp 

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
    """
    Calculate the Discounted Cumulative Gain (DCG) at a given rank (k) for a ranked list of documents.

    Parameters
    ----------
    doc_score : array-like, shape (n_documents,)
        Ground truth (true relevance labels) for the documents.

    y_score : array-like, shape (n_documents,)
        Predicted scores for the documents.

    k : int, optional, default: 10
        The rank at which to calculate DCG.

    Returns
    -------
    float
        The Discounted Cumulative Gain at rank k for the given document scores and predicted scores.
    """

    # get the list of indexes of the predicted score sorted in descending order.
    order = np.argsort(y_score)[::-1]  
    # sort the actual relevance label of the documents based on predicted score(hint: np.take) and take first k.
    doc_score = np.take(doc_score, order[:k]) 
    # Compute gain
    gain = 2** doc_score-1  
    # Compute denominator
    discounts = np.log2(np.arange(len(doc_score)) + 2)  
    #return dcg@k
    return np.sum(gain / discounts)  

def ndcg_at_k(doc_score, y_score, k=10):
    """
    Calculate the Normalized Discounted Cumulative Gain (nDCG) at a given rank (k) for a ranked list of documents.

    Parameters
    ----------
    doc_score : array-like, shape (n_documents,)
        Ground truth (true relevance labels) for the documents.

    y_score : array-like, shape (n_documents,)
        Predicted scores for the documents.

    k : int, optional, default: 10
        The rank at which to calculate nDCG.

    Returns
    -------
    float
        The Normalized Discounted Cumulative Gain at rank k for the given document scores and predicted scores.
    """
    # Ideal dcg
    dcg_max = dcg_at_k(doc_score, doc_score, k) 
    if not dcg_max:
        return 0
    # return ndcg@k
    return np.round(dcg_at_k(doc_score, y_score, k)/dcg_max, 4) 