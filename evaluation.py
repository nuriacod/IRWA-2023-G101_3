def precision_at_k(doc_score, y_score, k=10):
    """
    Parameters
    ----------
    doc_score: Ground truth (true relevance labels).
    y_score: Predicted scores.
    k : number of doc to consider.

    Returns
    -------
    precision @k : float

    """
    order = np.argsort(y_score)[::-1] # [::-1] is the notation for "descending order"
    doc_score = np.take(doc_score, order[:k]) #y_true # we only consider the top k documents
    relevant = sum(doc_score==1) # get the number of documents that are relevant
    return float(relevant/k) # formula for the precision
