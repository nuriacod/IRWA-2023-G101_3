
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


    
def tweet2Vec(list_of_tweets):
    """
    Generate tweet representations using Word2Vec word embeddings.

    Arguments:
    list_of_tweets (list): A list of strings representing individual tweets.

    Returns:
    np.ndarray: A 1D NumPy array representing the averaged vector for the entire collection of tweets.
    """
    
    tokenized_tweets = [word_tokenize(tw) for tw in list_of_tweets]
    
    # Create Word2Vec model
    model = Word2Vec(sentences=tokenized_tweets, vector_size=200, window=5, min_count=1, workers=4, epochs=60)
    tweet_vectors = []
    for tweet in list_of_tweets:
        tweet_arr = tweet.split("|") 

        tweet_text = tweet_arr[2]

        tokenized_tweet = word_tokenize(tweet_text)
        word_vectors = [model.wv[word] for word in tokenized_tweet]
        
        tweet_vector = np.mean(word_vectors, axis=0)
        
        tweet_vectors.append(tweet_vector)
        
    return tweet_vectors,model
    
def get_top_tweets(query,model, tweet_vectors,list_of_tweets, top_n=20):
    """
    Get top tweets based on cosine similarity with a query.

    Arguments:
    query (str): The query string.
    model: The Word2Vec model.
    tweet_vectors (list): A list of tweet vectors corresponding to each tweet.
    list_of_tweets (list): A list of strings representing individual tweets without preprocessing.
    top_n (int): Number of top tweets to retrieve.

    Returns:
    list: A list of top-ranked tweets based on cosine similarity.
    """

    # Tokenize the query
    tokenized_query = word_tokenize(query)

    # Generate vector for the query words exiting in the vocabulary
    query_vector = np.mean([model.wv[word] for word in tokenized_query if word in model.wv], axis=0)

    # Calculate cosine similarity between query vector and tweet vectors
    similarity_scores = cosine_similarity([query_vector], tweet_vectors)[0]

    # Rank tweets based on similarity scores
    ranked_tweets_indices = np.argsort(similarity_scores)[::-1][:top_n]

    # Retrieve top-ranked tweets
    top_tweets = [list_of_tweets[i] for i in ranked_tweets_indices]

    return top_tweets







