from modeling_indexing import create_inverted_index, create_index_tfidf, rank_documents, search_tf_idf,subset_search_tf_idf

def search_in_corpus(query, list_of_tweets):
    # 1. create create_tfidf_index
    num_documents = len(list_of_tweets)

    print('\nCreating tf-idf index...')
    tf_idf_index, tf, df, idf, our_score = create_index_tfidf(list_of_tweets, num_documents)
    

    # 2. apply ranking
    return ""


