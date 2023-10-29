from nltk import FreqDist
from csv import reader
import matplotlib.pyplot as plt
import pandas as pd
#import tkinter as Tk
import numpy as np
import json
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

from evaluation import avg_precision_at_k, map_at_k, rr_at_k, dcg_at_k, ndcg_at_k, docids_for_evaluation
from modeling_indexing import create_inverted_index, create_index_tfidf, rank_documents, search_tf_idf,subset_search_tf_idf
from preprocessing import preprocessing, get_fields
from plots_data_analytics import temporal_plot, word_cloud, text_length_distribution




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

 

def print_query(doc_id):
    """
    Given a doc id returns the id as an int an the corresponding doc text (tweet)
    """
    
    id = int(doc_id.replace('doc_', ''))
    text = tweet_fulltext[id]
    print("{}:{}".format(id,text))
    return id,text

            
def main():
    docs_path = './IRWA_data_2023/Rus_Ukr_war_data.json'
    dict_path = './IRWA_data_2023/Rus_Ukr_war_data_ids.csv'
    our_query_path = './IRWA_data_2023/our_query_labels.csv'
    query_map_path = './IRWA_data_2023/queryId_queryText.csv'

    our_query_df = pd.read_csv(our_query_path)
    query_map = csv_to_dict(query_map_path)
    query_map = {v: k for k, v in query_map.items()}
    
    # Initialize dictionary for word count
    word_dist = FreqDist()
    global list_of_tweets, tweet_fulltext
    list_of_tweets = []
    tweet_fulltext = []

    # key = tweet_id --> value = doc_xxxx
    tweet_to_doc = csv_to_dict(dict_path) 
    total_count = 0
    with open(docs_path) as fp:
        for i, line in enumerate(fp):
            json_line = json.loads(line)
            our_docid = tweet_to_doc[str(json_line['id'])]
            our_str, aux_text = get_fields(json_line,our_docid,word_dist)
            total_count = update_count(our_str,total_count)
            list_of_tweets.append(our_str)
            tweet_fulltext.append(aux_text)

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

    # Build inverted index
    inverted = create_inverted_index(list_of_tweets)
    print("Number of words in 'inverted'",len(inverted.keys()))

    top = 10
    term = 'ukrain'
    print("\n======================\nFirst {} Index results for the term 'ukrain':".format(top))
    print('document_id ==> positions in the document')
    for i in range(top):
        print("{} ==> {}".format(inverted[term][i][0], inverted[term][i][1].tolist()))
   
    
    print('\nQUERY MAP ---->', query_map)
    
    # Build  tfidf index
    num_documents = len(list_of_tweets)
    global tf, idf, df, tf_idf_index
    print('\nCreating tf-idf index...')
    tf_idf_index, tf, df, idf = create_index_tfidf(list_of_tweets, num_documents)
    
    
    averages = []
    rr = []
    list_of_list_tweets = []
    #iterate over queries in our_query_df for evaluation
    for query in our_query_df.our_query_id.unique():
        print('\n***** Searching docs for {}... *****'.format(query))

        # creating the subset of documents that we have tagged as relevant (or not) in the csv file
        q_ids = docids_for_evaluation(query, our_query_df) 
        q_ranking = subset_search_tf_idf(query_map[query], tf_idf_index, q_ids,idf,tf)
        # Chosen number of documents: 
        top = 10

        print("======================\nTop {} results out of {} for the searched query '{}':".format(top, len(q_ranking), query_map[query]))
        doc_ids =[]
        map_doc_ids = {}
        list_of_tweets = []
        # Iterate over the top 10 mos relevant documents for each query
        for i, d_id in enumerate(q_ranking[:top]):
            _,tweet = print_query(d_id)
            list_of_tweets.append(tweet)
            doc_ids.append(d_id)
            map_doc_ids[d_id.strip()] = i+1

        # List of tweets for each query for the 2D representation
        list_of_list_tweets.append(list_of_tweets)
        doc_ids = [s.strip() for s in doc_ids]
        
        query_df = our_query_df.copy()
        
        # Add to the dataframe with the doc id, query id and label a column for predicted set to 1 if predicted else to 0 and 
        # and also another column to add in which position tweets have been ranked
        query_df['predicted'] = query_df.apply(lambda row: 1 if row['doc'] in doc_ids else 0, axis=1)
        query_df['order'] = query_df.apply(lambda row: map_doc_ids[row['doc']] if row['doc'] in doc_ids else 0, axis=1)
    
        
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
        
        
        recall_at_k = TP/(TP+FN)
        print("* Recall of query {} is: {}".format(query,recall_at_k))
        
        # AVERAGE PRECISION (AP)
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
    # outside the loop because it is a metric for all the queries
    mAP  = sum(averages)/len(averages)
    print("Mean average precision for all queries is:", mAP)

    # MEAN RECIPROCAL RANK (MRR)
    # Compute the average of the RR for all the queries
    mrr = sum(rr)/len(rr)
    print("Mean Reciprocal Rank for all queries:", mrr)
    
    ## 2 DIMENSIONAL REPRESENTATION
    colors = ['#ABEBC6', '#AED6F1', '#EC7063', '#CCD1D1', '#D7BDE1', '#EB984E', '#52BE80', '#F7DC6F']
    
    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Initialize a list to store the t-SNE results for each list of tweets
    tsne_results = []

    # Apply t-SNE to each list of tweets and store the results
    for tweets in list_of_list_tweets:
        tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(tfidf_matrix.toarray())
        tsne_results.append(X_tsne)

    # Plot the t-SNE representations with different colors for each list
    _, ax = plt.subplots()
    for i, tsne_result in enumerate(tsne_results):
        ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors[i], label=f'Q{i + 1}')

    # Add a legend to distinguish the lists
    ax.legend()

    # Save the plot as an image (e.g., PNG)
    plt.savefig("./results/tsne_plot.png")

    # Close the plot to free up memory (optional)
    plt.close()

    

if __name__ == '__main__':
    main()


    