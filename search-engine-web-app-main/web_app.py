import json
import os
from json import JSONEncoder
import csv
import time
import pickle
import random
import itertools 

# pip install httpagentparser
import httpagentparser  # for getting the user agent as json
import nltk
from flask import Flask, render_template, session
from flask import request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc, save_dicts,file_exists
from myapp.search.load_corpus import load_corpus, create_list_of_tweets
from myapp.search.objects import Document, StatsDocument, Session, Click, RequestData
from myapp.search.search_engine import SearchEngine



# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default

# end lines ***for using method to_json in objects ***

# instantiate the Flask application
app = Flask(__name__)



sessions = {}
clicks = []
requests_data = []

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = 'afgsreg86sr897b6st8b76va8er76fcs6g8d7'
# open browser dev tool to see the cookies
app.session_cookie_name = 'IRWA_SEARCH_ENGINE'

# instantiate our search engine
search_engine = SearchEngine()

# instantiate our in memory persistence
analytics_data = AnalyticsData()

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

file_path = "./Rus_Ukr_war_data.json"
dict_path = './Rus_Ukr_war_data_ids.csv'


corpus = load_corpus(file_path)

print('CORPUS IS LOADED')
global tf, idf, df, tf_idf_index, our_score, list_of_tweets
tweet_to_doc = csv_to_dict(dict_path) 

list_of_tweets = create_list_of_tweets(file_path, tweet_to_doc)
# print(list_of_tweets[0])

@app.before_request
def before_request():
    if 'session_index' not in session:
        session['session_index'] = str(random.randint(0,100000))

# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    #Collect HTTP Requests data
    request_data = RequestData(
        session_id=session.get('session_index'),  # Assuming you have a session_id in the session
        method=request.method,
        path=request.path,
        user_agent=request.headers.get('User-Agent'),
        # Add more relevant data
    )


    # Store request_data in-memory or database
    requests_data.append(request_data)
    
    ### Guardar request data 

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests
    session['some_var'] = "IRWA 2021 home"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))

    #print('agent--->', agent['browser'])

    #### GUARDAR SESSION DATA
    analytics_data.save_browser(agent['browser']['name'])

    print('browser_stats --->', analytics_data.fact_browser)


    return render_template('index.html', page_title="Welcome",session = agent)


@app.route('/search', methods=['GET', 'POST'], endpoint='search_results')
def search_form_post():
    
    request_data = RequestData(
        session_id=session.get('session_index'),  # Assuming you have a session_id in the session
        method=request.method,
        path=request.path,
        user_agent=request.headers.get('User-Agent'),
        # Add more relevant data
    )
    save_data()
    
    print("SESSION INDEX",session["session_index"])

    # Store request_data in-memory or database
    requests_data.append(request_data)
    if request.method == 'POST':
        search_query = request.form['search-query']
        search_type = request.form['search-type']
        session['last_search_query'] = search_query
        session['last_search_type'] = search_type
        search_id = analytics_data.save_query_terms(search_query,search_type)
        session['last_search_id']= search_id
        
        
    else:
        search_id = session['last_search_id']
        search_query = session['last_search_query']
        search_type = session['last_search_type']

 
    
    results = search_engine.search(search_query, search_id, corpus, list_of_tweets,search_type)

    found_count = len(results)
    session['last_found_count'] = found_count

    print(session)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count, s_type = search_type, query=search_query)

@app.route('/doc_details', methods=['GET'])
def doc_details():
    request_data = RequestData(
        session_id=session.get('session_index'),  # Assuming you have a session_id in the session
        method=request.method,
        path=request.path,
        user_agent=request.headers.get('User-Agent'),
        # Add more relevant data
    )
    # Store request_data in-memory or database
    requests_data.append(request_data)

    print("doc details session: ")
    print(session)

    res = session["some_var"]

    print("recovered var from session:", res)

    # get the query string parameters from request
    clicked_doc_id = request.args["id"]
    tweet = corpus[int(clicked_doc_id)]
    p1 = int(request.args["search_id"])  # transform to Integer
    p2 = int(request.args["param2"])  # transform to Integer
    print("click in id={}".format(clicked_doc_id))
    # store data in statistics table 1
    analytics_data.save_click(clicked_doc_id)
    analytics_data.save_requests(requests_data)
    requests_data.clear()
    

    print("fact_clicks count for id={} is {}".format(clicked_doc_id, analytics_data.fact_clicks[clicked_doc_id]))
    return render_template('doc_details.html',tweet=tweet)


@app.route('/stats', methods=['GET'])
def stats():
    request_data = RequestData(
        session_id=session.get('session_index'),  # Assuming you have a session_id in the session
        method=request.method,
        path=request.path,
        user_agent=request.headers.get('User-Agent'),
        # Add more relevant data
    )
    # Store request_data in-memory or database
    requests_data.append(request_data)
    


    docs = []
    
    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[int(doc_id)]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(row.id, row.title, row.description, row.doc_date, row.url, count)
        docs.append(doc)

    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    sorted_queries = dict(sorted(analytics_data.fact_queries.items(), key=lambda item: item[1], reverse=True))
    return render_template('stats.html', clicks_data=docs, query_data = sorted_queries)
    # ### End replace with your code ###


@app.route('/dashboard', methods=['GET'])
def dashboard():
    request_data = RequestData(
        session_id=session.get('session_index'),  # Assuming you have a session_id in the session
        method=request.method,
        path=request.path,
        user_agent=request.headers.get('User-Agent'),
        # Add more relevant data
    )

    # Store request_data in-memory or database
    requests_data.append(request_data)
    
    
    visited_docs = []
    visited_queries = []
    for doc_id in analytics_data.fact_clicks.keys():

        d: Document = corpus[int(doc_id)]
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append({
            'doc_id': doc.doc_id,
            'description': doc.description,
            'counter': doc.counter
            
        })
        
    for i, query in enumerate(analytics_data.fact_queries.keys()):
        
        visited_queries.append({
            'query': query,
            'counter': analytics_data.fact_queries[query]
        })

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc['counter'], reverse=True)
    visited_docs = visited_docs[:10]
    visited_queries.sort(key=lambda query: query['counter'], reverse=True)
    visited_queries  = visited_queries[:10]
    
    # PIE CHART SEARCH ENGINE
    headers = ['Search engine', 'Times used']
    search_type = [[key, value] for key, value in analytics_data.fact_se_type.items()]
    search_type.insert(0, headers)

    q_terms = analytics_data.fact_terms
    sorted_q_terms = dict(sorted(q_terms.items(), key=lambda item: item[1], reverse=True))
    sorted_q_terms = dict(itertools.islice(sorted_q_terms.items(), 15)) 

    return render_template('dashboard.html', visited_docs=visited_docs,searched_queries = sorted_q_terms, search_type = search_type,visited_queries = visited_queries, browser_data=analytics_data.fact_browser)



@app.route('/sentiment')
def sentiment_form():
    request_data = RequestData(
        session_id=session.get('session_index'),  # Assuming you have a session_id in the session
        method=request.method,
        path=request.path,
        user_agent=request.headers.get('User-Agent'),
        # Add more relevant data
    )

    print(request_data.path)

    # Store request_data in-memory or database
    requests_data.append(request_data)
    return render_template('sentiment.html')


@app.route('/sentiment', methods=['POST'])
def sentiment_form_post():
    request_data = RequestData(
        session_id=session.get('session_index'),  # Assuming you have a session_id in the session
        method=request.method,
        path=request.path,
        user_agent=request.headers.get('User-Agent'),
        # Add more relevant data
    )

    print(request_data.path)

    # Store request_data in-memory or database
    requests_data.append(request_data)
    text = request.form['text']
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    score = ((sid.polarity_scores(str(text)))['compound'])
    return render_template('sentiment.html', score=score)



def save_data():
    save_dicts(analytics_data.fact_clicks, analytics_data.fact_queries, analytics_data.fact_terms, analytics_data.fact_se_type, analytics_data.fact_browser, analytics_data.fact_request)
        

if __name__ == "__main__":
    # Schedule the function to run every 1 minute

    app.run(port=8088, host="0.0.0.0", threaded=False, debug=True)
    
