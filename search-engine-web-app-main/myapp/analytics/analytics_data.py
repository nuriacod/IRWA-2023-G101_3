import json
import random
import pickle
import os
filename = "data_collection.pkl"
def file_exists(filename):
        return os.path.exists(filename)

 # Function to load dictionaries from a pickle file
def load_dicts(filename):
    with open(filename, 'rb') as file:
        fact_clicks = pickle.load(file) 
        fact_queries = pickle.load(file)
        fact_terms = pickle.load(file)
        fact_se_type = pickle.load(file)
        fact_browser = pickle.load(file)
        fact_request = pickle.load(file)  

    return fact_clicks, fact_queries, fact_terms, fact_se_type, fact_browser ,fact_request


# Function to save dictionaries to a pickle file
def save_dicts(fact_click, fact_queries, fact_terms, fact_se_type, fact_browser, fact_request):
    with open(filename, 'wb') as file:
        pickle.dump(fact_click, file)
        pickle.dump(fact_queries, file)
        pickle.dump(fact_terms, file)
        pickle.dump(fact_se_type, file)
        pickle.dump(fact_browser, file)
        pickle.dump(fact_request, file)
        
class AnalyticsData:

    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """
    filename = filename
    # statistics table 1
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
    fact_clicks = dict([])

    # Dictionary with the query counter 
    fact_queries = dict([])

    # Fer un diccionary per terms (key = term| value = counter)
    fact_terms = dict([])
    
    # Fer un diccionary per search engine type (tf-idf / our_score)
    fact_se_type = dict([])

    fact_browser = dict([])
    
    fact_request = dict([])
    
    if file_exists(filename): 
        fact_clicks, fact_queries, fact_terms, fact_se_type, fact_browser,fact_request  = load_dicts(filename)
        


    def save_query_terms(self, terms: str,type: str) -> int:

        if type in self.fact_se_type.keys():
            self.fact_se_type[type] += 1
        else: 
            self.fact_se_type[type] = 1

        if terms in self.fact_queries.keys():
            self.fact_queries[terms] += 1
        else: 
            self.fact_queries[terms] = 1
            
        for term in terms.split(): 
            if term in self.fact_terms.keys():
                self.fact_terms[term] += 1
            else: 
                self.fact_terms[term] = 1
            
        return random.randint(0, 100000)

    
    def save_click(self, clicked_doc_id):
        if clicked_doc_id in self.fact_clicks.keys():
            self.fact_clicks[clicked_doc_id] += 1
        else: 
            self.fact_clicks[clicked_doc_id] = 1

    
    def save_browser(self, browser):
        if browser in self.fact_browser.keys():
            self.fact_browser[browser] += 1
        else: 
            self.fact_browser[browser] = 1
            
    def save_requests(self,requests):
        session_id = requests[0].session_id
        if session_id not in self.fact_request.keys():
            self.fact_request[session_id] = requests
            
        else:
            self.fact_request[session_id].extend(requests)



class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
