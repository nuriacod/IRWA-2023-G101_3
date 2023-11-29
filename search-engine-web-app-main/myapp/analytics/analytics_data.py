import json
import random


class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """
    # statistics table 1
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
    fact_clicks = dict([])

    # Dictionary with the query counter 
    fact_queries = dict([])

    # Fer un diccionary per terms (key = term| value = counter)
    fact_terms = dict([])
    
    # Fer un diccionary per search engine
    fact_se_type = dict([])
    
    ####(acabar de revisar perque ara cada cop que fas go back li suma a tots i no hauria
    ### Pude podem mirar de posar un if dins de la funcio de /search per controlar això

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
            
                
        # AFegir un diccionari o algo per controlar les queries 
        #que no es repeteixin ids i si una query es repeteix donar li el que li pertoca
        return random.randint(0, 100000)

## vegades per query 
## clicks per query 


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
