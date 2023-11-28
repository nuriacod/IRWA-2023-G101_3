import json
from datetime import datetime

class Document:
    """
    Original corpus data as an object
    """

    def __init__(self, id, title, description, doc_date, likes, retweets, url, hashtags,followers_count,verified):
        self.id = id
        self.title = title
        self.description = description
        self.doc_date = doc_date
        self.likes = likes
        self.retweets = retweets
        self.url = url
        self.hashtags = hashtags
        self.followers_count = followers_count
        self.verified = verified

    def to_json(self):
        return self.__dict__

    def format_date(self):
        # Parse the original date string
        original_date = datetime.strptime(self.doc_date, '%a %b %d %H:%M:%S %z %Y')

        # Format the date into the desired format
        formatted_date = original_date.strftime('%a %b %d %Y %H:%M')

        # Update the doc_date attribute with the formatted date
        self.doc_date = formatted_date

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)


class StatsDocument:
    """
    Original corpus data as an object
    """

    def __init__(self, id, title, description, doc_date, url, count):
        self.id = id
        self.title = title
        self.description = description
        self.doc_date = doc_date
        self.url = url
        self.count = count

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)


class ResultItem:
    def __init__(self, id, title, description, doc_date, url, ranking):
        self.id = id
        self.title = title
        self.description = description
        self.doc_date = doc_date
        self.url = url
        self.ranking = ranking
    def format_date(self):
        # Parse the original date string
        original_date = datetime.strptime(self.doc_date, '%a %b %d %H:%M:%S %z %Y')

        # Format the date into the desired format
        formatted_date = original_date.strftime('%a %b %d %Y %H:%M')

        # Update the doc_date attribute with the formatted date
        self.doc_date = formatted_date

class Session:
    def __init__(self, session_id, user_agent, ip_address, country):
        self.session_id = session_id
        self.user_agent = user_agent
        self.ip_address = ip_address
        self.country = country
        self.clicks = []
        self.requests = []

class Click:
    def __init__(self, session_id, action, query, document_id, rank, dwell_time):
        self.session_id = session_id
        self.action = action
        self.query = query
        self.document_id = document_id
        self.rank = rank
        self.dwell_time = dwell_time

class RequestData:
    def __init__(self, session_id, method, path, user_agent):
        self.session_id = session_id
        self.method = method
        self.path = path
        self.user_agent = user_agent
        # Add more relevant data

class Query:
    def __init__(self, session_id, terms, order):
        self.session_id = session_id
        self.terms = terms
        self.order = order

class Result:
    def __init__(self, session_id, query, document_id, rank):
        self.session_id = session_id
        self.query = query
        self.document_id = document_id
        self.rank = rank

class UserContext:
    def __init__(self, session_id, browser, os, time_of_day, date, ip_address, country, city=None):
        self.session_id = session_id
        self.browser = browser
        self.os = os
        self.time_of_day = time_of_day
        self.date = date
        self.ip_address = ip_address
        self.country = country
        self.city = city