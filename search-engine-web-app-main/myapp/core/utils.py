import datetime
import json
from random import random
import pickle
import os

from faker import Faker

fake = Faker()


# fake.date_between(start_date='today', end_date='+30d')
# fake.date_time_between(start_date='-30d', end_date='now')
#
# # Or if you need a more specific date boundaries, provide the start
# # and end dates explicitly.
# start_date = datetime.date(year=2015, month=1, day=1)
# fake.date_between(start_date=start_date, end_date='+30y')

def get_random_date():
    """Generate a random datetime between `start` and `end`"""
    return fake.date_time_between(start_date='-30d', end_date='now')


def get_random_date_in(start, end):
    """Generate a random datetime between `start` and `end`"""
    return start + datetime.timedelta(
        # Get a random amount of seconds between `start` and `end`
        seconds=random.randint(0, int((end - start).total_seconds())), )


def load_json_file(path):
    """Load JSON content from file in 'path'

    Parameters:
    path (string): the file path

    Returns:
    JSON: a JSON object
    """

    # Load the file into a unique string
    with open(path) as fp:
        #iterar lineas i afegir llista
        text_data = []
        for line in fp.readlines():
            text_data.append(json.loads(line))


            
        
    # Parse the string into a JSON object
    #json_data = json.loads(text_data) # hem de passarli linia per linia

    '''text_data is a list of dictionaries. each entry is a line of the json converted to python dictionary'''

    '''df = pd.json_normalize(data)'''
    return text_data


def file_exists(filename):
    return os.path.exists(filename)

# Function to load dictionaries from a pickle file
def load_dicts(filename):
    with open(filename, 'rb') as file:
        index= pickle.load(file)
        tf = pickle.load(file)
        df = pickle.load(file)
        idf = pickle.load(file)
        our_score = pickle.load(file)
    return index, tf, df, idf,our_score


# Function to save dictionaries to a pickle file
def save_dicts(filename, index, tf, df, idf,our_socore):
    with open(filename, 'wb') as file:
        pickle.dump(index, file)
        pickle.dump(tf, file)
        pickle.dump(df, file)
        pickle.dump(idf, file)
        pickle.dump(our_socore, file)

