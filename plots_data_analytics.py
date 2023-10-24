import matplotlib.pyplot as plt
import datetime
from wordcloud import WordCloud
import string

def temporal_plot(list_of_tweets):
    """
    Description: Generates a temporal plot of tweet activity over time based on a list of tweets. 
    The function parses the creation dates of tweets, counts the tweets per day, and 
    visualizes the temporal distribution in a line plot.
    """

    # Tweet counts per day
    tweet_counts = {}

    for tweet in list_of_tweets:
        # Get timeStamp field and convert to date type 
        created_at = tweet.split('|')[3].strip()
        created_at = datetime.datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')
        date = created_at.date()
        
        # Update the tweet count
        tweet_counts[date] = tweet_counts.get(date, 0) + 1

    # Sort the dictionary by date
    sorted_tweet_counts = sorted(tweet_counts.items())

    # Extract the dates and corresponding tweet counts
    dates, counts = zip(*sorted_tweet_counts)

    # Create a line plot to visualize tweet activity over time
    plt.figure(figsize=(10, 6))
    plt.plot(dates, counts, marker='o')
    plt.title('Temporal Distribution of Tweets')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.grid()
    plt.savefig('./results/temporalPlot.jpg')
    plt.close()
    
def word_cloud(word_dist):
    """
    Creates a word cloud visualization based on a dictionary of word frequency distribution. 
    The word cloud displays words with sizes relative to their frequencies in the input data.
    """
    wordcloud = WordCloud(width = 800, height = 800,
            background_color ='white').generate(' '.join(word_dist.keys()))
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig('./results/wordcloud.jpg')
    plt.close()
    
def text_length_distribution(list_of_tweets):
    """
    Plots a histogram representing the distribution of text lengths in a list of tweets. 
    The function counts the number of words in each tweet and generates a histogram to 
    show the frequency of various text lengths.
    """
    #LENGTH DISTRIBUTION
    text_lengths = []

    for tweet in list_of_tweets:
        #get the second element -> tweet text
        full_text = tweet.split('|')[2].strip()
        words = full_text.split()
        # Append the word count to the word_counts array
        text_lengths.append(len(words))

    # Create a histogram plot
    plt.hist(text_lengths, bins=20, edgecolor='black')

    # Set labels and title
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Full Text Lengths')

    # Save the plot 
    plt.savefig('./results/full_length_distribution.jpg')
    plt.close()
