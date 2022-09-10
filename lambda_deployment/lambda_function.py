import snscrape.modules.twitter as sntwitter
import pandas as pd
from simpletransformers.classification import ClassificationModel
import numpy as np

def lambda_handler(event, context):
    # Created a list to append all tweet attributes(data)
    attributes_container = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    twitter_handle = event['queryStringParameters']['user']
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:' + twitter_handle).get_items()):
        if i==100:
            break
        attributes_container.append([tweet.content, tweet.url])
        
    # Creating a dataframe from the tweets list above 
    tweets_df = pd.DataFrame(attributes_container, columns=['Tweets', 'Urls'])

    model = ClassificationModel('roberta', 'trained_model', use_cuda=False)
    eval_set = tweets_df['Tweets'].astype(str)
    predictions = model.predict(eval_set.tolist())
    results = np.array([])
    results = np.append(results, np.argmax(predictions[1], axis=1))

    tweets_df['Results'] = results.tolist()
    toxic_df = tweets_df[tweets_df['Results'] != 2]

    return {
        'statusCode': 200,
        'body': toxic_df['Urls'].to_json()
    }
