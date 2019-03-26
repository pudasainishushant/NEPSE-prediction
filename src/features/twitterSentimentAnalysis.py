import tweepy
import csv
import numpy as np
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense


consumer_key = '0FAwDEdtG0DlUCHdKgICtLmHf'
consumer_secret = 'y1msNFUm95fwb0VPtRcFBFgqWuGx3asReDFN4Nlf9KqEUi4Zrg'
access_token = '955769490490384384-jZiHSMxcnnWkhuyCYjinvzVQjBKwc3p'
access_token_secret = 'JLQY4tAntt8qeEnoh0N0SxZ2Fe04QvA98FTGmnUOMgT4m'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


def stockSentiment(stockName, numTweets=100):
    public_tweets = api.search(stockName, count=numTweets)
    print(public_tweets[2].text)
    threshold = posSentTweet = negSentTweet = 0

    for tweet in public_tweets:
        analysis = TextBlob(tweet.text)
        print(analysis.sentiment)
        if analysis.sentiment.polarity >= threshold:
            posSentTweet = posSentTweet + 1
        else:
            negSentTweet = negSentTweet + 1

    if posSentTweet > negSentTweet:
        print("Overall Positive")
        return True
    else:
        print("Overall Negative")
        return False


stockSentiment('Nabil Bank')

# data collection from csv files
dates = []
prices = []


def stock_prediction(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))
    return


stock_prediction('NBL.csv')
# creating the dataset matrix


def create_datasets(dates, prices):
    train_size = int(0.80 * len(dates))
    TrainX, TrainY = [], []
    TestX, TestY = [], []
    cntr = 0
    for date in dates:
        if cntr < train_size:
            TrainX.append(date)
        else:
            TestX.append(date)
    for price in prices:
        if cntr < train_size:
            TrainY.append(price)
        else:
            TestY.append(price)

    return TrainX, TrainY, TestX, TestY


def predict_prices(dates, prices, x):

    TrainX, TrainY, TestX, TestY = create_datasets(dates, prices)

    TrainX = np.reshape(TrainX, (len(TrainX), 1))
    TrainY = np.reshape(TrainY, (len(TrainY), 1))
    TestX = np.reshape(TestX, (len(TestX), 1))
    TestY = np.reshape(TestY, (len(TestY), 1))

# create multilayer perrceptron model
    model = Sequential()
    model.add(Dense(32, input_dim=1, init='uniform', activation='relu'))
    model.add(Dense(32, input_dim=1, init='uniform', activation='relu'))
    model.add(Dense(16, init='uniform', activation='relu'))

    model.add(Dense(1, init='uniform', activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(TrainX, TrainY, nb_epoch=100, batch_size=3, verbose=1)

#    # # Our prediction for tomorrow
#    # prediction = model.predict(np.array([dataset[0]]))
#    # result = 'The price will move from %.2f to %.2f' % (
#    #     dataset[0], prediction[0][0])
#    # return result

#    predicted_price = predict_price(dates, prices, 29)
#    print(predicted_price)


predict_prices(dates, prices, 2)


# if __name__ == "__main__":
# Ask user for a stock name
# stockName = input('Enter a stock quote: ').upper()
