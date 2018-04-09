#naive bayes sentiment analysis
#we need natural language toolkit
import requests
import json
import time
import csv
import nltk
from io import open
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
#nltk.download('vader_lexicon')
# Set your header according to the form below
# :: (by /u/)

# Add your username below

subreddits = ['politics','news','USNEWS','worldnews','truenews']

def subreddit_search(subreddits):
    data_all = []
    for subreddit in subreddits:
        print("Taking posts from r/" +subreddit)
        hdr = {'User-Agent': 'windows:r/'+subreddit+'.single.result:v1.0' +
               '(by /u/arrowshaft)'}
        url = 'https://www.reddit.com/r/'+subreddit+'/.json'
        req = requests.get(url, headers=hdr)
        json_data = json.loads(req.text)
        posts = json.dumps(json_data['data']['children'], indent=4, sort_keys=True)
        #this scrapes the data
        data_sub = json_data['data']['children']

        num_of_posts = 0
        while len(data_sub) <= 100:
            print("Number of headlines from r/"+subreddit+" : "+str(len(data_sub)))
            time.sleep(2)
            last = data_sub[-1]['data']['name']
            url = 'https://www.reddit.com/r/'+subreddit+'/.json?after=' + str(last)
            req = requests.get(url, headers=hdr)
            data = json.loads(req.text)
            data_sub += data['data']['children']
            if num_of_posts == len(data_sub):
                break
            else:
                num_of_posts = len(data_sub)
        data_all += data_sub
    return data_all

data_all = subreddit_search(subreddits)
sia = SIA()
positive_list = []
negative_list = []
polarity_list = []
neutral_list = []

for post in data_all:
    res = sia.polarity_scores(post['data']['title'])
    polarity_list.append(str(res))
    neutral_list.append(post['data']['title'])
    if res['compound'] > 0.2:
        positive_list.append(post['data']['title'])
    elif res['compound'] < 0.2:
        negative_list.append(post['data']['title'])

#encoding and errors should be passed in python 3
with open("compound.txt","w",encoding='utf-8') as f_comp:
    for post in polarity_list:
        f_comp.write(post.decode('utf-8')+"\n")
with open("positive_news.txt","w",encoding='utf-8') as f_pos:
    for post in positive_list:
        f_pos.write(post+"\n")
with open("negative_news.txt","w",encoding='utf-8') as f_neg:
    for post in negative_list:
        f_neg.write(post+"\n")
with open("neutral_news.txt","w",encoding='utf-8') as f_neu:
    for post in neutral_list:
        f_neu.write(post+"\n")
