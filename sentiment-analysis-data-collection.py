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
hdr = {'User-Agent': 'windows:r/politics.single.result:v1.0' +
       '(by /u/arrowshaft)'}
url = 'https://www.reddit.com/r/politics/.json'
req = requests.get(url, headers=hdr)
json_data = json.loads(req.text)

posts = json.dumps(json_data['data']['children'], indent=4, sort_keys=True)

#this scrapes the data
data_all = json_data['data']['children']
num_of_posts = 0
while len(data_all) <= 200:
    time.sleep(2)
    last = data_all[-1]['data']['name']
    url = 'https://www.reddit.com/r/politics/.json?after=' + str(last)
    req = requests.get(url, headers=hdr)
    data = json.loads(req.text)
    data_all += data['data']['children']
    if num_of_posts == len(data_all):
        break
    else:
        num_of_posts = len(data_all)

sia = SIA()
positive_list = []
negative_list = []
all_list = []

for post in data_all:
    res = sia.polarity_scores(post['data']['title'])
    print(res)
    all_list.append(str(res))
    if res['compound'] > 0.2:
        positive_list.append(post['data']['title'])
    elif res['compound'] < 0.2:
        negative_list.append(post['data']['title'])
#encoding and errors should be passed in python 3
with open("all_news.txt","w",encoding='utf-8') as f_pos:
    for post in all_list:
        f_pos.write(post.decode('utf-8')+"\n")
with open("positive_news.txt","w",encoding='utf-8') as f_pos:
    for post in positive_list:
        f_pos.write(post+"\n")
with open("negative_news.txt","w",encoding='utf-8') as f_neg:
    for post in negative_list:
        f_neg.write(post+"\n")
