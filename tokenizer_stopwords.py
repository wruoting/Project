import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
import matplotlib.pyplot as plt
import math

tokenizer = RegexpTokenizer('\w+')
stop_words = set(stopwords.words('english'))
stop_words.update(('\xe2'))
stop_words.update(('u'))

#print(stop_words)
all_words_pos = []
with open("positive_news.txt", "r") as f_pos:
    for line in f_pos.readlines():
        words = tokenizer.tokenize(line)
        for w in words:
            if w.lower() not in stop_words:
                all_words_pos.append(w.lower())
pos_res = nltk.FreqDist(all_words_pos)

all_words_neg = []
with open("negative_news.txt", "r") as f_pos:
    for line in f_pos.readlines():
        words = tokenizer.tokenize(line)
        for w in words:
            if w.lower() not in stop_words:
                all_words_neg.append(w.lower())
neg_res = nltk.FreqDist(all_words_neg)

all_words_neu = []
with open("neutral_news.txt", "r") as f_pos:
    for line in f_pos.readlines():
        words = tokenizer.tokenize(line)
        for w in words:
            if w.lower() not in stop_words:
                all_words_neu.append(w.lower())
neu_res = nltk.FreqDist(all_words_neu)


print(neg_res.unicode_repr())
