import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
import matplotlib.pyplot as plt
import math

tokenizer = RegexpTokenizer('\w+')
stop_words = set(stopwords.words('english'))
stop_words.update(('\xe2'))

#print(stop_words)
all_words_pos = []
with open("positive_news.txt", "r") as f_pos:
    for line in f_pos.readlines():
        words = tokenizer.tokenize(line)
        print(words)
        for w in words:
            if w.lower() not in stop_words:
                all_words_pos.append(w.lower())
pos_res = nltk.FreqDist(all_words_pos)
print(pos_res)
