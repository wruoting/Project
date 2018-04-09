# -*- coding: utf-8 -*-
import nltk,io,math
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
import numpy as np
import matplotlib.pyplot as plt

tokenizer = RegexpTokenizer('\w+')
stop_words = set(stopwords.words('english'))
stop_words.update(('u'))


#print(stop_words)
all_words_pos = []
all_words_neg = []
all_words_neu = []

def token_analysis(filename,type_of_words,stop_words):
    with io.open(filename,"r",encoding='utf-8') as f_words:
        for line in f_words.readlines():
            words = tokenizer.tokenize(line)
            for w in words:
                if w.lower() not in stop_words:
                    type_of_words.append(w.lower())
    return type_of_words

all_words_pos = token_analysis("positive_news.txt",all_words_pos,stop_words)
all_words_neg = token_analysis("negative_news.txt",all_words_neg,stop_words)
all_words_neu = token_analysis("neutral_news.txt",all_words_neu,stop_words)

pos_res = nltk.FreqDist(all_words_pos)
neg_res = nltk.FreqDist(all_words_neg)
neu_res = nltk.FreqDist(all_words_neu)

y_val_pos = [x[1] for x in pos_res.most_common(len(all_words_pos))]
y_log_pos = np.log(y_val_pos)
x_log_pos = [math.log(i+1) for i in range(len(y_val_pos))]

y_val_neg = [x[1] for x in neg_res.most_common(len(all_words_neg))]
y_log_neg = np.log(y_val_neg)
x_log_neg = [math.log(i+1) for i in range(len(y_val_neg))]

y_val_neu = [x[1] for x in neu_res.most_common(len(all_words_neu))]
y_log_neu = np.log(y_val_neu)
x_log_neu = [math.log(i+1) for i in range(len(y_log_neu))]

fig, (ax_pos,ax_neg) = plt.subplots(2,sharex = True,figsize=[16,9])
ax_pos.plot(x_log_pos,y_log_pos,label="Positive sentiment")
ax_pos.set_ylabel('Frequency (Log)')
ax_pos.set_xlabel('Number of Words (Log)')
ax_pos.legend(loc='best')
ax_pos.grid()

ax_neg.plot(x_log_neg,y_log_neg,label="Negative sentiment")
ax_neg.set_ylabel('Frequency (Log)')
ax_neg.set_xlabel('Number of Words (Log)')
ax_neg.legend(loc='best')
ax_neg.grid()
plt.show()

#analysis shows that zipf's law P * R = c where R is the rank of the word, P is the probability of the
# word in the set, and c is the constant, that P = c/r should be a constant line.
# shows that there is a small minority of words that appear the most in posts (omitting the stopwords)
