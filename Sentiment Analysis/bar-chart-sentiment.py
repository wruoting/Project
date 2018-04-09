import matplotlib.pyplot as plt
import numpy as np
import ast

with open("all_news.txt","r") as f_pos:
    all_news = f_pos.read().splitlines()

positive = 0
negative = 0
neutral = 0
for value in all_news:
    new_dict = ast.literal_eval(value)
    if(float(new_dict['compound']) > 0.2):
        positive += 1
    elif(float(new_dict['compound']) < -0.2):
        negative+=1
    else:
        neutral += 1
total = positive+negative+neutral
#For the bar-chart distribution
y_val = [float(neutral)/float(total),float(negative)/float(total),float(positive)/float(total)]
x_val = [1, 2, 3]
plt.style.use('ggplot')

ind = np.arange(len(x_val))
width = 0.3
fig, ax = plt.subplots()
ax.bar(ind+0.1,y_val,width, color='green')
ax.set_xticks(ind+0.1+width/2)
ax.set_xticklabels(['Neutral', 'Negative', 'Positive'])
ax.legend()
plt.title("Categories Distribution")
plt.xlabel("Categories")
plt.ylabel("Percentage")
plt.show()
