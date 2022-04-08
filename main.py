import nltk
#nltk.download()

import numpy as np
import pandas as pd
import gzip

from wordcloud import WordCloud
import PIL
import itertools
import matplotlib.pyplot as plt

import re
import itertools


import sklearn.naive_bayes as nb
import sklearn.feature_extraction.text as text
import sklearn.model_selection as cv
from sklearn.naive_bayes import MultinomialNB

def parse(path):
    g = open(path, 'rb')
    for l in g:
        yield eval(l)
 
def getDF(path): 
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

stop_words=nltk.corpus.stopwords.words('italian')

frasi=pd.read_json('frasi.json')
frasi=frasi.drop(columns=['id'])





full_tokens = []
for index, row in frasi.iterrows():
    #print(row['testo'])        
    testo = row['testo'].lower() 
    raw_word_tokens = re.findall(r'(?:\w+)', testo,flags = re.UNICODE) #remove pontuaction
    word_tokens = [w for w in raw_word_tokens if not w in stop_words] # do not add stop words
    full_tokens.append(word_tokens)

#print(reviews_tokens)
words = list(itertools.chain(*full_tokens))
words = " ".join(words)
wordcloud = WordCloud( max_words=1000,margin=0).generate(words)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
# #     #image = wordcloud.to_image()
# #     #image.show()
# #     #image.save(name+'.bmp')





counts = text.CountVectorizer()

X = counts.fit_transform(frasi['testo'])
y = frasi['clas']
(text_train, text_test, label_train, label_test) = cv.train_test_split(X, y,test_size=0.2)

classifier = MultinomialNB()
classifier.fit(text_train, label_train)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
score=classifier.score(text_test, label_test)
#print(score)

messages = ["ciao a tutti", "ciao a voi"]
transformed_messages = counts.transform(messages)
predictions = classifier.predict(transformed_messages)
print(predictions)



# labels = []
# for i in range(sample_len):
#     labels.append(0)
# digital_music_data=[]

# X_train, X_test, y_train, y_test = train_test_split(batch_features, labels, test_size=0.33, random_state=42)

# # #clf = Perceptron(max_iter=50)
# # clf = MultinomialNB(alpha=.01)
# # clf.fit(X_train, y_train)
# # pred = clf.predict(X_test)
# # score = metrics.accuracy_score(y_test, pred)
# # print("accuracy:   %0.3f" % score)

# # print(X_train)

# # reviews_names = ['digital_music']
# # for reviews,name in zip(frames,reviews_names):
# #     tokenized_reviews = preprocess(reviews) #apply the preprocess step
# #     reviews = list(itertools.chain(*tokenized_reviews))
# #     text_reviews = " ".join(reviews)
# #     wordcloud = WordCloud( max_words=1000,margin=0).generate(text_reviews)
# #     plt.figure()
# #     plt.imshow(wordcloud, interpolation="bilinear")
# #     plt.axis("off")
# #     plt.show()


