import numpy as np
import pandas as pd
import gzip

from wordcloud import WordCloud
import PIL
import itertools
import matplotlib.pyplot as plt

import re
import itertools

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from time import time
from sklearn import metrics


stop_words =['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself',
            'yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself',
            'they','them','their','theirs','themselves','what','which','who','whom','this','that',
            'these','those','am','is','are','was','were','be','been','being','have','has','had',
            'having','do','does','did','doing','a','an','the','and','but','if','or','because','as',
            'until','while','of','at','by','for','with','about','against','between','into','through',
            'during','before','after','above','below','to','from','up','down','in','out','on','off',
            'over','under','again','further','then','once','here','there','when','where','why','how',
            'all','any','both','each','few','more','most','other','some','such','no','nor','not',
            'only','own','same','so','than','too','very','s','t','can','will','just','don','should',
            'now','uses','use','using','used','one','also']

def preprocess(data):
    reviews_tokens = []
    for review in data:
        #print(review)
        review = review.lower() #Convert to lower-case words
        raw_word_tokens = re.findall(r'(?:\w+)', review,flags = re.UNICODE) #remove pontuaction
        word_tokens = [w for w in raw_word_tokens if not w in stop_words] # do not add stop words
        reviews_tokens.append(word_tokens)
    return reviews_tokens #return all tokens

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)
 
def getDF(path): 
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def featurize(sentence_tokens,bag_of_words):
    sentence_features = [0 for x in range(len(bag_of_words))]

    for word in sentence_tokens:
        index = bag_of_words[word]
        sentence_features[index] +=1
    return sentence_features

def get_batch_features(data,bag_of_words):
    batch_features = []
    reviews_text_tokens = preprocess(data)
    for review_text in reviews_text_tokens:
        feature_review_text = featurize(review_text,bag_of_words)
        batch_features.append(feature_review_text)
    return batch_features

def construct_bag_of_words(data):    
    #print(data)
    corpus = preprocess(data)
    bag_of_words = {}
    word_count = 0
    for sentence in corpus:
        for word in sentence:
            if word not in bag_of_words: # do not allow repetitions
                bag_of_words[word] = word_count #set indexes
                word_count+=1
            
    #print(dict(Counter(bag_of_words).most_common(5)))
    return bag_of_words #index of letters

digital_music_data = getDF('reviews_Digital_Music_5.json.gz')

sample_len=10
frames = [digital_music_data.reviewText[:sample_len]]
print(frames)
complete_data = pd.concat(frames, keys = ['digital_music'])
#print(complete_data)
bag_of_words = construct_bag_of_words(complete_data)
# batch_features = get_batch_features(complete_data,bag_of_words)

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
# #     #image = wordcloud.to_image()
# #     #image.show()
# #     #image.save(name+'.bmp')


