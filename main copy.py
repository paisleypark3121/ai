import numpy as np
import pandas as pd

import sklearn
import sklearn.naive_bayes as nb
import sklearn.feature_extraction.text as text
import sklearn.model_selection as cv
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score

#print("Hello World")

df = pd.read_csv('data.csv', sep=';')

counts = text.CountVectorizer()

#print(df['Value'])
X = counts.fit_transform(df['Value'])
y = df['Classification'] == 'man'

print(X)
print(y)

(text_train, text_test, label_train, label_test) = cv.train_test_split(X, y, test_size=0.2)

# from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(text_train, label_train)

MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
sc=classifier.score(text_test, label_test)
print(sc)

predictions = classifier.predict(text_test)
confusion_matrix(label_test, predictions)

values = ["Gianni", "Sara"]
transformed_messages = counts.transform(values)
predictions = classifier.predict(transformed_messages)
print(predictions)