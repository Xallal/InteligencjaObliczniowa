from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
import random
import operator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("diabetes.csv")

training_length = len(df)*0.67

training_data = df.iloc[:int(training_length),:]
test_data = df.iloc[int(training_length)+1:,:]


classifier = GaussianNB()
classifier.fit(training_data.iloc[:,:-2], training_data.iloc[:,-1])
predicted_labels = classifier.predict(test_data.iloc[:,:-2])
expected_labels = test_data.iloc[:,-1]
accuracy1 = classifier.score(test_data.iloc[:,:-2], expected_labels)
print(confusion_matrix(predicted_labels,expected_labels))
print(accuracy1)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(training_data.iloc[:,:-2], training_data.iloc[:,-1])
predicted_labels = classifier.predict(test_data.iloc[:,:-2])
accuracy2 = classifier.score(test_data.iloc[:,:-2], expected_labels)
print(confusion_matrix(predicted_labels,expected_labels))
print(accuracy2)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(training_data.iloc[:,:-2], training_data.iloc[:,-1])
predicted_labels = classifier.predict(test_data.iloc[:,:-2])
accuracy3 = classifier.score(test_data.iloc[:,:-2], expected_labels)
print(confusion_matrix(predicted_labels,expected_labels))
print(accuracy3)

classifier = KNeighborsClassifier(n_neighbors=11)
classifier.fit(training_data.iloc[:,:-2], training_data.iloc[:,-1])
predicted_labels = classifier.predict(test_data.iloc[:,:-2])
accuracy4 = classifier.score(test_data.iloc[:,:-2], expected_labels)
print(confusion_matrix(predicted_labels,expected_labels))
print(accuracy4)

classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(training_data.iloc[:,:-2], training_data.iloc[:,-1])
predicted_labels = classifier.predict(test_data.iloc[:,:-2])
accuracy5 = classifier.score(test_data.iloc[:,:-2], expected_labels)
print(confusion_matrix(predicted_labels,expected_labels))
print(accuracy5)


etykiety = ['Bayes', 'Nk = 3', 'Nk = 5', 'Nk = 11', 'Decisive tree']
wartosci = [accuracy1,accuracy2,accuracy3,accuracy4,accuracy5]
plt.bar(etykiety, wartosci)
plt.xticks(rotation=45)
plt.ylabel('Procentowa skuteczność')
plt.xlabel('Rodzaj klasyfikatora')
plt.show()
