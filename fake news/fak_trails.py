import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import itertools as it
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

class sai():
    def __init__(self):
        pass

    def fake_news24(self):
        my_data = pd.read_csv('data.csv')
        labels = my_data.Label
        x_train,x_test,y_train,y_test=train_test_split(my_data['Body'], labels, test_size=0.2, random_state=7)
        tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
        tfidf_train=tfidf_vectorizer.fit_transform(x_train)
        tfidf_test=tfidf_vectorizer.transform(x_test)
        pac=PassiveAggressiveClassifier(max_iter=50)
        pac.fit(tfidf_train,y_train)
        #print(tfidf_test)
        y_pred=pac.predict(tfidf_test)
        print(y_pred)
        return y_pred

