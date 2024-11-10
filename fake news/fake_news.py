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
#for dirname, _, filenames in os.walk('data.csv'):
    #for filename in filenames:
       # print(os.path.join(dirname, filename))
my_data = pd.read_csv('data.csv')
my_data.head(10)

