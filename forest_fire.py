import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
'''Pickle is a useful Python tool that allows you to save your models, to minimise lengthy re-training and
   allow you to share, commit, and re-load pre-trained machine learning models.
   Pickle is a generic object serialization module that can be used for serializing and deserializing objects.'''
from sklearn import metrics
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

# Load the csv file
data = pd.read_csv("Forest_fire.csv")

print(data.head())

# converting data into numpy array.Because Numpy arrays are more compact than python lists,
# which uses less memory and is convenient to use.
data = np.array(data)

# select dependent and independent variables
X = data[1:, 1:-1]
y = data[1:, -1]

y = y.astype('int')
X = X.astype('int')

print("\n", X)
print("\n", y)

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

'''Logistic Regression is a supervised Machine Learning algorithm, which means the data provided for 
   training is labeled i.e., answers are already provided in the training set. The algorithm learns from 
   those examples and their corresponding answers (labels) and then uses that to classify new examples.'''

# Instantiate the model
log_reg = LogisticRegression()

# Fit the model
log_reg.fit(X_train, y_train)

y_predict = log_reg.predict(X_test)
print("\n Prediction after training:", y_predict)
print("\n Accuracy:", metrics.accuracy_score(y_test, y_predict))

print("\n Confusion Matrix\n", confusion_matrix(y_test, y_predict))

'''So now that we have trained our model we will import it into a pickle file. As we don't want our entire model 
   to run again and again when ever using our website ,so we train our model once and importing our model  
   somewhere else from where we can directly fetch it.So that's where pickle comes in.'''

# pickle file of model
# Dump is used for converts a Python object into a byte stream. This process is also called as serialization.
# The converted byte stream can be written to a buffer or to a disk file.
pickle.dump(log_reg, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
