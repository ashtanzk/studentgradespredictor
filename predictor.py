import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=";")
data = pd.DataFrame(data) # not sure if need to make into a dataframe or just leave it as a np array
data = data[["G1","G2","G3","studytime","failures","absences"]]

# separating the dataset into label & features
# the features will determine the label (which is what we are trying to predict)
# in this case, we are using the student's 1st and 2nd grade scores, and their studytime, failures and absences to predict their 3rd grade
predict = "G3"
X = np.array(data.drop([predict], 1)) # dataset of features
y = np.array(data[predict]) # dataset of label
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)


pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)

acc = linear.score(x_test, y_test)

print(acc)