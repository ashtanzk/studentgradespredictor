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

# splitting the data set into the training and testing data with 90% as training data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# the actual algorithm we will be using to make predictions
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

# finding the accuracy of the algorithm
acc = linear.score(x_test, y_test) # acc stands for accuracy

print(acc)

print('Coefficient: \n', linear.coef_) # These are each slope value
print('Intercept: \n', linear.intercept_) # This is the intercept

# comparing the test data with the predictions
predictions = linear.predict(x_test) # Gets a list of all predictions

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# saving the model from above training code
# linear is the name of the model we created in the last tutorial
# it should be defined above this
with open("studentgrades.pickle", "wb") as f:
    pickle.dump(linear, f)

