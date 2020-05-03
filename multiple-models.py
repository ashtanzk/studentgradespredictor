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


best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    # If the current model has a better score than one we've already trained then save it
    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)


# Drawing and plotting model
plot = "failures" # Change this to G1, G2, studytime or absences to see other graphs
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()