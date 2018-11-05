import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import *
from plot_decision_regions import *
import pdb

df = pd.read_csv('training_data/iris.csv',
                 header=None)

# get two sets of features (sepal and petal length)
# for our two species of interest
X = df.iloc[0:100, [0, 2]].values

# get labels converted to integers for perceptron classification
# setosa and versicolor
y = df.iloc[0:100, 4].values
Y = np.where(y == 'Iris-setosa', -1, 1)

# make perceptron object
ppn = Perceptron(eta = 0.1, n_iter = 10, random_state = 12345)

# plot classifier errors vs fit iterations
plt.figure(0)
ppn.fit(X, Y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
plt.xlabel('Fit iterations')
plt.ylabel('Misclassifications')
plt.savefig('iris_error.png')


# plot decision regions
plt.figure(1)
plot_decision_regions(X, Y, classifier = ppn)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(loc='upper left')
plt.savefig('iris_decision.png')
