import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import *
from plot_decision_regions import *

df = pd.read_csv('training_data/vacuole_data_merged.csv')

# get two sets of features
# for our two species of interest
df_smaller = df.loc[df['input_body_mean'] == 25]
import pdb; pdb.set_trace()
X = df.iloc[:, [3, 5]].values

# get labels converted to integers for perceptron classification
# setosa and versicolor
y = df.iloc[:, 6].values
Y = np.where(y == 'glpk', -1, 1)

# make perceptron object
ppn = Perceptron(eta = 0.1, n_iter = 10, random_state = 12345)

# plot classifier errors vs fit iterations
plt.figure(0)
ppn.fit(X, Y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
plt.xlabel('Fit iterations')
plt.ylabel('Misclassifications')
plt.savefig('vacuole_error.png')


# plot decision regions
plt.figure(1)
plot_decision_regions(X, Y, classifier = ppn)
plt.xlabel('Mean Body Radii (nm)')
plt.ylabel('Observed Bodies')
plt.legend(loc='upper left')
plt.savefig('vacuole_decision.png')
