import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#Generate random data for two classes
np.random.seed(0)
X = np.concatenate([np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]])
y = np.array([-1] * 20 + [1] * 20)

#Fit the SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

#Get the coefficients of the support vector machine
w = clf.coef_[0]
b = clf.intercept_[0]

# Calculate the slope and intercept of the decision boundary
slope = -w[0] / w[1]
intercept = -b / w[1]

#Plot the data points and decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y)
x_vals = np.linspace(-5, 5, 100)
y_vals = slope * x_vals + intercept
plt.plot(x_vals, y_vals, '-r', label='Decision Boundary')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
