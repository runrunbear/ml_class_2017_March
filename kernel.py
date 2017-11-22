# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 20:58:19 2015

@author: Rachel Zhang
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris();
print 'type of iris: ', type(iris) #<class 'sklearn.datasets.base.Bunch'>
print 'keys:', iris.keys() #['target_names', 'data', 'target', 'DESCR', 'feature_names']

X = iris.data[:,:2] #only use the first two features
Y = iris.target



# create instances of svm
linear_svc = svm.SVC(kernel = 'linear').fit(X,Y)#W'x
rbf_svc = svm.SVC(kernel = 'rbf', gamma = 0.7).fit(X,Y)
poly_svc = svm.SVC(kernel = 'poly', degree = 3).fit(X,Y)

# create a mesh
x0_min, x0_max = X[:,0].min()-0.1, X[:,0].max()+0.1
x1_min, x1_max = X[:,1].min()-0.1, X[:,1].max()+0.1
h = .02 # step size in the mesh
xx,yy = np.meshgrid(np.arange(x0_min,x0_max,h),np.arange(x1_min,x1_max,h))

titles = ['linear','rbf','poly']
for i,clf in enumerate((linear_svc,rbf_svc,poly_svc)):
    plt.subplot(1,3,i+1)
    plt.subplots_adjust(wspace = 0.4, hspace = 0.4)
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    #Z = np.random.randint(0,3,xx.shape)
    
    #draw decision boundary
    plt.contourf(xx,yy,Z,cmap = plt.cm.cool)
    plt.scatter(X[:,0],X[:,1],c = Y, cmap = plt.cm.cool)
    plt.title(titles[i])
    
plt.show()
    