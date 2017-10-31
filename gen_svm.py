from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import Data_load
X,Y=Data_load.load_data_wrapper()
X = preprocessing.scale(X)
pca = PCA(n_components=1000, svd_solver='full')
X=pca.fit_transform(X) 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
#print Y_train.shape,Y_test.shape
clf = svm.SVC(kernel='rbf',C=8)
clf.fit(X_train, Y_train)
Y_pred=clf.predict(X_test)
print accuracy_score(Y_test, Y_pred)
#Accuracy for KNN model 
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, Y_train) 
Y_pred=clf.predict(X_test)
print accuracy_score(Y_test, Y_pred)
