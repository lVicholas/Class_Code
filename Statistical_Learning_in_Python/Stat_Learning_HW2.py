import pandas as pd

train=pd.read_csv('C:\local\Visual Studio 2019\PythonAnalyses\data\HW2P2_train.csv')
test=pd.read_csv('C:\local\Visual Studio 2019\PythonAnalyses\data\HW2P2_test.csv')

import itertools, numpy as np
x1seq=list(np.linspace(min(train['X1'].tolist())-2,max(train['X1'].tolist())+2,100))
x2seq=list(np.linspace(min(train['X2'].tolist())-2,max(train['X2'].tolist())+2,100))
grid=pd.DataFrame(list(itertools.product(x1seq,x2seq)))
grid.columns=['X1','X2']

from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression(random_state=3).fit(train.drop('Y',axis=1),train['Y'].tolist())
grid_log_predict=list(log_model.predict(grid))

import matplotlib
import matplotlib.pyplot as pyplot

pyplot.scatter(grid['X1'].tolist(),grid['X2'].tolist(),c=grid_log_predict,cmap=matplotlib.colors.ListedColormap(['orange','blue']))
pyplot.ylabel('X2')
pyplot.xlabel('X1')
pyplot.title('GLM Classification Regions')
pyplot.show()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_model=LinearDiscriminantAnalysis().fit(train.drop('Y',axis=1),train['Y'].tolist())
grid_lda_predict=list(lda_model.predict(grid))

pyplot.scatter(grid['X1'].tolist(),grid['X2'].tolist(),c=grid_lda_predict,cmap=matplotlib.colors.ListedColormap(['orange','blue']))
pyplot.ylabel('X2')
pyplot.xlabel('X1')
pyplot.title('LDA Classification Regions')
pyplot.show()

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda_model=QuadraticDiscriminantAnalysis().fit(train.drop('Y',axis=1),train['Y'].tolist())
grid_qda_predict=list(qda_model.predict(grid))

pyplot.scatter(grid['X1'].tolist(), grid['X2'].tolist(), c=grid_qda_predict, cmap=matplotlib.colors.ListedColormap(['orange','blue']))
pyplot.ylabel('X2')
pyplot.xlabel('X1')
pyplot.title('QDA Classification Regions')
pyplot.show()

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svc_model=SVC()
svc_model.fit(train.drop('Y',axis=1),train['Y'].tolist())
paramters={'C':[1,10],'gamma':[.001, 10]}
svc_cv=GridSearchCV(svc_model,paramters)
svc_cv.fit(train.drop('Y',axis=1),train['Y'].tolist())
svc_cv.best_estimator_

grid_svc_predict=list(svc_model.predict(grid))
pyplot.scatter(grid['X1'].tolist(),grid['X2'].tolist(),c=grid_svc_predict,cmap=matplotlib.colors.ListedColormap(['blue','orange']))
pyplot.ylabel('X2')
pyplot.xlabel('X1')
pyplot.title('SVM Classification Regions')
pyplot.show()

train=pd.read_csv('C:\local\Visual Studio 2019\PythonAnalyses\data\HW2P3_train.csv')
test=pd.read_csv('C:\local\Visual Studio 2019\PythonAnalyses\data\HW2P3_test.csv')

x1seq=list(np.linspace(min(train['X1'].tolist())-2,max(train['X1'].tolist())+2,100))
x2seq=list(np.linspace(min(train['X2'].tolist())-2,max(train['X2'].tolist())+2,100))
grid=pd.DataFrame(list(itertools.product(x1seq,x2seq)))
grid.columns=['X1','X2']

lda_model=LinearDiscriminantAnalysis().fit(train.drop('Y',axis=1),train['Y'].tolist())
grid_lda_predict=list(lda_model.predict(grid))

pyplot.scatter(grid['X1'].tolist(),grid['X2'].tolist(),c=grid_lda_predict,cmap=matplotlib.colors.ListedColormap(['orange','blue','red','green']))
pyplot.ylabel('X2')
pyplot.xlabel('X1')
pyplot.title('LDA Classification Regions')
pyplot.show()

qda_model=QuadraticDiscriminantAnalysis().fit(train.drop('Y',axis=1),train['Y'].tolist())
grid_qda_predict=list(qda_model.predict(grid))

pyplot.scatter(grid['X1'].tolist(),grid['X2'].tolist(),c=grid_qda_predict,cmap=matplotlib.colors.ListedColormap(['orange','blue','red','green']))
pyplot.ylabel('X2')
pyplot.xlabel('X1')
pyplot.title('QDA Classification Regions')
pyplot.show()

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svc_model=SVC()
svc_model.fit(train.drop('Y',axis=1),train['Y'].tolist())
paramters={'C':[1,10],'gamma':[.001, 10]}
svc_cv=GridSearchCV(svc_model,paramters)
svc_cv.fit(train.drop('Y',axis=1),train['Y'].tolist())
svc_cv.best_estimator_

grid_svc_predict=list(svc_model.predict(grid))
pyplot.scatter(grid['X1'].tolist(),grid['X2'].tolist(),c=grid_svc_predict,cmap=matplotlib.colors.ListedColormap(['blue','orange']))
pyplot.ylabel('X2')
pyplot.xlabel('X1')
pyplot.title('SVM Classification Regions')
pyplot.show()
