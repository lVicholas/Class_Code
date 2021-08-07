import pandas as pd
import numpy as np

def my_soft_max(x,l): 
    if x<0:
        return max(abs(x)-l,0)*-1
    else: return max(abs(x)-l,0)

# LASSO by cordinate descent
def lasso_descent(X, Y, a, tol):
    p=X.shape[1]
    B=np.ones((p,1))
    r=np.matrix(Y-(X@B))
    my_error=1
    Bprev=B.copy()
    while tol < my_error:
        Bprev=B.copy()
        for j in range(p):
            Xj=np.matrix(X[:,j])
            rj=r+Xj*float(B[j])
            rj_t=np.transpose(rj)
            Xj_t=np.transpose(Xj)
            Bp=float(my_soft_max(float(rj_t@Xj)/float(Xj_t@Xj),a))
            B[j]=[Bp]
            r=rj-(Xj*float(B[j]))
        my_error=np.linalg.norm(Bprev-B,ord=2)
    return np.transpose(B.copy()).tolist()[0]

p=X.shape[1]
B=[1]*p
r=(Y-np.transpose(X@B))
my_error=1
Bprev=B.copy()
Xj=X[:,0]
rj=r+Xj*float(B[0])
rj_t=np.transpose(rj).tolist()[0]
Xj_t=np.transpose(Xj).tolist()[0]
Bp=my_soft_max(float(rj_t@Xj)/float(Xj_t@Xj),a)

data=pd.read_csv('C:\local\Visual Studio 2019\PythonAnalyses\data\HW4P1.csv')
X=np.matrix(data.drop('y',axis=1))
Y=np.transpose(np.matrix(data['y'].tolist()))
B_lasso=lasso_descent(X,Y,.5,10**(-7))
B_lasso

lasso=Lasso(alpha=.5).fit(X,Y)
lasso.coef_

from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
data_s=pd.DataFrame(scaler.fit_transform(data))
data_s.head()
X=np.matrix(data_s.drop(52,axis=1))
Y=np.transpose(np.matrix(data_s[52].tolist()))

alpha_sequence=np.logspace(-5,2,100)
manual_lasso_coefs=[]
scikit_lasso_coefs=[]

for a in alpha_sequence:
    manual_lasso_coefs.append(lasso_descent(X,Y,a,10**(-7)))
    lasso_coef=Lasso(alpha=a).fit(X,Y).coef_
    scikit_lasso_coefs.append(lasso_coef)

ax=pyplot.gca()
ax.plot(alpha_sequence,manual_lasso_coefs)
ax.set_xscale('log')
pyplot.xlabel('alpha')
pyplot.ylabel('coefficient weight')
pyplot.title('LASSO coefficients vs alpha')
pyplot.axis('tight')
pyplot.show()

ax=pyplot.gca()
ax.plot(alpha_sequence,scikit_lasso_coefs)
ax.set_xscale('log')
pyplot.xlabel('alpha')
pyplot.ylabel('coefficient weight')
pyplot.title('LASSO coefficients vs alpha')
pyplot.axis('tight')
pyplot.show()

#####
train=pd.read_csv('C:\local\Visual Studio 2019\PythonAnalyses\data\HW4P2_train.csv')
test=pd.read_csv('C:\local\Visual Studio 2019\PythonAnalyses\data\HW4P2_test.csv')

import seaborn as sns
import matplotlib.pyplot as pyplot

train_cor_mat=train.corr()

sns.heatmap(train_cor_mat,cmap='hot')
pyplot.show()

pyplot.imshow(train_cor_mat)
pyplot.show()

from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSCanonical
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

np.random.seed(4)

scaler=StandardScaler()
X_train=pd.DataFrame(scaler.fit_transform(train.drop('y',axis=1)))
X_test=pd.DataFrame(scaler.fit_transform(test.drop('y',axis=1)))

lasso_mod_cv_plain=LassoCV().fit(train.drop('y',axis=1),train['y'])
mean_squared_error(lasso_mod_cv_plain.predict(test.drop('y',axis=1)),test['y'])

lasso_mod_cv=LassoCV(eps=.01,max_iter=1000,cv=10,tol=1e-5,normalize=True).fit(train.drop('y',axis=1),train['y'].tolist())
mean_squared_error(lasso_mod_cv.predict(X_test),test['y'])

ridge_mod_cv=RidgeCV(cv=10,normalize=True).fit(X_train,train['y'].tolist())
mean_squared_error(ridge_mod_cv.predict(X_test),test['y'])

pcr=make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())
pcr.fit(X_train, train.drop(''))

pca_95=PCA(n_components=.95,svd_solver='full').fit(train.drop('y',axis=1))


plsr=PLSCanonical()

# Get PCA cross-validation scores
def pca_pls_cv_scores(X,y):
    pca, pls = PCA(svd_solver='full'), PLSCanonical()
    pca_scores, pls_scores = [], []
    for ncomp in range(min(len(X.loc[0]),len(X.columns))):
        pca.n_components=ncomp
        pca_scores.append(np.mean(cross_val_score(pca,X,y)))
        pls.n_components=ncomp
        pls_scores.append(np.mean(cross_val_score(pls,X,y)))
    return pca_scores, pls_scores

pca_cv_scores, pls_cv_scores = pca_pls_cv_scores(train.drop('y',axis=1),train['y'].tolist())
pca_ncomp_cv=np.argsort(pca_cv_scores)[-1]+1
pls_ncomp_cv=np.argsort(pls_cv_scores)[-1]+1
pca_cv=PCA(n_components=pca_ncomp_cv,svd_solver='full').fit(train.drop('y',axis=1),train['y'].tolist())
pls_cv=PLSCanonical(n_components=pls_ncomp_cv).fit(train.drop('y',axis=1),train['y'].tolist())
