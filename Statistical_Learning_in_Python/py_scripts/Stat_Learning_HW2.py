import pandas as pd
import matplotlib
import matplotlib.pyplot as pyplot

train=pd.read_csv('C:\local\Visual Studio 2019\PythonAnalyses\data\HW2P2_train.csv')
test=pd.read_csv('C:\local\Visual Studio 2019\PythonAnalyses\data\HW2P2_test.csv')

import itertools, numpy as np
x1seq=list(np.linspace(min(train['X1'].tolist())-2,max(train['X1'].tolist())+2,100))
x2seq=list(np.linspace(min(train['X2'].tolist())-2,max(train['X2'].tolist())+2,100))
grid=pd.DataFrame(list(itertools.product(x1seq,x2seq)))
grid.columns=['X1','X2']

positives=train[train['Y']==1]
negatives=train[train['Y']==0]
  
# Logistic regression and LDA models ###################################################################

from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression(random_state=3).fit(train.drop('Y',axis=1),train['Y'].tolist())
grid_log_predict=list(log_model.predict(grid))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_model=LinearDiscriminantAnalysis().fit(train.drop('Y',axis=1),train['Y'].tolist())
grid_lda_predict=list(lda_model.predict(grid))

fig, axs = pyplot.subplots(1,2)
axs[0].scatter(grid['X1'].tolist(),grid['X2'].tolist(),c=grid_log_predict, cmap=matplotlib.colors.ListedColormap(['green','cyan']), alpha=.2,s=3)
axs[0].scatter(positives['X1'].tolist(),positives['X2'].tolist(),c='white',label='Y=1',alpha=.9)
axs[0].scatter(negatives['X1'].tolist(),negatives['X2'].tolist(),c='black',label='Y=0',alpha=.9)
axs[0].legend()
axs[0].set_ylabel('X2')
axs[0].set_xlabel('X1')
axs[0].set_title('Logistic Regression')

axs[1].scatter(grid['X1'].tolist(),grid['X2'].tolist(),c=grid_lda_predict, cmap=matplotlib.colors.ListedColormap(['green','cyan']), alpha=.2,s=3)
axs[1].scatter(positives['X1'].tolist(),positives['X2'].tolist(),c='white',label='Y=1',alpha=.9)
axs[1].scatter(negatives['X1'].tolist(),negatives['X2'].tolist(),c='black',label='Y=0',alpha=.9)
axs[1].legend()
axs[1].set_ylabel('X2')
axs[1].set_xlabel('X1')
axs[1].set_title('LDA')

axs[0].set_box_aspect(1)
axs[1].set_box_aspect(1)

for ax in axs.flat:
    ax.label_outer()

pyplot.show()

print('Logistic regression training error')
print(str(1-log_model.score(train.drop('Y',axis=1),train['Y'])))

print('Logistic regression testing error')
print(str(1-log_model.score(test.drop('Y',axis=1),test['Y'])))

print('LDA training error')
print(str(1-lda_model.score(train.drop('Y',axis=1),train['Y'])))

print('LDA testing error')
print(str(1-lda_model.score(test.drop('Y',axis=1),test['Y'])))

# QDA model ##########################################################################################
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda_model=QuadraticDiscriminantAnalysis().fit(train.drop('Y',axis=1),train['Y'].tolist())
grid_qda_predict=list(qda_model.predict(grid))

pyplot.style.use('ggplot')
pyplot.scatter(grid['X1'].tolist(),grid['X2'].tolist(),c=grid_qda_predict, cmap=matplotlib.colors.ListedColormap(['green','cyan']), alpha=.2,s=3)
pyplot.scatter(positives['X1'].tolist(),positives['X2'].tolist(),c='white',label='Y=1',alpha=.9)
pyplot.scatter(negatives['X1'].tolist(),negatives['X2'].tolist(),c='black',label='Y=0',alpha=.9)
pyplot.legend()
pyplot.ylabel('X2')
pyplot.xlabel('X1')
pyplot.title('QDA')
pyplot.show()

print('QDA training error')
print(str(1-qda_model.score(train.drop('Y',axis=1),train['Y'])))

print('QDA testing error')
print(str(1-qda_model.score(test.drop('Y',axis=1),test['Y'])))

# Support Vector Machines with radial kernel and varying cost ####################################
from sklearn.svm import SVC

c_seq=[.5,1,5,10,20,50,100,500]
svc_training_predictions, svc_gid_predictions = [], []

fig1, axs1 = pyplot.subplots(2,2)
fig2, axs2 = pyplot.subplots(2,2)

def grid_plot(model, train, grid, axs, colors, _alpha, s, title):
    
    grid_predictions=model.predict(grid).tolist() 
    positives=train[train['Y']==1]
    negatives=train[train['Y']==0]

    axs.scatter(grid['X1'].tolist(),grid['X2'].tolist(),c=grid_predictions,cmap=matplotlib.colors.ListedColormap(['green','cyan']),alpha=_alpha,s=3)
    axs.scatter(positives['X1'].tolist(),positives['X2'].tolist(),c='white',label='Y = 1',alpha=.9)
    axs.scatter(negatives['X1'].tolist(),negatives['X2'].tolist(),c='black',label='Y = 0',alpha=.9)
    axs.set_ylabel('X2')
    axs.set_xlabel('X1')
    axs.set_title(title)

svc_training_errors, svc_testing_errors = [], []
for i, c in enumerate(c_seq):
    svc=SVC(C=c).fit(train.drop('Y',axis=1),train['Y'])
        
    row = int(i/2)
    col = int(i % 2 == 1)

    if i <= 3:
        grid_plot(svc,train,grid,axs1[row,col],['blue','cyan'],.2,3,'C = '+str(c))
        axs1[row,col].label_outer()
    else: 
        grid_plot(svc,train,grid,axs2[row-2,col],['blue','cyan'],.2,3,'C = '+str(c))
        axs2[row-2,col].label_outer()

    if(i==0):
        axs1[row,col].legend()
    if(i==4):
        axs2[row-2,col].legend()

    svc_training_errors.append(1-svc.score(train.drop('Y',axis=1),train['Y']))
    svc_testing_errors.append(1-svc.score(test.drop('Y',axis=1),test['Y']))

pyplot.show()

for i, c in enumerate(c_seq):
    print('SVC with C = '+str(c))
    print('Training error: ' + str(svc_training_errors[i]))
    print('Testing error: '+str(svc_testing_errors[i]))


from sklearn import GridSearchCV
np.random.seed(17)
c_seq=np.linspace(.1, 3.1, 101)

svc_cv_on_c=GridSearchCV(SVC(),{'C':list(c_seq)},cv=10)
svc_cv_on_c.fit(train.drop('Y',axis=1),train['Y'])

fig, ax = pyplot.subplots()
ax.plot(c_seq,list(1-svc_cv_on_c.cv_results_['mean_test_score']))
c_rank_1=list(svc_cv_on_c.cv_results_['rank_test_score']).index(1)
min_error_cv_on_c=1-svc_cv_on_c.cv_results_['mean_test_score'][c_rank_1]

ax.annotate('C = '+str(round(c_seq[c_rank_1],4))+'\n Error = '+str(round(min_error_cv_on_c,4)), 
            xy=(c_seq[c_rank_1],min_error_cv_on_c), 
            xytext=(c_seq[c_rank_1], min_error_cv_on_c*1.05),
            arrowprops=dict(facecolor='black', shrink=0.05)
            )
ax.set_ylabel('Testing error')
ax.set_xlabel('C')
ax.set_title('Testing error vs C')
pyplot.show()

# SVMs with varying gamma ##########################################################################
g_seq=[.001,.05,.01,.1,1,5,10]
svc_training_errors, svc_testing_errors = [], []

fig, axs1 = pyplot.subplots(2,2)
fig, axs2 = pyplot.subplots(2,2)

for i, g in enumerate(g_seq):
    svc=SVC(gamma=g).fit(train.drop('Y',axis=1),train['Y'])
        
    row = int(i/2)
    col = int(i % 2 == 1)

    if i <= 3:
        grid_plot(svc,train,grid,axs1[row,col],['blue','cyan'],.2,3,'G = '+str(g))
        axs1[row,col].label_outer()
    else: 
        grid_plot(svc,train,grid,axs2[row-2,col],['blue','cyan'],.2,3,'G = '+str(g))
        axs2[row-2,col].label_outer()

    if(i==0):
        axs1[row,col].legend()
    if(i==4):
        axs2[row-2,col].legend()

    svc_training_errors.append(1-svc.score(train.drop('Y',axis=1),train['Y']))
    svc_testing_errors.append(1-svc.score(test.drop('Y',axis=1),test['Y']))

pyplot.show()

np.random.seed(17)
g_seq=np.linspace(.01,2.01,101)

svc_cv_on_g=GridSearchCV(SVC(),{'gamma':list(g_seq)},cv=10)
svc_cv_on_g.fit(train.drop('Y',axis=1),train['Y'])

fig, ax = pyplot.subplots()
ax.plot(g_seq,list(1-svc_cv_on_g.cv_results_['mean_test_score']))
g_rank_1=list(svc_cv_on_g.cv_results_['rank_test_score']).index(1)
min_error_cv_on_g=1-svc_cv_on_g.cv_results_['mean_test_score'][g_rank_1]

ax.annotate('G = '+str(round(g_seq[g_rank_1],4))+'\n Error = '+str(round(min_error_cv_on_g,4)), 
            xy=(g_seq[c_rank_1],min_error_cv_on_g), 
            xytext=(g_seq[g_rank_1], min_error_cv_on_g*1.05),
            arrowprops=dict(facecolor='black', shrink=0.05)
            )
ax.set_ylabel('Testing error')
ax.set_xlabel('G')
ax.set_title('Testing error vs G')
pyplot.show()

# CV on C and Gamma in SVC #########################################################################
np.random.seed(17)

paramters={'C':list(c_seq),'gamma':list(g_seq)}
svc_cv=GridSearchCV(SVC(),paramters)
svc_cv.fit(train.drop('Y',axis=1),train['Y'].tolist())
grid_svc_predict=list(svc_cv.best_estimator_.predict(grid))

fig, ax = pyplot.subplots()
svc_cv_best_model_plot_title='C = '+str(svc_cv.best_estimator_.C)+' G = '+str(round(svc_cv.best_estimator_.gamma,3))
grid_plot(svc_cv.best_estimator_,train,grid,ax,['green','cyan'],.3,3,svc_cv_best_model_plot_title)
pyplot.text(-4,4,'Testing Error = '+str(1-svc_cv.best_estimator_.score(test.drop('Y',axis=1),test['Y'])))
ax.legend()
pyplot.show()

# Ternary classification #################################################################################
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
