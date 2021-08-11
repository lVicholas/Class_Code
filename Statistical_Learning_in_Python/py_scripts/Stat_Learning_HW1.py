
import pandas as pd, numpy as np, matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate

data=pd.read_csv('C:\local\Visual Studio 2019\PythonAnalyses\data\HW1P2.csv')
# Predictors x1, x2, x1_random, ... x20_random
# Response is binary classification

np.random.seed(777)
train, test = train_test_split(data, test_size=.2)
# train, valid = train_test_split(train, test_size=.15)

# Known distibution of response
def prob(x1,x2):
    return norm.cdf(.5*x1-.4*x2)

# Classify based on known distribution
def simple_classify(x1,x2):
    y=[]
    for i in range(len(x1)):
        y.append(int(prob(x1[i],x2[i])>=.5))
    return y

def classification_error_rate(P,R):
    e=0
    for p,r in zip(P,R):
        if p != r: e+=1
    return e/len(P)

# Training error using known distibution
mean_squared_error(simple_classify(train['x1'].tolist(),train['x2'].tolist()),train['y'].tolist()) # .32875
mean_squared_error(simple_classify(test['x1'].tolist(),test['x2'].tolist()),test['y'].tolist()) # .285

np.random.seed(777)
noisy_training_error_rates, noisy_testing_error_rates = [], []
no_noise_training_error_rates, no_noise_testing_error_rates = [], []
for k in range(1,101):
    knn_noisy=KNeighborsClassifier(n_neighbors=k).fit(train.drop('y',axis=1),train['y'])
    knn_no_noise=KNeighborsClassifier(n_neighbors=k).fit(train[['x1','x2']],train['y'])
    noisy_training_error_rates.append(1-knn_noisy.score(train.drop('y',axis=1),train['y']))
    no_noise_training_error_rates.append(1-knn_no_noise.score(train[['x1','x2']],train['y']))
    noisy_testing_error_rates.append(1-knn_noisy.score(test.drop('y',axis=1),test['y']))
    no_noise_testing_error_rates.append(1-knn_no_noise.score(test[['x1','x2']],test['y'].tolist()))

# Get values of k with smallest error rates
np.argsort(testing_error_rates)[:5]+1 # k = 77, 81, 91, 83, 99
np.sort(testing_error_rates)[:5] # mse = .26, .265, .265, .265, .27

pyplot.plot(noisy_testing_error_rates, label='noisy_testing_error_rates')
pyplot.plot(noisy_training_error_rates, label='noisy_training_error_rates')
pyplot.plot(no_noise_testing_error_rates, label='no_noise_testing_error_rates')
pyplot.plot(no_noise_training_error_rates, label='no_noise_training_error_rates')
pyplot.title('Error rates vs k')
pyplot.xlabel('k for KNN')
pyplot.ylabel('Error rate')
pyplot.legend()
pyplot.show()

# Stock Market Data
data=pd.read_csv('C:\local\Visual Studio 2019\PythonAnalyses\data\HW1P3.csv')
np.random.seed(778)
train, test = train_test_split(data, test_size=.25)

training_error_rates, testing_error_rates = [], []
predictors=['Lag1','Lag2','Lag3','Lag4','Lag5','Volume','Today','y2003','y2004','y2005']
for k in range(1,101):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(train[predictors],train['Direction'].tolist())
    training_error_rates.append(1-knn.score(train[predictors],train['Direction']))
    testing_error_rates.append(1-knn.score(test[predictors],test['Direction']))

pyplot.plot(testing_error_rates, label='testing_error_rates')
pyplot.plot(training_error_rates, label='training_error_rates')
pyplot.title('Error rates vs k')
pyplot.xlabel('k for KNN')
pyplot.ylabel('Error rate')
pyplot.legend()
pyplot.show()


