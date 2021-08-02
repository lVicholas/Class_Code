# #3 Python 3.8.5
import matplotlib.pyplot as pyplot
import numpy as np

# Part a

def sigmoid(x):
    return (1 + np.exp(-x)) ** -1

def S(x):

    w1 = [5,5,2,2]
    w2 = [-1,1,-1,1]
    b  = [-3,-1,-6,-8] 

    S = 0
    for i in range(4):
        S = S + w2[i] * sigmoid(w1[i]*x + b[i])

    return S

# Part b
x = np.linspace(0,6,100)
pyplot.plot(x , list(map(S , x)))
pyplot.show()