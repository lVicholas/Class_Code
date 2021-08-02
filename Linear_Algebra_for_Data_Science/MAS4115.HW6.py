import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.image as img

def toGrayScale(color):
    return np.dot(color[...,:3], [.2989, .587, .114])

def getA_i(U, sigma, Q):
    rank1s = []
    for i in range(len(sigma)):
        R = np.diag(sigma)[i][i] * np.array(np.outer(U[:,i], Q[i,:]))
        rank1s.append(R)
    A_i = []
    A_i.append(rank1s[0])
    for i in range(1, len(rank1s)):
        A_i.append(rank1s[i] + A_i[i-1])
    return A_i

# Part A
pic1 = img.imread('Python\Resources\SonOfMan.jpg') # 374 x 266
A = np.array(pic1, dtype=float) 
A = toGrayScale(A)
pyplot.imshow(A, cmap='gray')
pyplot.show()
k = min(len(A), len(A[0])) # k = 266

# Part B
U , sigma, Q = np.linalg.svd(A, full_matrices=False)
    
# Part C
pyplot.plot(sigma[0:k])
pyplot.show()

# Part D
A_i = np.array(getA_i(U, sigma, Q))
Max = -1

# Part E

pyplot.imshow(A_i[7] , cmap='gray')
pyplot.show()

# Anything after A_5 is highly recognizable