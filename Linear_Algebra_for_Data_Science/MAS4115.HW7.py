import numpy as np
import random
import statistics as stat

random.seed(0)

print("Part b")
H = []
for i in range(1,6):
    r = [1 / (i + j - 1) for j in range(1,6)]
    H.append(r)
H = np.array(H)

print("Hilbert Matrix")
for r in H:
    print(str(r))

# Part c
U , S , VT = np.linalg.svd(H)

print("\nPart d")
k2 = max(S) / min(S)
print("k2: " + str(k2))

# Part e
def randomUnitVector():
    v = np.array([random.randrange(-100000 , 100000) for i in range(5)])
    return [x / np.linalg.norm(v) for x in v]

def getKxy(x , y , H):
    return np.linalg.norm((H @ y)) / np.linalg.norm((H @ x))

def partE(H):
    K = []
    for i in range(10):
        x = np.array(randomUnitVector())
        y = np.array(randomUnitVector())
        K.append(getKxy(x , y , H))
    return max(K) , min(K) , stat.mean(K) , stat.stdev(K)

print("\nPart e")
lab = ["Max: " , "Min: " , "Mean: " , "SD: "]
dat = partE(H)
for i in range(4):
    print(lab[i] + str(dat[i]))

print("\nPart f")
k = getKxy(VT[4] , VT[0] , H)
print("K(v_5 , v_1): " + str(k))