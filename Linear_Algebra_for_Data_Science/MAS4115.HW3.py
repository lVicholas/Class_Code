import random
import numpy

random.seed(0)
def generateMatrix():
    M = []
    for i in range(1,6):
        x = []
        for j in range(1,6):
            x.append(round(random.random(), 3))
        M.append(x)
    return M

def createDiagonal(eVal):
    M = []
    for i in range(len(eVal)):
        r = []
        for j in range(len(eVal)):
            if i == j:
                r.append(eVal[i])
            else:
                r.append(0)
        M.append(r)
    return M

M = generateMatrix()
print("Start Matrix")
for r in M:
    print(str(r))

s = numpy.linalg.eig(M)
D = createDiagonal(s[0])
Xinv = numpy.linalg.inv(s[1])
XAX = numpy.matmul(Xinv, numpy.matmul(M, s[1]))
DIFF = numpy.subtract(XAX, D)

for i in range(len(DIFF)):
    for j in range(len(DIFF[i])):
        DIFF[i][j] = format(DIFF[i][j], '.3g')

print("X^-1 x A x X - D")
for r in DIFF:
    print(str(r))