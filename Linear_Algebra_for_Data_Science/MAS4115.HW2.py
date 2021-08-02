def printMatrix(M):
    print("Matrix")
    for r in M:
        print(str(r))
    print("")

def forwardSub(L, b):
    x = []
    for i in range(len(b)):
        x.append(b[i])
        for j in range(i):
            x[i] = x[i] - (L[i][j] * x[j])
        x[i] = x[i]/L[i][i]
    printMatrix(L)
    print("Vector\n" + str(b) + "\n")
    print("Result of forward substitution\n" + str(x))
    return x
    

L = []

for i in range(1,6):
    r = []
    for j in range(1,6):
        if i >= j:
            r.append(i + j*j)
        else:
            r.append(0)
    L.append(r)

b = []
for i in range(1,6):
    b.append(i*i)

forwardSub(L, b)