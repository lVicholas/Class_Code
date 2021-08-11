def printMatrix(M):
    print("Matrix")
    for r in M:
        print(str(r))
    print("")

# Forward-substitution to solve matrix equation Lx=b
# where L is a lower-traingular matrix
def forwardSub(L, b):
    x = []
    for i in range(len(b)):
        x.append(b[i])
        for j in range(i):
            x[i] = x[i] - (L[i][j] * x[j])
        x[i] = x[i]/L[i][i]
    return x
    
# Create simple matrix
# Defined w/o numpy because we were instructed to not use numpy

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

x=forwardSub(L, b)
printMatrix(L)
print('Vector b\n'+str(b))
print('Result of forward substitution\n'+str(x))
