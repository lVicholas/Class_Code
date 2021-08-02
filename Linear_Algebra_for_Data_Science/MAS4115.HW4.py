import numpy
def Nabla(x):
    n = [
        round(20*x[0]*x[1] - 10*x[0] - 4*x[0]*x[0]*x[0], 3),
        round(10*x[0]*x[0] - 8*x[1] - 8*x[1]*x[1]*x[1], 3)
        ]
    return n

def Hessian(x):
    H = [
        [ round(20*x[1] - 10 - 12*x[0]*x[0], 3), round(20*x[0], 3) ],
        [ round(20*x[0], 3), round(-8 - 24*x[1], 3)                ]
        ]
    return H

points = [ [0,0],
           [2.6442, 1.8984],
           [.8567, .6428]
         ]

print("Part A")
for i in range(len(points)):
    print("p" + str(i) + ": " + str(points[i]))
    print("Grad p" + str(i) + ": " + str(Nabla(points[i])))

print("\n")
print("Part B")
for i in range(len(points)):
    H = Hessian(points[i])
    E = numpy.linalg.eigvals(H)
    print("Eigenvectors of H of p" + str(i) + ": " + str(E))

    print("2nd derivative test conclusion: ", end='')
    if(numpy.sign(E[0]) == numpy.sign(E[1])):
        m = "Max" if E[0] < 0 else "Min"
        print(m)
    else:
        print("Saddle point")