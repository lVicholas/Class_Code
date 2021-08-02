# #3 Python 3.8.5
import numpy as np

# Part d)
A = np.array([
    [1,1],
    [2,1],
    [0,1],
    [2,3]
    ])
b = np.array([[6],[0],[3],[9]])

x0 = np.array([[1],[1]])

def nablaPhi(x, A, AT, b):
    return 2*(AT @ (A @ x - b))

def iterativeDescent(x0, A, b, h):
    AT = np.transpose(A)
    for i in range(500):
        x0 = x0 - h * nablaPhi(x0, A, AT, b)
    return x0    

print("Part d)")
xn = iterativeDescent(x0, A, b, .05)
print(str(xn) + "\n")

# Part e)
x_star = np.array([[-4/3],[4]])

def toleranceDescent(x0, A, b, h, x_star):
    AT = np.transpose(A)
    i = 0
    while np.linalg.norm(x0 - x_star) >= 10**-3:
        x0 = x0 - h * nablaPhi(x0, A, AT, b)
        i += 1
    return x0, i

xn , n = toleranceDescent(x0, A, b, .01, x_star)
print("Part e)")
print("x_n: \n" + str(xn))
print("Norm of grad(phi(x_n)): " + str(round(np.linalg.norm(2 * np.transpose(A) @ (A @ xn - b)),6)))
print("n: " + str(n) + "\n")

# Part f)
print("Part f)")
H = [.06, .055, .05]
for h in H:
    xn, n = toleranceDescent(x0, A, b, h, x_star)
    print("For h = " + str(h))
    print("x_n:\n" + str(xn))
    print("Norm of grad(phi(x_n)): " + str(round(np.linalg.norm(2 * np.transpose(A) @ (A @ xn - b)),6)))
    print("Number of steps: " + str(n) + "\n")

# Part g)
H2 = np.linspace(.01,.1,100)
for h in H2:
    xn , n = toleranceDescent(x0, A, b, h, x_star)
    print("h = " + str(h) + ": " + 
          str(np.allclose(xn, x_star, atol=10**-3)) + ", n = " + str(n))