import numpy as np

# Part a
A = []
b = []
for i in range(1,11):
    b.append(i)
    r = []
    for j in range(1,6):
        r.append(1 / np.sin(i + j - 1))
    A.append(r)

A = np.array(A)
b = np.array(b)

# Part b
U , sigma , VT = np.linalg.svd(A , full_matrices=0)

U = np.array(U)
simga = np.array(sigma)
VT = np.array(VT)

V = np.transpose(VT)
sigInv = np.diag(1 / sigma)
UT = np.transpose(U)

Aplus1 = V @ sigInv @ UT
x1 = Aplus1 @ b

print("Part b)\nx: " + str(x1))

# Part c
Q , R = np.linalg.qr(A)

Aplus2 = np.linalg.inv(R) @ np.transpose(Q)
x2 = Aplus2 @ b

print("Part c)\nx: " + str(x2))