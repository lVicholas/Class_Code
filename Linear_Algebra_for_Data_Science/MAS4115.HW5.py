import random, numpy, statistics, matplotlib.pyplot as pyplot

random.seed(0)

def generateMatrix():
    M = []
    for i in range(1,10):
        r = []
        for j in range(1,10):
            r.append(round(random.random(), 3))
        M.append(r)
    return M

C = []

for i in range(1,100):
    Ai = generateMatrix()
    sigma = numpy.linalg.svd(Ai)[1]
    Ci = sigma[0] / sigma[len(sigma)-1]
    C.append(Ci)

mu = statistics.mean(C)
sd = statistics.stdev(C)

print("Mean of Ci: " + str(mu))
print("Std. Dev of Ci: " + str(sd))

pyplot.plot(C)
pyplot.show()