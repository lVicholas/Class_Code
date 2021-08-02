import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as pyplot
import winsound

# Part a)

def getPeriodogram(x):
    N = len(x)
    xh = np.fft.fft(x)

    P = [np.absolute(xh[0])**2]
    P.extend([2*np.absolute(xh[j])**2 for j in range(1,int(N/2))])
    P.append(np.absolute(xh[int(N/2)])**2)

    return N**-2 * np.array(P)

# Part b)
def g(t):
    return .7*np.sin(2*np.pi*50*t) + 2*np.sin(2*np.pi*20*t)

time = np.linspace(0,3,600)
x = list(map(g , time))

P = getPeriodogram(x)
F = np.array(list(range(int(len(x)/2)+1))) / 3

pyplot.plot(F, P)
pyplot.show()

# Part c)
file_name = "Resources\Wilhelm_scream.wav"
Fs, y = wavfile.read(file_name)

y = np.delete(y, y.shape[0]-1, axis=0)

y1 = y[:,0]
y2 = y[:,1]

P_1 = getPeriodogram(y1)
P_2 = getPeriodogram(y2)

F1 = np.array(list(range(int(len(y1)/2)+1))) / (len(y1) / Fs)
F2 = np.array(list(range(int(len(y2)/2)+1))) / (len(y2) / Fs)

pyplot.plot(F1, P_1)
pyplot.show()

pyplot.plot(F2, P_2)
pyplot.show()

# winsound.PlaySound(file_name, winsound.SND_FILENAME)