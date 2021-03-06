import numpy as np
import matplotlib.pyplot as plt
import sys


#discretizar espaco e tempo
dx = 0.005
lim = 100
X = np.arange(-lim/2,lim/2,dx)
numNeu = len(X)
eps = 0.010

T = 10
numSteps = 100
dt = T/numSteps
N=np.arange(0,T,dt)

argc = len(sys.argv)

maxF = []
minF = []
abcF = []

med_U = []

for i in range(1,argc):
    with open(sys.argv[i], 'r') as f:
        time = float(f.readline())
        numTest = int(f.readline())
        limit = f.readline()
        maxF.append(eval(f.readline()))
        limit = f.readline()
        minF.append(eval(f.readline()))
        limit = f.readline()
        abcF.append(eval(f.readline()))

        limit = f.readline()
        med_U.append(eval(f.readline()))

    f.close()

fig, _ = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
i = 1
for f in maxF:
    plt.plot(N, f, str(i/argc))
    i+=1

plt.show()

fig, _ = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
i = 1
for f in minF:
    plt.plot(N, f, str(i/argc))
    i+=1

plt.show()

fig, _ = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
i = 1
for f in abcF:
    plt.plot(N, f, str(i/argc))
    i+=1

plt.show()

fig, _ = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
i = 1
for f in med_U:
    plt.plot(X, f, str(i/argc))
    i+=1

plt.show()



