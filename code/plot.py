import numpy as np
import matplotlib.pyplot as plt
import sys


#discretizar espaco e tempo

X = np.arange(0,10,1)

argc = len(sys.argv)

maxF = []
minF = []
abcF = []

for i in range(1,argc):
    with open(sys.argv[i], 'r') as f:
        time = float(f.readline())
        numTest = int(f.readline())
        limit = f.readline().strip('\n')
        maxF.append(eval(f.readline()))
        limit = f.readline().strip('\n')
        minF.append(eval(f.readline()))
        limit = f.readline().strip('\n')
        abcF.append(eval(f.readline()))

    f.close()

fig, _ = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
i = 1
for f in maxF:
    plt.plot(X, f, str(i/argc))
    i+=1

plt.show()

fig, _ = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
i = 1
for f in minF:
    plt.plot(X, f, str(i/argc))
    i+=1

plt.show()

fig, _ = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
i = 1
for f in abcF:
    plt.plot(X, f, str(i/argc))
    i+=1

plt.show()







def plot_U(fun_u, sliders = False):
    #Plot
    fig, _ = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.axis((-lim/2,lim/2,-15,20))

    plt.plot(X,fun_u[0])
    plt.plot(X,fun_u[-1])
    plt.plot(X,np.ones(numNeu)*h)

    plt.show()

def plotBoundaries(P):
    min_P = [ 0 for i in range(numSteps)]
    med_P = [ 0 for i in range(numSteps)]
    max_P = [ 0 for i in range(numSteps)]

    for i in range(numSteps):
        min_P[i] = min([ P[j][i] for j in range(numTest)])
        med_P[i] = np.ma.average([ P[j][i] for j in range(numTest)])
        max_P[i] = max([ P[j][i] for j in range(numTest)])

    plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    plt.plot( np.arange(0,numSteps,1), min_P )
    plt.plot( np.arange(0,numSteps,1), med_P )
    plt.plot( np.arange(0,numSteps,1), max_P )

    plt.show()

