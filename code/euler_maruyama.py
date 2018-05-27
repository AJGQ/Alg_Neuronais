import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy import signal

iniTime = time.time()
#variaveis
h = 2.8997
A = 2
b = 0.08
alfa = np.pi/10

#discretizar espaco e tempo
dx = 0.005
lim = 100
X = np.arange(-lim/2,lim/2,dx)
numNeu = len(X)
eps = 0.005

T = 10
numSteps = 100
dt = T/numSteps

#print(dx)
#print(dt)

#definicao de funcoes

def fun_w(x):
    return A*np.exp(-b*np.abs(x))*(b*np.sin(np.abs(alfa*x)) + np.cos(alfa*x))

def calculate_U(met = "DET", intgr = "TRAPZ"):
    #random
    if met == "E_M":
        noise = eps*np.random.random((numSteps,numNeu))
    elif met == "M":
        noise = (eps/2)*(np.random.random((numSteps,numNeu))**2 - dt )
    elif met == "DET":
        noise = np.zeros((numSteps,numNeu))

    #regra para integrar
    if intgr == "TRAPZ":
    	du = lambda t: dt*(-fun_u[t] +
    		       dx*np.trapz([[fun_w(x-X[y])*np.heaviside(fun_u[t,y]-h,1) for y in range(numNeu)] for x in X]))
    elif intgr == "FFT":
        du = lambda t: dt*(-fun_u[t] +
        	       dx*signal.fftconvolve(fun_w(X),np.heaviside((fun_u[t] - h),1),mode='same'))

    #inicializar fun_u
    fun_u = np.zeros((numSteps,numNeu))

    fun_S = -0.5 + 20 * np.exp(-(X**2)/18)
    fun_u[0] = fun_S

    for i in range(1,numSteps):
    	fun_u[i] = du(i-1) + fun_u[i-1] + noise[i]

    return fun_u

def plot_U(fun_u, sliders = False):
    #Plot
    fig, _ = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.axis((-lim/2,lim/2,-15,20))

    plt.plot(X,fun_u[0])
    plt.plot(X,fun_u[-1])
    plt.plot(X,np.ones(numNeu)*h)
    if(sliders):
        #Sliders
        l, = plt.plot(X,fun_u[0])
        axtemp = plt.axes([0.25, 0.1, 0.65, 0.03])

        stemp = Slider(axtemp, 'Time', 0, T-dt, valinit=0)

        def update(val):
            temp = stemp.val
            l.set_ydata(fun_u[int(temp/dt)])
            fig.canvas.draw_idle()

        stemp.on_changed(update)

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

#fun_u = calculate_U()
numTest = 10
M = [[ 0 for i in range(numSteps)] for j in range(numTest)]

m = [[ 0 for i in range(numSteps)] for j in range(numTest)]

X_M = [[ 0 for i in range(numSteps)] for j in range(numTest)]

for i in range(numTest):
    fun_u = calculate_U("E_M","FFT")
    for j in range(numSteps):
        M[i][j] = max(fun_u[j])
        X_M[i][j] = X[np.argmax(fun_u[j])]
        m[i][j] = min(fun_u[j])

print(time.time() - iniTime)
print(numTest)
print("Maximos")
print (M)
print("Minimos")
print (m)
print("Abcissas dos Maximos")
print (X_M)

#plotBoundaries(M)
#plotBoundaries(m)
#plotBoundaries(X_M)
plot_U(fun_u, True)
