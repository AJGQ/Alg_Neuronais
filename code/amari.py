import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy import signal
from noise import *

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
eps = 0.05

T = 10
numSteps = 100
dt = T/numSteps

#definicao de funcoes

def fun_w(x):
    return A*np.exp(-b*np.abs(x))*(b*np.sin(np.abs(alfa*x)) + np.cos(alfa*x))

def calculate_U(b, met = "E_M_Normal", intgr = "FFT"):
    
    #random
    if met == "Normal_SC_Cos":
        noise = np.sqrt(eps) * np.array([dx * signal.fftconvolve(np.cos(X), 
						np.random.standard_normal((numNeu)),
						mode = 'same') for i in range (numSteps)])
    elif met == "Normal_TSC_Perlin":
        noise = eps * np.array([ np.array([pnoise2(j, i*0.01 + b, 20) for j in X]) for i in range(numSteps)])
    elif met == "E_M_Normal":
        noise = eps*np.random.standard_normal((numSteps,numNeu))
    elif met == "M_Normal":
        noise = (eps/2)*(np.random.standard_normal((numSteps,numNeu))**2 - dt )
    elif met == "E_M":
        noise = eps*np.random.random((numSteps,numNeu))
    elif met == "M":
        noise = (eps/2)*(np.random.random((numSteps,numNeu))**2 - dt )
    elif met == "DET":
        noise = np.zeros((numSteps,numNeu))

    #regra para integrar
    if intgr == "TRAPZ":
    	du = lambda t: dt*(-fun_u[t] +
    		       dx*np.trapz([[fun_w(x-X[y])*np.heaviside(fun_u[t,y]-h,1) for y in range(numNeu)] for x in X]) +
                       fun_S[t])
    elif intgr == "FFT":
        du = lambda t: dt*(-fun_u[t] +
        	       dx*signal.fftconvolve(fun_w(X),np.heaviside((fun_u[t] - h),1),mode='same') +
                       fun_S[t])

    #inicializar fun_u
    fun_u = np.zeros((numSteps,numNeu))
    fun_S = np.zeros((numSteps,numNeu))

    fun_S[0] = -0.5 + 20 * np.exp(-(X**2)/18)
    #for i in range(10):
    #    fun_S[numSteps//2 + i] = -0.5 + 20 * np.exp(-((X + 5)**2)/18)

    fun_u[0] = fun_S[0]

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

def plotBoundaries(P, numTest):
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

