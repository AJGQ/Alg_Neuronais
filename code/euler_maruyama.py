import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

lim = 100

#variaveis
h = 2.8997
A = 2
b = 0.08
alfa = np.pi/10

#discretizar espaco e tempo
dx = 0.5
lim = 100
X = np.arange(-lim/2,lim/2,dx)
numNeu = len(X)

T = 4
numSteps = 100
dt = T/numSteps

def fun_f(x):
    if(x < 0):
        return 0
    else:
        return 1

def fun_w(x):
    return A*np.exp(-b*np.abs(x))*(b*np.sin(np.abs(alfa*x)) + np.cos(alfa*x))

def calculate_U():

    #inicializar fun_u
    fun_u = np.zeros((numSteps,numNeu))

    fun_S = -0.5 + 20 * np.exp(-(X**2)/18)
    fun_u[0] = fun_S

    def Integral(x,t):
        return sum([dx*(fun_w(x-X[y])*fun_f(fun_u[t,y]-h)) for y in range(numNeu)])

    for i in range(1,numSteps):
        du = dt*(-fun_u[i-1] + Integral(X,i-1))
        fun_u[i] = du + fun_u[i-1]

    return fun_u

fun_u = calculate_U()
#Plot
fig, _ = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
plt.axis((-lim/2,lim/2,-15,20))

plt.plot(X,fun_u[0])
plt.plot(X,fun_u[-1] - h )
plt.plot(X,np.zeros(numNeu) - h)
'''
#Sliders
l, = plt.plot(X,fun_u[0])
axtemp = plt.axes([0.25, 0.1, 0.65, 0.03])

stemp = Slider(axtemp, 'Time', 0, T-dt, valinit=0)

def update(val):
    temp = stemp.val
    l.set_ydata(fun_u[int(temp/dt)])
    fig.canvas.draw_idle()

stemp.on_changed(update)
#
'''

plt.show()
#
