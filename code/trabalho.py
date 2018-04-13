import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons



#variaveis
h = 2.8997
A = 2
b = 0.08
alfa = np.pi/10

#discretizar espaco e tempo
dx = 0.1
lim = 100
X = np.arange(-lim/2,lim/2,dx)
numNeu = len(X)
eps = 0.1

T = 4
numSteps = 100
dt = T/numSteps

#definicao de funcoes
def fun_f(x):
    if(x < 0):
        return 0
    else:
        return 1

def fun_w(x):
    return A*np.exp(-b*np.abs(x))*(b*np.sin(np.abs(alfa*x)) + np.cos(alfa*x))

def calculate_U(met = "E_M"):
    #random
    if met == "E_M":
        noise = eps*np.random.random((numSteps,numNeu))
    elif met == "M":
        noise = (eps/2)*(np.random.random((numSteps,numNeu))**2 - dt )
    #inicializar fun_u
    fun_u = np.zeros((numSteps,numNeu))

    fun_S = -0.5 + 20 * np.exp(-(X**2)/18)
    fun_u[0] = fun_S

    def Integral(x,t):
        return sum([(dx/2)*(fun_w(x-X[y])*fun_f(fun_u[t,y]-h)+ fun_w(x-X[y+1])*fun_f(fun_u[t,y+1]-h)) for y in range(numNeu-1)])

    for i in range(1,numSteps):
        du = dt*(-fun_u[i-1] + Integral(X,i-1))
        fun_u[i] = du + fun_u[i-1] + noise[i]

    return fun_u

def plot_U(fun_u, sliders = False):
    #Plot
    fig, _ = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.axis((-lim/2,lim/2,-15,20))

    plt.plot(X,fun_u[0])
    plt.plot(X,fun_u[-1] - h )
    plt.plot(X,np.zeros(numNeu) - h)
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

numTest = 5
M = [[ 0 for i in range(numSteps)] for j in range(numTest)]
med_M = [ 0 for i in range(numSteps)]
max_M = [ 0 for i in range(numSteps)]
min_M = [ 0 for i in range(numSteps)]

m = [[ 0 for i in range(numSteps)] for j in range(numTest)]
med_m = [ 0 for i in range(numSteps)]
max_m = [ 0 for i in range(numSteps)]
min_m = [ 0 for i in range(numSteps)]

for i in range(numTest):
    fun_u = calculate_U()
    if i == 0:
        plot_U(fun_u,True)
    for j in range(numSteps):
        M[i][j] = max(fun_u[j])
        m[i][j] = min(fun_u[j])

for i in range(numSteps):
    med_M[i] = sum([ M[j][i] for j in range(numTest)])/numTest
    max_M[i] = max([ M[j][i] for j in range(numTest)])
    min_M[i] = min([ M[j][i] for j in range(numTest)])

    med_m[i] = sum([ m[j][i] for j in range(numTest)])/numTest
    max_m[i] = max([ m[j][i] for j in range(numTest)])
    min_m[i] = min([ m[j][i] for j in range(numTest)])

#Plot - Maximos
plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
plt.axis((0,numSteps,0,50))

plt.plot( np.arange(0,numSteps,1), max_M )
plt.plot( np.arange(0,numSteps,1), med_M )
plt.plot( np.arange(0,numSteps,1), min_M )

plt.show()

#Plot - Minimos
plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
plt.axis((0,numSteps,-20,10))

plt.plot( np.arange(0,numSteps,1), max_m )
plt.plot( np.arange(0,numSteps,1), med_m )
plt.plot( np.arange(0,numSteps,1), min_m )

plt.show()
'''
print("\n\n ------ Máximos-------\n\n")
#print("Máximos:\n", M)
print("\n\nMedia dos máximos:\n",med_M)
print("\n\nMáximo dos máximos:\n",max_M)
print("\n\nMínimo dos máximos:\n",min_M)
print("\n\n ------ Mínimos-------\n\n")
#print("Mínimos:\n", m)
print("\n\nMedia dos mínimos:\n",med_m)
print("\n\nMáximo dos mínimos:\n",max_m)
print("\n\nMínimo dos mínimos:\n",min_m)
'''

#plot_U(fun_u, True)
