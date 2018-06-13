from amari import *

#fun_u = calculate_U("M_Normal")
numTest = 10
M = [[ 0 for i in range(numSteps)] for j in range(numTest)]

m = [[ 0 for i in range(numSteps)] for j in range(numTest)]

X_M = [[ 0 for i in range(numSteps)] for j in range(numTest)]

max_U = [ -100 for x in X]

med_U = [ 0 for x in X]

min_U = [ 100 for x in X]

for i in range(numTest):
    fun_u = calculate_U(i, met = "Normal_TSC_Perlin",intgr = "FFT")
    for x in range(numNeu):
        max_U[x] = max(max_U[x],fun_u[-1][x])
        med_U[x] = ((med_U[x]*i) + fun_u[-1][x])/(i+1)
        min_U[x] = min(min_U[x],fun_u[-1][x])
    for j in range(numSteps):
        M[i][j] = max(fun_u[j])
        X_M[i][j] = X[np.argmax(fun_u[j])]
        m[i][j] = min(fun_u[j])

print(time.time() - iniTime)
#print(numTest)
#print("Maximos")
#print (M)
#print("Minimos")
#print (m)
#print("Abcissas dos Maximos")
#print (X_M)
#print("Maximos de u")
#print (max_U)
#print("Medios de u")
#print (med_U)
#print("Minimos de u")
#print (min_U)
plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

plt.plot( X, max_U )
plt.plot( X, med_U )
plt.plot( X, min_U )

plt.show()
plotBoundaries(M  , numTest)
plotBoundaries(m  , numTest)
plotBoundaries(X_M, numTest)
#plot_U(fun_u, True)
