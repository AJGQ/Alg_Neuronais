from amari import *

'''
N1= np.arange(0,10,0.9)
N2= np.arange(0,10,0.9)
fig, _ = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
plt.plot(N, [[pnoise2(i, j, 5) for i in N1] for j in N2])
plt.show()
for l in [[pnoise2(i, j, 5) for i in N1] for j in N2]:
	print(l)

eps = 5.5
fun_u = calculate_U("Normal_SC_Cos", "FFT")

plot_U(fun_u,True)
'''

fig, _ = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

plt.plot(X,fun_w(X))
plt.show()



