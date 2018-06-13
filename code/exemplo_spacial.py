from amari import *



'''
w_ex, w_in     = 4,  2
sig_ex, sig_in = 2,  3.5
g_in           = 0.5

kernel = w_ex*np.exp(-X**2/(2*sig_ex**2)) - w_in*np.exp(-X**2/(2*sig_in**2)) - g_in
'''

N1= np.arange(0,10,0.9)
N2= np.arange(0,10,0.9)
'''
fig, _ = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
plt.plot(N, [[pnoise2(i, j, 5) for i in N1] for j in N2])
plt.show()
'''
for l in [[pnoise2(i, j, 5) for i in N1] for j in N2]:
	print(l)

#fun_u = calculate_U("Normal_SC_Perlin", "FFT")

#plot_U(fun_u,True)



